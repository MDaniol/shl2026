#!/usr/bin/env python3
"""Convert SHL raw `.txt` channel files into one Parquet per (split, location).

The SHL Challenge ships each ``(split, location)`` as 9 channel files (and, for
train/validation, a ``Label.txt``), each a whitespace-separated matrix of shape
``(n_frames, 500)`` — 5 s @ 100 Hz windows. Re-parsing ~9 GiB of text on every
load is untenable, so we parse **once** into a columnar Parquet that
`shl2026`'s data loader can memory-map.

Input may be a **`.zip`** (read in place, no unpack) or an unpacked directory.
A single archive may hold several locations:

    SHL-2026-Train_Bag.zip   -> train/Bag/*.txt                 (1 location, labelled)
    SHL-2026-Validation.zip  -> validation/{Bag,Hand,Hips,Torso}/*.txt  (4, labelled)
    SHL-2026-Test.zip        -> test/*.txt                       (no location, UNLABELLED)

Output:  <out>/<split>/<location>.parquet   (test -> <out>/test/all.parquet)

Schema: one row per frame; nine ``list<float32>`` channel columns (length 500)
plus, when present, a ``label`` ``list<int8>`` column of per-sample labels.
Channel order, window size and sampling rate are recorded in the Parquet
key-value metadata so the loader is self-describing.

Streaming: files are read in lock-step row batches and flushed to Parquet row
groups, so peak memory is ``batch_size x 4500`` floats regardless of file size.

    # whole archive (every location inside), straight from the zip:
    python scripts/raw_to_parquet.py --src dataset/SHL-2026-Validation.zip --out dataset_parquet
    # a quick correctness sample (first N frames of each location):
    python scripts/raw_to_parquet.py --src dataset/SHL-2026-Test.zip --out /tmp/pq --limit 200

Channels are named ``Mag_*`` to match the real archives (NOT ``Magn_*`` as some
docs/params say).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Canonical channel order, matching the actual SHL archive file names.
CHANNELS = (
    "Acc_x", "Acc_y", "Acc_z",
    "Gyr_x", "Gyr_y", "Gyr_z",
    "Mag_x", "Mag_y", "Mag_z",
)
WINDOW_SAMPLES = 500
SAMPLING_RATE_HZ = 100
LABEL_FILE = "Label.txt"


@dataclass(frozen=True)
class Group:
    """One (split, location) worth of channel/label files within a source."""

    split: str
    location: str
    members: dict[str, str]  # "Acc_x".."Mag_z"[, "label"] -> member path / abs path
    has_labels: bool


def discover_groups(src: Path) -> tuple[Path | None, list[Group]]:
    """Find every (split, location) group in a zip or unpacked directory.

    Returns ``(zip_path_or_None, groups)``; ``zip_path`` is non-None when the
    source is a zip (so readers stream members instead of opening files).
    """
    def make(prefix_parts: list[str], member_of) -> Group:
        split = prefix_parts[0]
        location = prefix_parts[1] if len(prefix_parts) > 1 else "all"
        members = {ch: member_of(ch) for ch in CHANNELS}
        has = member_of("__label__") is not None
        if has:
            members["label"] = member_of("__label__")
        return Group(split, location, members, has)

    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            names = set(zf.namelist())
        prefixes = sorted({n[: -len("Acc_x.txt")] for n in names if n.endswith("Acc_x.txt")})
        if not prefixes:
            raise FileNotFoundError(f"no Acc_x.txt inside {src}")
        groups = []
        for pre in prefixes:
            def member_of(name: str, _pre: str = pre) -> str | None:
                m = f"{_pre}{LABEL_FILE}" if name == "__label__" else f"{_pre}{name}.txt"
                return m if m in names else None
            groups.append(make(pre.strip("/").split("/"), member_of))
        return src, groups

    # unpacked directory: each dir containing Acc_x.txt is a group
    locs = sorted({p.parent for p in src.rglob("Acc_x.txt")})
    if not locs:
        raise FileNotFoundError(f"no Acc_x.txt under {src}")
    groups = []
    for d in locs:
        def member_of(name: str, _d: Path = d) -> str | None:
            p = _d / (LABEL_FILE if name == "__label__" else f"{name}.txt")
            return str(p) if p.exists() else None
        # split/location from the path tail; a bare "test/" dir -> location "all"
        parts = [d.parent.name, d.name] if d.name in {"Bag", "Hips", "Torso", "Hand"} else [d.name]
        groups.append(make(parts, member_of))
    return None, groups


@contextmanager
def open_group(group: Group, zip_path: Path | None) -> Iterator[dict[str, IO[str]]]:
    """Open all of a group's files as text streams; close them all on exit.

    For zips, each member gets its own ``ZipFile`` handle so the 10 streams can
    be read concurrently in lock-step without sharing a file position.
    """
    names = list(CHANNELS) + (["label"] if group.has_labels else [])
    handles: dict[str, IO[str]] = {}
    closers: list = []
    try:
        for name in names:
            member = group.members[name]
            if zip_path is not None:
                zf = zipfile.ZipFile(zip_path)
                closers.append(zf)
                stream: IO[str] = io.TextIOWrapper(zf.open(member), encoding="utf-8")
            else:
                stream = open(member)  # noqa: SIM115 - closed via `closers` in finally
            handles[name] = stream
            closers.append(stream)
        yield handles
    finally:
        for c in reversed(closers):
            with contextlib.suppress(Exception):
                c.close()


def _list_array(batch_rows: list[np.ndarray], dtype: pa.DataType) -> pa.ListArray:
    """Pack a batch of equal-length 1-D rows into an Arrow ``list<dtype>`` column."""
    flat = np.concatenate(batch_rows)
    offsets = np.arange(0, (len(batch_rows) + 1) * WINDOW_SAMPLES, WINDOW_SAMPLES, dtype=np.int32)
    return pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), pa.array(flat, dtype))


def _parse_row(line: str, fname: str, frame: int) -> list[str]:
    # Splits differ by archive: train/validation are space-separated, test is
    # comma-separated. Normalise both to whitespace before tokenising.
    toks = line.replace(",", " ").split()
    if len(toks) != WINDOW_SAMPLES:
        raise ValueError(f"{fname} frame {frame}: expected {WINDOW_SAMPLES} cols, got {len(toks)}")
    return toks


def convert_group(
    group: Group,
    zip_path: Path | None,
    out_path: Path,
    *,
    limit: int | None = None,
    batch: int = 4096,
    compression: str = "zstd",
) -> int:
    """Convert one (split, location) group to a single Parquet file. Returns frame count."""
    tag = f"{group.split}/{group.location}"
    fields = [pa.field(ch, pa.list_(pa.float32())) for ch in CHANNELS]
    if group.has_labels:
        fields.append(pa.field("label", pa.list_(pa.int8())))
    schema = pa.schema(
        fields,
        metadata={
            b"split": group.split.encode(),
            b"location": group.location.encode(),
            b"sampling_rate_hz": str(SAMPLING_RATE_HZ).encode(),
            b"window_samples": str(WINDOW_SAMPLES).encode(),
            b"channels": ",".join(CHANNELS).encode(),
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open_group(group, zip_path) as handles:
        writer = pq.ParquetWriter(out_path, schema, compression=compression)
        try:
            frame = 0
            while limit is None or n_written < limit:
                take = batch if limit is None else min(batch, limit - n_written)
                cols: dict[str, list[np.ndarray]] = {k: [] for k in handles}
                rows = 0
                for _ in range(take):
                    lines = {k: h.readline() for k, h in handles.items()}
                    if not lines[CHANNELS[0]]:  # EOF on first channel
                        if any(v for v in lines.values()):
                            raise ValueError(f"{tag}: row-count mismatch across files at frame {frame}")
                        break
                    for ch in CHANNELS:
                        cols[ch].append(np.asarray(_parse_row(lines[ch], ch, frame), dtype=np.float32))
                    if group.has_labels:
                        cols["label"].append(np.asarray(_parse_row(lines["label"], LABEL_FILE, frame), dtype=np.int8))
                    frame += 1
                    rows += 1
                if rows == 0:
                    break
                arrays = [_list_array(cols[ch], pa.float32()) for ch in CHANNELS]
                if group.has_labels:
                    arrays.append(_list_array(cols["label"], pa.int8()))
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                n_written += rows
                print(f"  {tag}: {n_written} frames", end="\r", flush=True)
        finally:
            writer.close()

    size_mb = out_path.stat().st_size / 1e6
    lab = " [labelled]" if group.has_labels else " [unlabelled]"
    print(f"  {tag}: {n_written} frames -> {out_path} ({size_mb:.1f} MB){lab}")
    return n_written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="a .zip archive or an unpacked dir")
    ap.add_argument("--out", type=Path, required=True, help="dataset_parquet root")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N frames per location")
    ap.add_argument("--batch", type=int, default=4096, help="frames per Parquet row group")
    ap.add_argument("--compression", default="zstd", help="zstd|snappy|gzip|none")
    args = ap.parse_args(argv)

    zip_path, groups = discover_groups(args.src)
    print(f"{args.src}: {len(groups)} group(s) -> {', '.join(f'{g.split}/{g.location}' for g in groups)}")
    if args.limit:
        print(f"  (sample mode: first {args.limit} frames per location)")
    total = 0
    for g in groups:
        out_path = args.out / g.split / f"{g.location}.parquet"
        total += convert_group(g, zip_path, out_path, limit=args.limit, batch=args.batch, compression=args.compression)
    print(f"done: {total} frames across {len(groups)} group(s) -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
