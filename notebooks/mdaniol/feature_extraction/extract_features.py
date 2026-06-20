#!/usr/bin/env python3
"""Extract the SHL feature bank from one Parquet file → one feature Parquet.

Reads a ``dataset_parquet/<split>/<location>.parquet`` (9 list<float32> channel
columns of length 500, optional ``label`` list<int8>), computes the 520
handcrafted features per window (see ``shl_features.py``), and writes a columnar
feature table. Memory is bounded by ``--chunk-size`` windows via streaming
Parquet row-groups, so the same script handles the 196k-row train files and the
92k-row test file on a modest node.

Per-window label = **majority vote** over the 500 frames (windows are 99.8%
constant; majority is robust to the rare mixed ones). Test files have no label.

Usage (single file):
    python extract_features.py --input  dataset_parquet/train/Bag.parquet \
                               --output dataset_parquet_features/train/Bag.parquet \
                               --split train --location Bag

Self-test only (no I/O):
    python extract_features.py --self-test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import shl_features as shl

CHANNELS = (
    "Acc_x", "Acc_y", "Acc_z",
    "Gyr_x", "Gyr_y", "Gyr_z",
    "Mag_x", "Mag_y", "Mag_z",
)
N = shl.N
N_CLASSES = 8  # SHL labels are 1..8


def _list_col_to_2d(col: pa.ChunkedArray | pa.Array, width: int) -> np.ndarray:
    """Convert a list<float/int> column of fixed length ``width`` to (n, width)."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    # FixedSizeList or List: flatten the values then reshape.
    flat = col.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(-1, width)


def _majority_label(label_2d: np.ndarray) -> np.ndarray:
    """Per-row majority over classes 1..8. (n, 500) int -> (n,) int8."""
    counts = np.empty((label_2d.shape[0], N_CLASSES), dtype=np.int32)
    for c in range(1, N_CLASSES + 1):
        counts[:, c - 1] = np.count_nonzero(label_2d == c, axis=1)
    return (counts.argmax(axis=1) + 1).astype(np.int8)


def _stack_axes(batch: pa.RecordBatch, names: tuple[str, str, str]) -> np.ndarray:
    """Stack three channel columns into (n, 3, N)."""
    x = _list_col_to_2d(batch.column(batch.schema.get_field_index(names[0])), N)
    y = _list_col_to_2d(batch.column(batch.schema.get_field_index(names[1])), N)
    z = _list_col_to_2d(batch.column(batch.schema.get_field_index(names[2])), N)
    return np.stack([x, y, z], axis=1).astype(np.float64)


def process_file(
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    zscore: bool,
) -> None:
    pf = pq.ParquetFile(input_path)
    has_label = "label" in pf.schema_arrow.names
    feat_names = shl.feature_names(zscore=zscore)

    # Build the output schema once.
    fields = [pa.field(n, pa.float32()) for n in feat_names]
    if has_label:
        fields.append(pa.field("label", pa.int8()))
    out_schema = pa.schema(
        fields,
        metadata={
            b"feature_count": str(len(feat_names)).encode(),
            b"zscore": str(zscore).encode(),
            b"source": str(input_path).encode(),
            b"window_samples": str(N).encode(),
            b"sampling_rate_hz": str(int(shl.FS)).encode(),
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output_path, out_schema)

    n_done = 0
    t0 = time.time()
    try:
        for batch in pf.iter_batches(batch_size=chunk_size):
            acc = _stack_axes(batch, ("Acc_x", "Acc_y", "Acc_z"))
            gyr = _stack_axes(batch, ("Gyr_x", "Gyr_y", "Gyr_z"))
            mag = _stack_axes(batch, ("Mag_x", "Mag_y", "Mag_z"))

            mat, names = shl.extract_features(acc, gyr, mag, zscore=zscore)
            assert names == feat_names, "feature name/order drift between chunks"

            arrays = [pa.array(mat[:, i], type=pa.float32())
                      for i in range(mat.shape[1])]
            if has_label:
                lab2d = _list_col_to_2d(
                    batch.column(batch.schema.get_field_index("label")), N
                )
                arrays.append(pa.array(_majority_label(lab2d), type=pa.int8()))

            writer.write_table(pa.Table.from_arrays(arrays, schema=out_schema))
            n_done += mat.shape[0]
            rate = n_done / max(time.time() - t0, 1e-6)
            print(f"  {n_done:>8d} windows  ({rate:,.0f}/s)", flush=True)
    finally:
        writer.close()

    dt = time.time() - t0
    print(f"DONE {input_path.name} -> {output_path}  "
          f"({n_done} windows, {len(feat_names)} features, {dt:.1f}s)", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, help="input parquet (split/location)")
    ap.add_argument("--output", type=Path, help="output feature parquet")
    ap.add_argument("--chunk-size", type=int, default=20000,
                    help="windows per streamed row-group (memory knob)")
    ap.add_argument("--zscore", action="store_true",
                    help="z-normalize streams before spectral features (ablation)")
    ap.add_argument("--split", default=None, help="annotation only")
    ap.add_argument("--location", default=None, help="annotation only")
    ap.add_argument("--self-test", action="store_true",
                    help="run shl_features.self_test() and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        shl.self_test()
        return 0

    if args.input is None or args.output is None:
        ap.error("--input and --output are required (or use --self-test)")
    if not args.input.exists():
        ap.error(f"input not found: {args.input}")

    print(f"[extract] {args.input}  split={args.split} location={args.location} "
          f"zscore={args.zscore} chunk={args.chunk_size}", flush=True)
    process_file(args.input, args.output, args.chunk_size, args.zscore)
    return 0


if __name__ == "__main__":
    sys.exit(main())
