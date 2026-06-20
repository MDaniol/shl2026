#!/usr/bin/env python3
"""Augment raw IMU windows → re-extract the 520 handcrafted features → Parquet.

The missing bridge between `augment.py` (physics aug on raw (n,3,500) windows)
and `shl_features.py` (the 520-feature bank). Output schema is byte-identical to
`feature_extraction/extract_features.py`, so `train_split.py` / `robustness_sweep.py`
consume augmented feature Parquets transparently.

Branch policy (AUGMENTATION_STRATEGY.md §7):
  * Branch A (handcrafted)  -> --branch A : NO rotation (magnitudes are
    rotation-invariant), only scale/jitter/time-warp. Use for train-time aug.
  * Branch B (raw-axis FM)  -> --branch B : full pipeline incl. rotation.

Streaming row-groups bound memory; --copies writes N augmented copies (distinct
seeds), each row-aligned to the original labels.

Usage (1 augmented copy of User-1 Bag, Branch-A aug):
    python augment_features.py --input  dataset_parquet/train/Bag.parquet \
                               --output dataset_parquet_features_aug/train/Bag.parquet \
                               --branch A --copies 1 --seed 0

Self-test (rotation must NOT change Branch-A features — the invariance proof):
    python augment_features.py --self-test --input dataset_parquet/validation/Bag.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# shl_features + extract_features I/O helpers live in ../feature_extraction
_FE = Path(__file__).resolve().parents[1] / "feature_extraction"
sys.path.insert(0, str(_FE))
import shl_features as shl                                          # noqa: E402
from extract_features import _stack_axes, _list_col_to_2d, _majority_label  # noqa: E402

import augment as aug                                              # noqa: E402

N = shl.N


def make_config(args) -> aug.AugConfig:
    """Build AugConfig from CLI; --branch sets the rotation default."""
    rotation = None if args.branch == "A" else args.rotation
    if args.rotation == "none":
        rotation = None
    return aug.AugConfig(
        rotation=rotation,
        max_tilt_deg=args.max_tilt,
        scale=not args.no_scale,
        scale_sigma=args.scale_sigma,
        jitter=not args.no_jitter,
        jitter_sigma_frac=args.jitter_sigma,
        timewarp=not args.no_timewarp,
        tw_sigma=args.tw_sigma,
    )


def process_file(input_path: Path, output_path: Path, cfg: aug.AugConfig,
                 with_rotation: bool, copies: int, seed: int, chunk_size: int) -> None:
    pf = pq.ParquetFile(input_path)
    has_label = "label" in pf.schema_arrow.names
    if not has_label:
        raise SystemExit(f"{input_path} has no label column — aug is for train/val only")
    feat_names = shl.feature_names(zscore=False)

    fields = [pa.field(n, pa.float32()) for n in feat_names] + [pa.field("label", pa.int8())]
    out_schema = pa.schema(fields, metadata={
        b"feature_count": str(len(feat_names)).encode(),
        b"augmented": b"true",
        b"branch": (b"B" if with_rotation else b"A"),
        b"rotation": str(cfg.rotation).encode(),
        b"copies": str(copies).encode(),
        b"seed": str(seed).encode(),
        b"source": str(input_path).encode(),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output_path, out_schema)

    n_done, t0 = 0, time.time()
    try:
        for ci in range(copies):
            rng = np.random.default_rng(seed + 1000 * ci)
            for batch in pf.iter_batches(batch_size=chunk_size):
                acc = _stack_axes(batch, ("Acc_x", "Acc_y", "Acc_z"))
                gyr = _stack_axes(batch, ("Gyr_x", "Gyr_y", "Gyr_z"))
                mag = _stack_axes(batch, ("Mag_x", "Mag_y", "Mag_z"))
                a, g, m = aug.augment(acc, gyr, mag, cfg, rng, with_rotation=with_rotation)
                mat, names = shl.extract_features(a, g, m, zscore=False)
                assert names == feat_names, "feature order drift"
                lab2d = _list_col_to_2d(
                    batch.column(batch.schema.get_field_index("label")), N)
                arrays = [pa.array(mat[:, i], type=pa.float32()) for i in range(mat.shape[1])]
                arrays.append(pa.array(_majority_label(lab2d), type=pa.int8()))
                writer.write_table(pa.Table.from_arrays(arrays, schema=out_schema))
                n_done += mat.shape[0]
            print(f"  copy {ci+1}/{copies} done ({n_done} windows, "
                  f"{n_done/max(time.time()-t0,1e-6):,.0f}/s)", flush=True)
    finally:
        writer.close()
    print(f"DONE {input_path.name} -> {output_path}  ({n_done} rows, {time.time()-t0:.1f}s)",
          flush=True)


def self_test(input_path: Path) -> int:
    """Rotation must leave Branch-A (magnitude) features ~unchanged."""
    pf = pq.ParquetFile(input_path)
    batch = next(pf.iter_batches(batch_size=512))
    acc = _stack_axes(batch, ("Acc_x", "Acc_y", "Acc_z"))
    gyr = _stack_axes(batch, ("Gyr_x", "Gyr_y", "Gyr_z"))
    mag = _stack_axes(batch, ("Mag_x", "Mag_y", "Mag_z"))
    base, names = shl.extract_features(acc, gyr, mag)
    rng = np.random.default_rng(0)
    cfg = aug.AugConfig(rotation="so3", scale=False, jitter=False, timewarp=False)
    a, g, m = aug.augment(acc, gyr, mag, cfg, rng, with_rotation=True)
    rot, _ = shl.extract_features(a, g, m)
    denom = np.abs(base).mean(0) + 1e-6
    rel = np.abs(rot - base).mean(0) / denom
    worst = np.argsort(rel)[-5:]
    print(f"[self-test] rotation-vs-baseline mean rel-diff = {rel.mean():.2e} "
          f"(max {rel.max():.2e})")
    for i in worst:
        print(f"   {names[i]:32s} rel-diff {rel[i]:.2e}")
    ok = rel.max() < 1e-3
    print("PASS: Branch-A features are rotation-invariant" if ok
          else "FAIL: rotation changed magnitude features — pipeline bug!")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--branch", choices=["A", "B"], default="A",
                    help="A: no rotation (handcrafted); B: rotation on (raw-axis FM)")
    ap.add_argument("--rotation", choices=["so3", "gravity_aware", "none"], default="so3")
    ap.add_argument("--max-tilt", type=float, default=45.0)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8000)
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--scale-sigma", type=float, default=0.1)
    ap.add_argument("--no-jitter", action="store_true")
    ap.add_argument("--jitter-sigma", type=float, default=0.03)
    ap.add_argument("--no-timewarp", action="store_true")
    ap.add_argument("--tw-sigma", type=float, default=0.15)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        if args.input is None:
            ap.error("--self-test needs --input (a labelled raw parquet)")
        return self_test(args.input)

    if args.input is None or args.output is None:
        ap.error("--input and --output required")
    if not args.input.exists():
        ap.error(f"input not found: {args.input}")
    cfg = make_config(args)
    with_rotation = args.branch == "B" and cfg.rotation is not None
    print(f"[aug] {args.input} branch={args.branch} rotation="
          f"{cfg.rotation if with_rotation else None} copies={args.copies} seed={args.seed}",
          flush=True)
    process_file(args.input, args.output, cfg, with_rotation, args.copies, args.seed,
                 args.chunk_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
