#!/usr/bin/env python3
"""Robustness diagnostic (AUGMENTATION_STRATEGY.md §8) on the held-out TEST slice.

Your idea: "expand the held-out set with physics-based modifications (rotation,
noise) and measure what happens." We take the 20% held-out TEST windows (Users
2&3, Bag/Hips/Torso), apply augmentation at increasing severity, re-extract the
520 features, and score the **Phase-A** model (which never saw TEST) — macro-F1
vs severity, per sweep.

Expected (Branch A = magnitude features):
  * ROTATION curve FLAT  -> confirms rotation-invariance (a sanity check: if it
    DROPS, the magnitude pipeline has a bug). Also the empirical novelty point.
  * NOISE / TIME-WARP curves slope down -> the real robustness numbers, and the
    severities where aug-at-train-time would help.

Honest mapping: val_split.npy is row-aligned to the LOCATIONS-order concatenation
of the validation feature/raw files, so per-location offsets recover the exact
TEST rows. Uses model_heldout + calibration weights from split_lgbm.joblib.

Usage:
    python robustness_sweep.py --raw-dir dataset_parquet/validation \
        --split artifacts/val_split.npy --model artifacts/split_lgbm.joblib \
        --out artifacts/robustness.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import joblib, numpy as np
import pyarrow.parquet as pq

_FE = Path(__file__).resolve().parents[1] / "feature_extraction"
sys.path.insert(0, str(_FE))
import shl_features as shl                                          # noqa: E402
from extract_features import _stack_axes, _list_col_to_2d, _majority_label  # noqa: E402
import augment as aug                                              # noqa: E402
from metrics import class_report                                  # noqa: E402

LOCATIONS = ("Bag", "Hips", "Torso", "Hand")
TEST = 2
N = shl.N


def load_test_windows(raw_dir: Path, assign: np.ndarray):
    """Return (acc,gyr,mag,label) for the held-out TEST rows, LOCATIONS order."""
    accs, gyrs, mags, labs = [], [], [], []
    offset = 0
    for loc in LOCATIONS:
        pf = pq.ParquetFile(raw_dir / f"{loc}.parquet")
        n = pf.metadata.num_rows
        loc_assign = assign[offset:offset + n]
        offset += n
        if not (loc_assign == TEST).any():
            continue
        row0 = 0
        for batch in pf.iter_batches(batch_size=20000):
            nb = batch.num_rows
            sel = np.where(loc_assign[row0:row0 + nb] == TEST)[0]
            row0 += nb
            if sel.size == 0:
                continue
            acc = _stack_axes(batch, ("Acc_x", "Acc_y", "Acc_z"))[sel]
            gyr = _stack_axes(batch, ("Gyr_x", "Gyr_y", "Gyr_z"))[sel]
            mag = _stack_axes(batch, ("Mag_x", "Mag_y", "Mag_z"))[sel]
            lab = _majority_label(_list_col_to_2d(
                batch.column(batch.schema.get_field_index("label")), N))[sel]
            accs.append(acc.astype(np.float32)); gyrs.append(gyr.astype(np.float32))
            mags.append(mag.astype(np.float32)); labs.append(lab)
    assert offset == len(assign), f"raw rows {offset} != split {len(assign)}"
    return (np.concatenate(accs), np.concatenate(gyrs),
            np.concatenate(mags), np.concatenate(labs))


def evaluate(acc, gyr, mag, y, model, w, feat_cols, cfg, with_rotation, rng):
    a, g, m = aug.augment(acc, gyr, mag, cfg, rng, with_rotation=with_rotation)
    X, names = shl.extract_features(a, g, m, zscore=False)
    assert names == feat_cols, "feature order mismatch vs trained model"
    pred = model.classes_[(model.predict_proba(X) * w).argmax(1)]
    rep = class_report(y, pred)
    return rep["macro_f1"], rep


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    ap.add_argument("--raw-dir", type=Path, default=root / "dataset_parquet" / "validation")
    ap.add_argument("--split", type=Path, default=here / "artifacts" / "val_split.npy")
    ap.add_argument("--model", type=Path, default=here / "artifacts" / "split_lgbm.joblib")
    ap.add_argument("--out", type=Path, default=here / "artifacts" / "robustness.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); t0 = time.time()

    bundle = joblib.load(args.model)
    model = bundle.get("model_heldout") or bundle["model"]
    w, feat_cols = bundle["weights"], bundle["feat_cols"]
    assign = np.load(args.split)

    print("loading held-out TEST raw windows ...", flush=True)
    acc, gyr, mag, y = load_test_windows(args.raw_dir, assign)
    print(f"  TEST windows: {len(y)}  ({time.time()-t0:.0f}s)", flush=True)

    NO = aug.AugConfig(rotation=None, scale=False, jitter=False, timewarp=False)
    clean_f1, clean_rep = evaluate(acc, gyr, mag, y, model, w, feat_cols, NO, False,
                                   np.random.default_rng(args.seed))
    print(f"\nclean (no aug) held-out macro-F1 = {clean_f1:.4f}")

    results = {"clean_macro_f1": clean_f1, "n_test": int(len(y)),
               "clean_per_class": clean_rep["per_class"], "sweeps": {}}

    # 1) rotation severity — should be FLAT for Branch A (invariance check)
    rot = []
    for tilt in (0, 15, 30, 45, 90):
        cfg = aug.AugConfig(rotation="gravity_aware", max_tilt_deg=tilt,
                            scale=False, jitter=False, timewarp=False)
        f1, _ = evaluate(acc, gyr, mag, y, model, w, feat_cols, cfg, True,
                         np.random.default_rng(args.seed))
        rot.append({"tilt_deg": tilt, "macro_f1": round(f1, 4)})
    cfg = aug.AugConfig(rotation="so3", scale=False, jitter=False, timewarp=False)
    f1, _ = evaluate(acc, gyr, mag, y, model, w, feat_cols, cfg, True,
                     np.random.default_rng(args.seed))
    rot.append({"tilt_deg": "SO3", "macro_f1": round(f1, 4)})
    results["sweeps"]["rotation"] = rot

    # 2) jitter / noise severity
    noise = []
    for s in (0.0, 0.02, 0.05, 0.1):
        cfg = aug.AugConfig(rotation=None, scale=False, jitter=(s > 0),
                            jitter_sigma_frac=s, timewarp=False)
        f1, _ = evaluate(acc, gyr, mag, y, model, w, feat_cols, cfg, False,
                         np.random.default_rng(args.seed))
        noise.append({"jitter_sigma_frac": s, "macro_f1": round(f1, 4)})
    results["sweeps"]["noise"] = noise

    # 3) time-warp severity
    tw = []
    for s in (0.0, 0.15, 0.3):
        cfg = aug.AugConfig(rotation=None, scale=False, jitter=False,
                            timewarp=(s > 0), tw_sigma=s)
        f1, _ = evaluate(acc, gyr, mag, y, model, w, feat_cols, cfg, False,
                         np.random.default_rng(args.seed))
        tw.append({"tw_sigma": s, "macro_f1": round(f1, 4)})
    results["sweeps"]["timewarp"] = tw

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print("\n=== ROBUSTNESS SWEEP (held-out TEST macro-F1) ===")
    print(f"clean = {clean_f1:.4f}")
    print(" rotation:", "  ".join(f"{r['tilt_deg']}:{r['macro_f1']}" for r in rot),
          " <- FLAT expected (Branch A invariance)")
    print(" noise   :", "  ".join(f"{r['jitter_sigma_frac']}:{r['macro_f1']}" for r in noise))
    print(" timewarp:", "  ".join(f"{r['tw_sigma']}:{r['macro_f1']}" for r in tw))
    drop = clean_f1 - rot[-1]["macro_f1"]
    print(f"\nSO(3) rotation drop = {drop:+.4f} "
          f"({'OK invariant' if abs(drop) < 0.01 else 'WARN: not flat — check pipeline'})")
    print(f"wrote {args.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
