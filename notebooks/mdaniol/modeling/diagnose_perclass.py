#!/usr/bin/env python3
"""Diagnostic — full per-class F1 + confusion for the MOMENT-fusion model under the
CONSERVATIVE temporal split. Answers: which classes drag macro-F1, and what is each
confused with (so we know whether a vehicle/rail corrector could even help).

Read-only, leakage-safe: train on User-1 + validation[FIT], evaluate on the held-out
temporal TEST (Bag/Hips/Torso). Reuses submit_fusion.fuse + evaluate_predictions.

Usage:  python diagnose_perclass.py --emb moment-small_V1 \
            --split notebooks/mdaniol/modeling/artifacts/val_split_temporal.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_labels  # noqa: E402
from split import load_split_with_location_map  # noqa: E402
from submit_fusion import fuse  # noqa: E402

CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]
BHT = ("Bag", "Hips", "Torso")
FIT, TEST = 0, 2


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--emb", default="moment-small_V1")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    args = ap.parse_args()
    from shl2026 import evaluate_predictions

    Xtr = fuse(args.feat_dir, args.emb_root, args.emb, "train")
    Xva = fuse(args.feat_dir, args.emb_root, args.emb, "validation")
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, _ = load_labels(args.feat_dir, "validation")
    assign, loc_off = load_split_with_location_map(args.split, args.feat_dir)
    va_loc = np.empty(len(assign), dtype=object)
    for loc, (s, e) in loc_off.items():
        va_loc[s:e] = loc
    bht = np.isin(va_loc.astype(str), BHT)
    fit_m, test_m = assign == FIT, (assign == TEST) & bht

    Xfit = np.concatenate([Xtr, Xva[fit_m]]); yfit = np.concatenate([ytr, yva[fit_m]])
    Xtest, ytest = Xva[test_m], yva[test_m]
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=600,
                             learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, class_weight="balanced", n_jobs=-1,
                             verbosity=-1, random_state=0,
                             deterministic=True, force_row_wise=True).fit(Xfit, yfit)
    res = evaluate_predictions(ytest, clf.predict(Xtest))

    print(f"\nMOMENT-fusion ({args.emb}) on temporal TEST  n={res.n}  "
          f"macro-F1={res.macro_f1:.4f}  acc={res.accuracy:.4f}\n")
    print("per-class F1 (sorted, weakest first):")
    for c, f1 in sorted(res.per_class_f1.items(), key=lambda kv: kv[1]):
        bar = "#" * int(f1 * 40)
        print(f"  {CLASS_NAMES[c-1]:7s} {f1:.3f} {bar}")

    cm = res.confusion                                  # rows=true, cols=pred, labels 1..8
    print("\nconfusion (row=true, col=pred):")
    print("        " + " ".join(f"{n[:4]:>5s}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n[:6]:7s}" + " ".join(f"{cm[i, j]:5d}" for j in range(8)))

    # the key off-diagonals for the corrector decision
    def pair(a, b):  # 1-based class ids
        return f"{CLASS_NAMES[a-1]}->{CLASS_NAMES[b-1]}={cm[a-1, b-1]}, " \
               f"{CLASS_NAMES[b-1]}->{CLASS_NAMES[a-1]}={cm[b-1, a-1]}"
    print("\nkey confusions:")
    print("  Car/Bus      :", pair(5, 6))
    print("  Train/Subway :", pair(7, 8))
    print("  Bus vs non-vehicle: Bus->Still={}, Bus->Walk={}, Bus->Car={}, Bus->Train={}, Bus->Subway={}"
          .format(cm[5, 0], cm[5, 1], cm[5, 4], cm[5, 6], cm[5, 7]))
    print("\n-> If Bus/Car/rail errors are WITHIN the vehicle subset, a V4 corrector can help;")
    print("   if they leak to Still/Walk (outside the subset), V4 cannot fix them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
