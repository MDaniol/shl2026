#!/usr/bin/env python3
"""Per-class decision calibration to maximize macro-F1 (cheap, big lever).

LightGBM optimizes log-loss; macro-F1 wants per-class decision reweighting. We
learn per-class multipliers w_c and predict argmax_c (w_c * p_c), tuning w on the
validation set (users 2&3 — same distribution as the test set). Mild optimism is
accepted (standard practice); the calibration targets the test distribution.

Saves the weights and writes a calibrated submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LOCATIONS = ("Bag", "Hips", "Torso", "Hand")
TEST_LOCS = ("Bag", "Hips", "Torso")
CLASSES = list(range(1, 9))
CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]
N_SAMPLES = 500


def load_split(feat_dir: Path, split: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(feat_dir / split / f"{loc}.parquet").assign(__loc=loc)
         for loc in LOCATIONS], ignore_index=True)


def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    here = Path(__file__).resolve().parent
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--model", type=Path, default=here / "artifacts" / "baseline_lgbm.joblib")
    ap.add_argument("--out", type=Path, default=here / "AGH_predictions_v2.txt")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf, feat_cols = bundle["model"], bundle["feat_cols"]

    val = load_split(args.feat_dir, "validation")
    Xva = val[feat_cols].to_numpy(np.float32)
    yva = val["label"].to_numpy()
    loc = val["__loc"].to_numpy()
    proba = clf.predict_proba(Xva)                       # (n, 8), columns ordered by clf.classes_
    classes = clf.classes_                                # should be [1..8]

    def f1_for(w, mask=None):
        pr = classes[(proba * w).argmax(1)]
        if mask is None:
            return macro_f1(yva, pr)
        return macro_f1(yva[mask], pr[mask])

    base_all = f1_for(np.ones(8))
    test_mask = np.isin(loc, TEST_LOCS)
    base_test = f1_for(np.ones(8), test_mask)

    # coordinate ascent on per-class multipliers (maximize macro-F1 over ALL val)
    w = np.ones(8)
    grid = np.linspace(0.3, 3.0, 28)
    for _ in range(4):
        improved = False
        for c in range(8):
            best_v, best_f1 = w[c], f1_for(w)
            for v in grid:
                w2 = w.copy(); w2[c] = v
                f = f1_for(w2)
                if f > best_f1:
                    best_f1, best_v = f, v
            if best_v != w[c]:
                w[c] = best_v; improved = True
        if not improved:
            break

    cal_all = f1_for(w)
    cal_test = f1_for(w, test_mask)
    print("=== threshold calibration (per-class multipliers) ===")
    print(f"  macro-F1 all  : {base_all:.4f} -> {cal_all:.4f}  (+{cal_all-base_all:.4f})")
    print(f"  macro-F1 test : {base_test:.4f} -> {cal_test:.4f}  (+{cal_test-base_test:.4f})")
    print("  weights:", {CLASS_NAMES[i]: round(float(w[i]), 2) for i in range(8)})

    json.dump({"weights": w.tolist(), "classes": classes.tolist(),
               "macro_f1_all": cal_all, "macro_f1_test": cal_test},
              open(args.model.parent / "class_weights.json", "w"), indent=2)

    # calibrated submission
    test = pd.read_parquet(args.feat_dir / "test" / "all.parquet")
    Xte = test[feat_cols].to_numpy(np.float32)
    pred = classes[(clf.predict_proba(Xte) * w).argmax(1)].astype(int)
    u, c = np.unique(pred, return_counts=True)
    print("  test pred dist:", {CLASS_NAMES[k-1]: round(n/len(pred), 3) for k, n in zip(u, c)})
    np.savetxt(args.out, np.repeat(pred[:, None], N_SAMPLES, axis=1), fmt="%d", delimiter=" ")
    print(f"  wrote calibrated submission -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
