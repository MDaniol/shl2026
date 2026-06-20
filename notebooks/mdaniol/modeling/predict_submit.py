#!/usr/bin/env python3
"""Generate the SHL 2026 submission matrix from a trained baseline model.

Predicts one label per test frame, then replicates it across the 500 samples of
that frame (windows are ~99.8% label-constant) to produce the required
**92726 x 500** plain-text integer matrix `teamName_predictions.txt`.

Usage:
    python predict_submit.py \
        --feat-dir /path/dataset_parquet_features \
        --model    /path/artifacts/baseline_lgbm.joblib \
        --out      /path/AGH_predictions.txt --team AGH
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

N_SAMPLES = 500
N_TEST = 92726
CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--model", type=Path,
                    default=Path(__file__).resolve().parent / "artifacts" / "baseline_lgbm.joblib")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "AGH_predictions.txt")
    ap.add_argument("--team", default="AGH")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf, feat_cols = bundle["model"], bundle["feat_cols"]

    test = pd.read_parquet(args.feat_dir / "test" / "all.parquet")
    assert len(test) == N_TEST, f"expected {N_TEST} test frames, got {len(test)}"
    X = test[feat_cols].to_numpy(np.float32)

    pred = clf.predict(X).astype(int)                 # one label (1..8) per frame
    # sanity: prediction class distribution
    uniq, cnt = np.unique(pred, return_counts=True)
    print("test prediction distribution:",
          {CLASS_NAMES[c - 1]: round(n / len(pred), 3) for c, n in zip(uniq, cnt)})

    matrix = np.repeat(pred[:, None], N_SAMPLES, axis=1)   # (92726, 500)
    assert matrix.shape == (N_TEST, N_SAMPLES)
    np.savetxt(args.out, matrix, fmt="%d", delimiter=" ")
    print(f"Wrote {matrix.shape[0]}x{matrix.shape[1]} submission -> {args.out}")
    print(f"Rename/team-tag as needed (e.g. {args.team}_predictions.txt) before emailing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
