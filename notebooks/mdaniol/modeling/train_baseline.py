#!/usr/bin/env python3
"""Track A baseline — LightGBM on the 520 handcrafted features.

Trains on user 1 (4 phone locations) and evaluates **user-independent** on
users 2&3 (validation). Reports macro-F1 (the challenge metric), per-class F1,
per-location macro-F1, and the confusion matrix. Saves the model for
`predict_submit.py`.

This is the MVP that de-risks the deadline: a real macro-F1 + a deployable model
before any foundation-model work.

Usage:
    python train_baseline.py \
        --feat-dir /path/dataset_parquet_features \
        --out-dir  /path/notebooks/mdaniol/modeling/artifacts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import f1_score, confusion_matrix

LOCATIONS = ("Bag", "Hips", "Torso", "Hand")
CLASSES = list(range(1, 9))
CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]


def load_split(feat_dir: Path, split: str) -> pd.DataFrame:
    dfs = []
    for loc in LOCATIONS:
        p = feat_dir / split / f"{loc}.parquet"
        df = pd.read_parquet(p)
        df["__loc"] = loc
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=63)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("Loading features…", flush=True)
    train = load_split(args.feat_dir, "train")
    val = load_split(args.feat_dir, "validation")
    feat_cols = [c for c in train.columns if c not in ("label", "__loc")]
    print(f"  train {train.shape}, val {val.shape}, {len(feat_cols)} features", flush=True)

    Xtr, ytr = train[feat_cols].to_numpy(np.float32), train["label"].to_numpy()
    Xva, yva = val[feat_cols].to_numpy(np.float32), val["label"].to_numpy()
    loc_va = val["__loc"].to_numpy()

    # class distribution (Run is the rare/critical class)
    uniq, cnt = np.unique(ytr, return_counts=True)
    print("  train class dist:", {int(c): round(n / len(ytr), 3) for c, n in zip(uniq, cnt)})

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=8,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",   # target macro-F1 (up-weights rare Run)
        n_jobs=-1,
        verbosity=-1,
        random_state=0,            # determinism (subsample<1 is stochastic)
    )
    print("Training LightGBM (early-stopping on val multi_logloss)…", flush=True)
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(150), log_evaluation(200)],
    )

    # --- evaluation on users 2&3 ---
    pred = clf.predict(Xva)
    overall = macro_f1(yva, pred)
    per_class = f1_score(yva, pred, labels=CLASSES, average=None, zero_division=0)
    per_loc = {loc: macro_f1(yva[loc_va == loc], pred[loc_va == loc]) for loc in LOCATIONS}
    test_locs = ["Bag", "Hips", "Torso"]  # the locations actually in the test set
    test_loc_macro = float(np.mean([per_loc[l] for l in test_locs]))
    cm = confusion_matrix(yva, pred, labels=CLASSES)

    print("\n================ RESULTS (user-independent, val = users 2&3) ================")
    print(f"  macro-F1 (all 4 locations) : {overall:.4f}")
    print(f"  macro-F1 (test locs B/Hi/T): {test_loc_macro:.4f}   <- closest to leaderboard")
    print("  per-class F1:")
    for n, f in zip(CLASS_NAMES, per_class):
        print(f"      {n:7s}: {f:.3f}")
    print("  per-location macro-F1:", {l: round(v, 3) for l, v in per_loc.items()})
    print("  confusion matrix (rows=true 1..8, cols=pred):")
    for i, row in enumerate(cm):
        print(f"    {CLASS_NAMES[i]:7s}", " ".join(f"{v:6d}" for v in row))

    # --- save ---
    joblib.dump({"model": clf, "feat_cols": feat_cols}, args.out_dir / "baseline_lgbm.joblib")
    json.dump(
        {"macro_f1_all": overall, "macro_f1_test_locs": test_loc_macro,
         "per_class_f1": dict(zip(CLASS_NAMES, per_class.tolist())),
         "per_loc_macro_f1": per_loc, "best_iteration": int(clf.best_iteration_ or args.n_estimators)},
        open(args.out_dir / "baseline_results.json", "w"), indent=2)
    print(f"\nSaved model + results to {args.out_dir}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
