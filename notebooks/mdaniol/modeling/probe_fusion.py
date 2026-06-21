#!/usr/bin/env python3
"""Evaluate frozen-FM embeddings: probe + fusion with handcrafted features.

For a cached embedding set (<model>_<variant>), reports user-independent macro-F1
(validation = users 2&3) under three heads:
  1. Logistic regression on standardized embeddings  (ranking probe)
  2. LightGBM on embeddings                            (nonlinear; SHL-2025 showed RF/GBM >> linear)
  3. LightGBM on [embeddings ⊕ 520 handcrafted]        (fusion — the headline)

Compares against the handcrafted-only baseline (~0.725 test-loc). Appends a row
to BAKEOFF_RESULTS.md.

Usage:
    python probe_fusion.py --emb mantisv2_V0
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from metrics import class_report, save_report

LOCATIONS = ("Bag", "Hips", "Torso", "Hand")
TEST_LOCS = ("Bag", "Hips", "Torso")
CLASSES = list(range(1, 9))


def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)


def load_emb(emb_dir: Path, split: str):
    arr = np.concatenate([np.load(emb_dir / f"{split}__{loc}.npy") for loc in LOCATIONS], 0)
    # a few windows have constant channels -> FM z-score NaN; sanitize for linear heads
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def load_labels(feat_dir: Path, split: str):
    ys, locs = [], []
    for loc in LOCATIONS:
        df = pd.read_parquet(feat_dir / split / f"{loc}.parquet", columns=["label"])
        ys.append(df["label"].to_numpy()); locs.append(np.full(len(df), loc))
    return np.concatenate(ys), np.concatenate(locs)


def load_feats(feat_dir: Path, split: str):
    dfs = [pd.read_parquet(feat_dir / split / f"{loc}.parquet") for loc in LOCATIONS]
    df = pd.concat(dfs, ignore_index=True)
    cols = [c for c in df.columns if c != "label"]
    return df[cols].to_numpy(np.float32)


def lgbm_eval(Xtr, ytr, Xva, yva):
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=2000,
                             learning_rate=0.05, num_leaves=63, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8,
                             class_weight="balanced", n_jobs=-1, verbosity=-1)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="multi_logloss",
            callbacks=[early_stopping(100)])
    return clf.predict(Xva)


def report(tag, yva, pred, loc_va):
    tm = np.isin(loc_va, TEST_LOCS)
    rep = class_report(yva[tm], pred[tm])           # test-loc per-class + averaged
    allf = macro_f1(yva, pred)
    pc = " ".join(f"{k[:2]}={d['f1']:.2f}" for k, d in rep["per_class"].items())
    print(f"  {tag:24s} test-loc macro-F1={rep['macro_f1']:.4f} (all={allf:.4f}) | {pc}")
    return {"all": allf, "test": rep["macro_f1"], "report": rep}


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--emb", help="<model>_<variant>, e.g. mantisv2_V0")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--handcrafted-baseline", action="store_true",
                    help="compute ONLY lgbm(handcrafted) under this exact protocol "
                         "(no embeddings) — the like-for-like reference for the fusion column")
    args = ap.parse_args()
    t0 = time.time()

    if args.handcrafted_baseline:
        ytr, _ = load_labels(args.feat_dir, "train")
        yva, loc_va = load_labels(args.feat_dir, "validation")
        Ftr, Fva = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
        r = report("lgbm(handcrafted-only)", yva, lgbm_eval(Ftr, ytr, Fva, yva), loc_va)
        print(f"\n>>> handcrafted-only reference (same protocol, uncalibrated): "
              f"test-loc macro-F1 = {r['test']:.4f}")
        print("    compare every lgbm(emb+handcrafted) row against THIS, not the 0.7457 calibrated.")
        return 0

    if not args.emb:
        ap.error("--emb is required (or use --handcrafted-baseline)")
    emb_dir = args.emb_root / args.emb

    print(f"[probe] {args.emb}")
    Etr, Eva = load_emb(emb_dir, "train"), load_emb(emb_dir, "validation")
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, loc_va = load_labels(args.feat_dir, "validation")
    print(f"  emb train {Etr.shape}, val {Eva.shape}")

    results = {}
    # 1. logistic regression on standardized embeddings (memory-careful)
    sc = StandardScaler().fit(Etr)
    Etr_s, Eva_s = sc.transform(Etr).astype(np.float32), sc.transform(Eva).astype(np.float32)
    lr = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced", n_jobs=-1)
    lr.fit(Etr_s, ytr)
    results["logreg_emb"] = report("logreg(emb)", yva, lr.predict(Eva_s), loc_va)
    del Etr_s, Eva_s, sc, lr; gc.collect()
    # 2. LightGBM on embeddings
    results["lgbm_emb"] = report("lgbm(emb)", yva, lgbm_eval(Etr, ytr, Eva, yva), loc_va)
    # 3. fusion: LightGBM on [emb ⊕ handcrafted]
    Ftr, Fva = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
    Xtr = np.concatenate([Etr, Ftr], 1); Xva = np.concatenate([Eva, Fva], 1)
    del Etr, Eva, Ftr, Fva; gc.collect()
    results["lgbm_fusion"] = report("lgbm(emb+handcrafted)", yva, lgbm_eval(Xtr, ytr, Xva, yva), loc_va)

    # append to results doc
    line = (f"| {args.emb} | {results['logreg_emb']['test']:.4f} | {results['lgbm_emb']['test']:.4f} "
            f"| {results['lgbm_fusion']['test']:.4f} |")
    md = root / "notebooks/mdaniol/BAKEOFF_RESULTS.md"
    header = ("# Bake-off results (val test-loc macro-F1; handcrafted baseline = 0.7457 calibrated)\n\n"
              "| emb (model_variant) | logreg(emb) | lgbm(emb) | lgbm(emb+handcrafted) |\n"
              "|---|---|---|---|\n")
    if not md.exists():
        md.write_text(header)
    md.write_text(md.read_text() + line + "\n")
    # full per-class + averaged records (per head)
    save_report(emb_dir / "probe_results.json", results)
    print(f"  appended -> {md}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
