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
TEST_LOCS = ("Bag", "Hips", "Torso")          # the BHT locations mirrored at test (eval masks)
CLASSES = list(range(1, 9))


def split_locs(split: str) -> tuple[str, ...]:
    """File tokens to load for a split. train/validation are stored per body location;
    the test set is a single merged, shuffled file ('all' — no per-location, no Hand)."""
    return ("all",) if split == "test" else LOCATIONS


def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)


def load_emb(emb_dir: Path, split: str):
    arr = np.concatenate([np.load(emb_dir / f"{split}__{loc}.npy") for loc in split_locs(split)], 0)
    # a few windows have constant channels -> FM z-score NaN; sanitize for linear heads
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def load_labels(feat_dir: Path, split: str):
    ys, locs = [], []
    for loc in split_locs(split):
        df = pd.read_parquet(feat_dir / split / f"{loc}.parquet", columns=["label"])
        ys.append(df["label"].to_numpy()); locs.append(np.full(len(df), loc))
    return np.concatenate(ys), np.concatenate(locs)


def load_feats(feat_dir: Path, split: str):
    dfs = [pd.read_parquet(feat_dir / split / f"{loc}.parquet") for loc in split_locs(split)]
    df = pd.concat(dfs, ignore_index=True)
    cols = [c for c in df.columns if c != "label"]
    return df[cols].to_numpy(np.float32)


def lgbm_eval(Xtr, ytr, Xva, yva):
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=2000,
                             learning_rate=0.05, num_leaves=63, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8,
                             class_weight="balanced", n_jobs=-1, verbosity=-1, random_state=0)
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


def calibrate(proba, classes, y):
    """Per-class multipliers maximizing macro-F1 (mirrors train_split.calibrate)."""
    w = np.ones(len(classes)); grid = np.linspace(0.3, 3.0, 28)
    f = lambda ww: macro_f1(y, classes[(proba * ww).argmax(1)])
    for _ in range(4):
        improved = False
        for c in range(len(classes)):
            best_v, best = w[c], f(w)
            for v in grid:
                w2 = w.copy(); w2[c] = v
                if f(w2) > best:
                    best, best_v = f(w2), v
            if best_v != w[c]:
                w[c] = best_v; improved = True
        if not improved:
            break
    return w


def aligned_proba(clf, X, w=None):
    """Predict probabilities re-indexed to the fixed CLASSES (1..8) order and
    row-normalized, so probas from different experts/heads are directly mixable
    (the MoE/β-blend in moe_combine.py). Optionally apply calibration weights w
    (in clf.classes_ order) first."""
    P = clf.predict_proba(X)
    if w is not None:
        P = P * w
    col = {c: i for i, c in enumerate(clf.classes_)}
    out = np.zeros((P.shape[0], len(CLASSES)), dtype=np.float64)
    for j, c in enumerate(CLASSES):
        if c in col:
            out[:, j] = P[:, col[c]]
    out /= (out.sum(axis=1, keepdims=True) + 1e-12)
    return out


def fit_cal_eval(tag, Xfit, yfit, Xtune, ytune, Xtest, ytest, return_probs=False):
    """train_split protocol: fit on FIT, early-stop + calibrate on TUNE, eval on
    the held-out TEST (already Bag/Hips/Torso).

    Default: returns the calibrated TEST class_report (back-compat).
    return_probs=True: returns a dict with the fitted model, calibration weights,
    class-aligned calibrated probabilities on TUNE and TEST, both reports, and the
    selection-lock gap — the shared artifact the oracle/router/MoE consume."""
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=2000,
                             learning_rate=0.05, num_leaves=63, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8,
                             class_weight="balanced", n_jobs=-1, verbosity=-1, random_state=0)
    clf.fit(Xfit, yfit, eval_set=[(Xtune, ytune)], eval_metric="multi_logloss",
            callbacks=[early_stopping(100)])
    w = calibrate(clf.predict_proba(Xtune), clf.classes_, ytune)
    cls = np.asarray(CLASSES)
    Ptu = aligned_proba(clf, Xtune, w)
    Pte = aligned_proba(clf, Xtest, w)
    rep = class_report(ytest, cls[Pte.argmax(1)])
    rep_tune = class_report(ytune, cls[Ptu.argmax(1)])
    gap = rep_tune["macro_f1"] - rep["macro_f1"]          # selection-lock gap
    pc = " ".join(f"{k[:2]}={d['f1']:.2f}" for k, d in rep["per_class"].items())
    print(f"  {tag:24s} TEST macro-F1={rep['macro_f1']:.4f} | TUNE={rep_tune['macro_f1']:.4f} "
          f"(gap {gap:+.4f}) | per-class: {pc}")
    if return_probs:
        return {"rep": rep, "rep_tune": rep_tune, "gap": gap, "model": clf,
                "weights": w, "classes": cls, "proba_tune": Ptu, "proba_test": Pte}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--emb", help="<model>_<variant>, e.g. mantisv2_V0")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--handcrafted-baseline", action="store_true",
                    help="compute ONLY lgbm(handcrafted) under this exact protocol "
                         "(no embeddings) — the like-for-like reference for the fusion column")
    ap.add_argument("--use-split", action="store_true",
                    help="use the train_split.py protocol: fit on User-1 + validation[FIT], "
                         "calibrate on TUNE, eval on held-out TEST (apples-to-apples vs handcrafted)")
    ap.add_argument("--split", type=Path,
                    default=Path(__file__).resolve().parent / "artifacts" / "val_split.npy")
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

    # ---- split-aligned protocol (apples-to-apples vs train_split.py) ----------
    if args.use_split:
        assign = np.load(args.split)
        Etr, Eva = load_emb(emb_dir, "train"), load_emb(emb_dir, "validation")
        ytr, _ = load_labels(args.feat_dir, "train")
        yva, loc_va = load_labels(args.feat_dir, "validation")
        Ftr, Fva = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
        assert len(assign) == len(yva), "val_split / validation length mismatch"
        fm, tm, sm = assign == 0, assign == 1, assign == 2
        print(f"[probe-split] {args.emb}  fit=User1+val[{fm.sum()}] tune={tm.sum()} test={sm.sum()}")

        def slices(Etr_, Eva_):                       # FIT = User1 + val[FIT]
            Xfit = np.concatenate([Etr_, Eva_[fm]]); yfit = np.concatenate([ytr, yva[fm]])
            return Xfit, yfit, Eva_[tm], yva[tm], Eva_[sm], yva[sm]

        reps = {}
        reps["emb"] = fit_cal_eval("lgbm(emb)", *slices(Etr, Eva))
        EFtr = np.concatenate([Etr, Ftr], 1); EFva = np.concatenate([Eva, Fva], 1)
        reps["fusion"] = fit_cal_eval("lgbm(emb+handcrafted)", *slices(EFtr, EFva))
        reps["handcrafted"] = fit_cal_eval("lgbm(handcrafted-only)", *slices(Ftr, Fva))
        d = reps["fusion"]["macro_f1"] - reps["handcrafted"]["macro_f1"]
        print(f"\n>>> FM verdict (split protocol): fusion {reps['fusion']['macro_f1']:.4f} "
              f"vs handcrafted-only {reps['handcrafted']['macro_f1']:.4f}  ->  "
              f"{d:+.4f} ({'FM HELPS' if d > 0 else 'FM does NOT help'})")
        # blocked vs temporal rows coexist via an explicit `split` column. A legacy
        # 5-column table (no split column) is auto-archived so old/new rows can't mix.
        split_label = "temporal" if "temporal" in args.split.stem else "blocked"
        md = root / "notebooks/mdaniol/BAKEOFF_SPLIT.md"
        header = ("# Bake-off (split protocol: User1+val[FIT] -> held-out TEST, calibrated)\n\n"
                  "| emb (model_variant) | split | lgbm(emb) | lgbm(emb+hc) | lgbm(hc-only) | Δ vs hc |\n"
                  "|---|---|---|---|---|---|\n")
        if md.exists() and "| split |" not in md.read_text():
            md.rename(md.with_name("BAKEOFF_SPLIT.legacy.md"))     # archive 5-col table, start fresh
        if not md.exists():
            md.write_text(header)
        md.write_text(md.read_text() + (
            f"| {args.emb} | {split_label} | {reps['emb']['macro_f1']:.4f} "
            f"| {reps['fusion']['macro_f1']:.4f} | {reps['handcrafted']['macro_f1']:.4f} | {d:+.4f} |\n"))
        json_path = emb_dir / "probe_split_results.json"
        save_report(json_path, reps)
        # MLflow: one run per embedding set; split scheme + bare macro_f1 + artifacts (rule §8).
        from shl2026 import track
        with track("mdaniol", run_name=f"bakeoff_{args.emb}", seed=0, params_path=None,
                   params={"emb": args.emb, "split": args.split.stem,
                           "protocol": "fit(User1+val[FIT])->cal(TUNE)->lock(TEST)"},
                   tags={"phase": "bakeoff", "branch": "fm_probe",
                         "verdict": "FM_HELPS" if d > 0 else "FM_NO_HELP"}) as run:
            run.log_metrics({"macro_f1": reps["fusion"]["macro_f1"],      # headline = fused
                             "emb_macro_f1": reps["emb"]["macro_f1"],
                             "handcrafted_macro_f1": reps["handcrafted"]["macro_f1"],
                             "delta_vs_handcrafted": d})
            run.log_artifact(md)
            run.log_artifact(json_path)
        print(f"  appended -> {md}  ({time.time()-t0:.0f}s)")
        return 0

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
