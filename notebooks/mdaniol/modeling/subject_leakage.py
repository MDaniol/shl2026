#!/usr/bin/env python3
"""Subject-leakage diagnostic (Phase 0) — quantify why a random/blocked split inflates.

The paper claim is "our temporal+embargo, train-User1 / test-Users2&3 split is honest;
a blocked/random split inflates macro-F1 by ~+0.07 (0.80->0.87) because features encode
SUBJECT identity, not just activity." This script provides the supporting evidence:

  PER-ACTIVITY domain-identifiability — for each activity class, can a classifier tell
  User-1 (train) from Users-2&3 (validation) using ONLY the 520 handcrafted features?
  Conditioning on the activity ISOLATES subject/device identity from activity differences.
  High AUC (>~0.7) => features carry subject identity => any split that mixes the same
  subjects across fit/eval leaks it -> inflated scores. Chance (~0.5) => no subject signal.

Also reports the overall (activity-agnostic) domain AUC for context, and restates the
measured blocked-vs-temporal gap from the bake-off as the realized leakage magnitude.

CAVEAT: the current parquets merge Users 2&3 (no per-user column), so we can only probe
User-1 vs {Users 2&3}; a strict leave-one-subject-out (U2 vs U3) needs re-ingestion with a
subject column (raw_to_parquet) — flagged, not done here.

Reuses: probe_fusion.{load_feats,load_labels,CLASSES}, shl2026.track. CPU-only.

    python subject_leakage.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_feats, load_labels, CLASSES  # noqa: E402

CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]


def domain_auc(X, dom, folds=5, max_per=4000, seed=0):
    """5-fold CV AUC of classifying domain (0=User1, 1=Users2&3) from features X.
    Balanced subsample per domain so AUC reflects separability, not prevalence."""
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.where(dom == 0)[0], np.where(dom == 1)[0]
    k = min(len(idx0), len(idx1), max_per)
    if k < 50:
        return float("nan"), 0
    sel = np.concatenate([rng.choice(idx0, k, replace=False), rng.choice(idx1, k, replace=False)])
    Xs, ys = X[sel], dom[sel]
    aucs = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xs, ys):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 n_jobs=-1, verbosity=-1, random_state=seed,
                                 deterministic=True, force_col_wise=True)
        clf.fit(Xs[tr], ys[tr])
        aucs.append(roc_auc_score(ys[te], clf.predict_proba(Xs[te])[:, 1]))
    return float(np.mean(aucs)), 2 * k


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol/SUBJECT_LEAKAGE.md")
    ap.add_argument("--blocked", type=float, default=0.87, help="blocked-split macro (measured)")
    ap.add_argument("--temporal", type=float, default=0.80, help="temporal-honest macro (measured)")
    args = ap.parse_args()

    Ftr = load_feats(args.feat_dir, "train")
    Fva = load_feats(args.feat_dir, "validation")
    ytr = load_labels(args.feat_dir, "train")[0]
    yva = load_labels(args.feat_dir, "validation")[0]
    X = np.concatenate([Ftr, Fva]); y = np.concatenate([ytr, yva])
    dom = np.concatenate([np.zeros(len(Ftr), int), np.ones(len(Fva), int)])   # 0=U1, 1=U2&3
    print(f"[subj] train(U1)={len(Ftr)} val(U2&3)={len(Fva)} feats={X.shape[1]}", flush=True)

    overall, n_all = domain_auc(X, dom)
    print(f"[subj] overall domain AUC (U1 vs U2&3) = {overall:.3f}  (n={n_all})", flush=True)

    per_act = {}
    for c in CLASSES:
        m = y == c
        a, n = domain_auc(X[m], dom[m])
        per_act[CLASS_NAMES[c - 1]] = (a, n)
        print(f"  {CLASS_NAMES[c-1]:6s} domain AUC={a:.3f} (n={n})", flush=True)
    valid = [a for a, _ in per_act.values() if not np.isnan(a)]
    mean_cond = float(np.mean(valid)) if valid else float("nan")
    gap = args.blocked - args.temporal
    verdict = ("STRONG subject signal -> blocked/random splits leak it" if mean_cond > 0.7
               else "weak subject signal" if mean_cond > 0.6 else "little subject signal")
    print(f"\n[subj] mean per-activity (subject-isolated) AUC = {mean_cond:.3f} -> {verdict}", flush=True)
    print(f"[subj] realized leakage = blocked {args.blocked} - temporal {args.temporal} = +{gap:.3f} macro", flush=True)

    hdr = ("# Subject-leakage diagnostic (Phase 0)\n\n"
           f"Overall domain AUC (User-1 vs Users-2&3) = **{overall:.3f}**.\n"
           f"Mean **per-activity** (subject-isolated) AUC = **{mean_cond:.3f}** -> {verdict}.\n"
           f"Realized leakage (measured): blocked {args.blocked} - temporal {args.temporal} = "
           f"**+{gap:.3f}** macro-F1.\n\n"
           "Per-activity domain-identifiability (conditioning on activity isolates subject/device "
           "identity from activity differences):\n\n"
           "| activity | domain AUC (U1 vs U2&3) | n |\n|---|---|---|\n")
    body = "".join(f"| {k} | {a:.3f} | {n} |\n" for k, (a, n) in per_act.items())
    note = ("\n_Caveat: parquets merge Users 2&3 (no per-user column) -> only U1-vs-rest is "
            "probed; strict leave-one-subject-out (U2 vs U3) needs re-ingestion with a subject "
            "column. High AUC means features encode subject identity, so a split mixing the same "
            "subjects across fit/eval inflates macro-F1 — which is exactly the measured +"
            f"{gap:.3f} blocked-vs-temporal gap._\n")
    args.out.write_text(hdr + body + note)

    from shl2026 import track
    with track("mdaniol", run_name="subject_leakage", seed=0, params_path=None,
               params={"feat_dir": str(args.feat_dir), "n_train_U1": len(Ftr), "n_val_U23": len(Fva)},
               tags={"phase": "phase0", "branch": "leakage_audit"}) as run:
        run.log_metrics({"overall_domain_auc": overall, "mean_perclass_domain_auc": mean_cond,
                         "blocked_minus_temporal": gap,
                         **{f"domain_auc_{k}": a for k, (a, n) in per_act.items() if not np.isnan(a)}})
        run.log_artifact(args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
