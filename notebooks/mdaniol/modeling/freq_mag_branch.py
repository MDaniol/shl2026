#!/usr/bin/env python3
"""Step 2 — controlled frequency + magnetometer branch (plan), reuse-first.

The audit found the 520-feature bank ALREADY contains the spectral + magnetometer
features (one-sided PSD bandpower energy/ratio, dominant frequency, fft mean/std,
mag_mag + mag_rate_mag = magnetic jerk streams). So this branch SELECTS that
curated subset (by column name) from the existing feature parquets and trains
*small linear* heads — deliberately NOT a high-capacity booster on a feature dump
(the report's failure mode). Scored under the split protocol on TUNE + TEST.

Selected columns (controlled, ~interpretable):
  * every spectral feature, any stream:  "__freq_" or "__sb_"  (PSD bandpower
    energy + normalized ratios, dominant freq, spectral level)
  * every magnetometer feature:          stream mag_mag__ / mag_rate_mag__
  * magnitude/jerk time-stats:           {acc,gyr,acc_jerk,gyr_jerk}_mag__time_

Heads: logistic regression, ridge, elastic-net (SGD log-loss), + one SHALLOW
LightGBM as an auxiliary only. Output: FREQMAG_RESULTS.md + MLflow (shl2026.track).

NOTE: spectral entropy / spectral centroid / magnetic-spike-count and the wider
band grid (0.1-0.5 … 35-45 Hz) are NOT in the current bank; adding them needs a
feature-bank extension + re-extraction (documented follow-up), so this branch
uses what is already extracted — no re-extraction, runs on cached features.

Usage:  python freq_mag_branch.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_feats, load_labels  # noqa: E402
from split import load_split_with_location_map  # noqa: E402

FIT, TUNE, TEST = 0, 1, 2
MAG_STREAMS = ("mag_mag__", "mag_rate_mag__")                       # all magnetometer features
FREQ_STREAMS = ("acc_mag__", "gyr_mag__")                           # freq of magnitude streams only
MAGNITUDE_TIME = ("acc_mag__time", "gyr_mag__time",
                  "acc_jerk_mag__time", "gyr_jerk_mag__time")       # magnitude/jerk time-stats


def select_freq_mag(columns) -> list[int]:
    """Indices of a *controlled* (not whole-bank) frequency + magnetometer subset:
      - every magnetometer feature (mag_mag / mag_rate_mag),
      - frequency features of the acc/gyr *magnitude* streams — spectral scalars
        + amplitude-invariant bandpower *ratios* (sb_r), not absolute energies,
      - magnitude + jerk time-domain stats.
    Deliberately excludes body/gravity spectral + absolute subband energies so the
    branch stays small and complementary to Branch A (linear-head friendly)."""
    idx = []
    for i, c in enumerate(columns):
        if c in ("label", "__loc"):
            continue
        is_mag = c.startswith(MAG_STREAMS)
        is_freq_mag = c.startswith(FREQ_STREAMS) and ("__freq_" in c or "__sb_r_" in c)
        is_mag_time = c.startswith(MAGNITUDE_TIME)
        if is_mag or is_freq_mag or is_mag_time:
            idx.append(i)
    return idx


def make_head(name: str):
    if name == "logreg":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                                    class_weight="balanced"))])
    if name == "ridge":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced"))])
    if name == "elasticnet":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", SGDClassifier(loss="log_loss", penalty="elasticnet",
                                               l1_ratio=0.15, alpha=1e-4,
                                               class_weight="balanced", max_iter=50,
                                               random_state=0))])
    if name == "lgbm_aux":          # shallow, regularized — auxiliary only
        return lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=400,
                                  learning_rate=0.03, max_depth=3, num_leaves=8,
                                  min_child_samples=200, feature_fraction=0.5,
                                  reg_lambda=5.0, class_weight="balanced",
                                  n_jobs=-1, verbosity=-1, random_state=0)
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split.npy")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol/FREQMAG_RESULTS.md")
    args = ap.parse_args()
    t0 = time.time()
    from shl2026 import track, evaluate_predictions

    import pandas as pd
    cols = list(pd.read_parquet(args.feat_dir / "train" / "Bag.parquet").columns)
    sel = select_freq_mag(cols)
    sel_names = [cols[i] for i in sel]
    Ftr = load_feats(args.feat_dir, "train")[:, sel]
    Fva = load_feats(args.feat_dir, "validation")[:, sel]
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, _ = load_labels(args.feat_dir, "validation")
    assign, _ = load_split_with_location_map(args.split, args.feat_dir)
    assert len(assign) == len(yva) == len(Fva), "split/val length mismatch"
    fm, tm, sm = assign == FIT, assign == TUNE, assign == TEST

    Xfit = np.concatenate([Ftr, Fva[fm]]); yfit = np.concatenate([ytr, yva[fm]])
    Xtune, ytune = Fva[tm], yva[tm]
    Xtest, ytest = Fva[sm], yva[sm]
    n_mag = sum(c.startswith(MAG_STREAMS) for c in sel_names)
    print(f"[freqmag] selected {len(sel)}/{len(cols)-1} features "
          f"({n_mag} magnetometer) | fit={len(yfit)} tune={len(ytune)} test={len(ytest)}",
          flush=True)

    rows = []
    for head in ("logreg", "ridge", "elasticnet", "lgbm_aux"):
        clf = make_head(head)
        clf.fit(Xfit, yfit)
        r_te = evaluate_predictions(ytest, clf.predict(Xtest))
        r_tu = evaluate_predictions(ytune, clf.predict(Xtune))
        gap = r_tu.macro_f1 - r_te.macro_f1
        rows.append((head, r_te.macro_f1, r_tu.macro_f1, gap,
                     r_te.per_class_f1[7], r_te.per_class_f1[8]))
        print(f"  {head:12s} TEST={r_te.macro_f1:.4f} TUNE={r_tu.macro_f1:.4f} "
              f"gap={gap:+.4f} Train={r_te.per_class_f1[7]:.3f} "
              f"Subway={r_te.per_class_f1[8]:.3f}", flush=True)
        with track("mdaniol", run_name=f"freqmag_{head}", seed=0, params_path=None,
                   params={"head": head, "n_features": len(sel), "n_magnetometer": n_mag},
                   tags={"phase": "freqmag", "branch": "freq_mag"}) as run:
            run.log_eval(r_te, prefix="test_")
            run.log_eval(r_tu, prefix="tune_")
            run.log_metrics({"selection_lock_gap": gap})

    hdr = (f"# Step 2 — frequency + magnetometer branch ({len(sel)} features, "
           f"{n_mag} magnetometer; lock-test=TEST, selection=TUNE)\n\n"
           "| head | TEST macro-F1 | TUNE macro-F1 | sel-lock gap | Train F1 | Subway F1 |\n"
           "|---|---|---|---|---|---|\n")
    body = "".join(f"| {h} | {te:.4f} | {tu:.4f} | {g:+.4f} | {tr:.3f} | {sb:.3f} |\n"
                   for h, te, tu, g, tr, sb in rows)
    args.out.write_text(hdr + body)
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
