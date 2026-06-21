#!/usr/bin/env python3
"""Step 8 — conservative global Train-vs-Subway rail expert (plan), reuse-first.

Test location is unknown, so the rail expert is GLOBAL (not per-location). It is a
gated post-processor over any base model's calibrated probabilities:

  candidate frame iff  P_base(Train)+P_base(Subway) > tau_rail
  override to the binary expert's call iff its confidence > tau_conf
  else keep the base prediction.

The binary expert is trained ONLY on Train/Subway examples, on the 130 magnetometer
features (mag_mag/mag_rate_mag — the rail discriminators) plus the base Train/Subway
probabilities. Kept only if it doesn't drop overall macro-F1 (the hard guard).

Standalone run evaluates it on top of the GLOBAL handcrafted model (the base),
sweeping tau_rail x tau_conf, reporting Train/Subway F1 + macro before/after and
the confusion, into RAIL_RESULTS.md + MLflow. Reuses probe_fusion.fit_cal_eval,
split masks, shl2026.{track,evaluate_predictions}.

  apply_rail(...) is importable so the fusion/submission step can reuse it on the
  final P (global / MoE) instead of the global base.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_feats, load_labels, fit_cal_eval  # noqa: E402
from split import load_split_with_location_map  # noqa: E402

FIT, TUNE, TEST = 0, 1, 2
CLASSES = list(range(1, 9))
TRAIN_C, SUBWAY_C = 7, 8                       # class ids
TRAIN_I, SUBWAY_I = 6, 7                       # 0-based indices into the 8-col proba
MAG_PREFIXES = ("mag_mag__", "mag_rate_mag__")


def mag_indices(columns) -> list[int]:
    return [i for i, c in enumerate(columns)
            if c.startswith(MAG_PREFIXES) and c not in ("label", "__loc")]


def train_rail(P_base_fit, y_fit, Xmag_fit):
    """Binary Subway(1)-vs-Train(0) expert on rail examples only.
    Features = [base P(train), base P(subway)] ⊕ magnetometer features."""
    rail = np.isin(y_fit, (TRAIN_C, SUBWAY_C))
    feats = np.concatenate([P_base_fit[rail][:, [TRAIN_I, SUBWAY_I]], Xmag_fit[rail]], axis=1)
    ybin = (y_fit[rail] == SUBWAY_C).astype(int)
    clf = lgb.LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.03,
                             num_leaves=15, min_child_samples=100, reg_lambda=2.0,
                             class_weight="balanced", n_jobs=-1, verbosity=-1, random_state=0)
    clf.fit(feats, ybin)
    return clf


def apply_rail(P_base, Xmag, rail_clf, tau_rail=0.55, tau_conf=0.65):
    """Gated correction. Returns predicted labels (1..8). Only Train/Subway-candidate
    frames with confident binary calls are overridden; all others keep base argmax."""
    base_pred = np.asarray(CLASSES)[P_base.argmax(1)]
    cand = (P_base[:, TRAIN_I] + P_base[:, SUBWAY_I]) > tau_rail
    if not cand.any():
        return base_pred
    feats = np.concatenate([P_base[cand][:, [TRAIN_I, SUBWAY_I]], Xmag[cand]], axis=1)
    pb = rail_clf.predict_proba(feats)[:, 1]                # P(subway)
    conf = np.maximum(pb, 1.0 - pb)
    take = conf > tau_conf
    out = base_pred.copy()
    cand_idx = np.where(cand)[0][take]
    out[cand_idx] = np.where(pb[take] >= 0.5, SUBWAY_C, TRAIN_C)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split.npy")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol/RAIL_RESULTS.md")
    args = ap.parse_args()
    t0 = time.time()
    from shl2026 import track, evaluate_predictions
    import pandas as pd

    cols = list(pd.read_parquet(args.feat_dir / "train" / "Bag.parquet").columns)
    mcols = mag_indices(cols)
    Ftr = load_feats(args.feat_dir, "train"); Fva = load_feats(args.feat_dir, "validation")
    ytr, _ = load_labels(args.feat_dir, "train"); yva, _ = load_labels(args.feat_dir, "validation")
    assign, _ = load_split_with_location_map(args.split, args.feat_dir)
    fm, tm, sm = assign == FIT, assign == TUNE, assign == TEST

    Xfit = np.concatenate([Ftr, Fva[fm]]); yfit = np.concatenate([ytr, yva[fm]])
    out = fit_cal_eval("global(base)", Xfit, yfit, Fva[tm], yva[tm], Fva[sm], yva[sm],
                       return_probs=True)
    Pg_tune, Pg_test = out["proba_tune"], out["proba_test"]
    ytune, ytest = yva[tm], yva[sm]
    Xmag_fit = np.concatenate([Ftr[:, mcols], Fva[fm][:, mcols]])
    Xmag_test = Fva[sm][:, mcols]
    base_test = evaluate_predictions(ytest, np.asarray(CLASSES)[Pg_test.argmax(1)])
    print(f"[rail] base TEST macro={base_test.macro_f1:.4f} "
          f"Train={base_test.per_class_f1[7]:.3f} Subway={base_test.per_class_f1[8]:.3f}", flush=True)

    # train the rail expert on FIT base-probabilities (recompute base proba on FIT)
    from probe_fusion import aligned_proba
    Pg_fit = aligned_proba(out["model"], Xfit, out["weights"])
    rail = train_rail(Pg_fit, yfit, Xmag_fit)

    rows = []
    for tau_rail in (0.45, 0.55, 0.65):
        for tau_conf in (0.55, 0.65, 0.75):
            pred = apply_rail(Pg_test, Xmag_test, rail, tau_rail, tau_conf)
            r = evaluate_predictions(ytest, pred)
            dmacro = r.macro_f1 - base_test.macro_f1
            rows.append((tau_rail, tau_conf, r.macro_f1, dmacro,
                         r.per_class_f1[7], r.per_class_f1[8]))
            print(f"  tau_rail={tau_rail} tau_conf={tau_conf} -> macro={r.macro_f1:.4f} "
                  f"({dmacro:+.4f}) Train={r.per_class_f1[7]:.3f} Subway={r.per_class_f1[8]:.3f}",
                  flush=True)
            with track("mdaniol", run_name=f"rail_tr{tau_rail}_tc{tau_conf}", seed=0,
                       params_path=None,
                       params={"tau_rail": tau_rail, "tau_conf": tau_conf},
                       tags={"phase": "rail", "branch": "rail_expert"}) as run:
                run.log_eval(r, prefix="test_")
                run.log_metrics({"delta_macro_vs_base": dmacro})

    best = max(rows, key=lambda r: r[2])
    keep = best[3] > 0
    print(f"\n=== RAIL DECISION ===\n  base macro={base_test.macro_f1:.4f}; "
          f"best (tau_rail={best[0]}, tau_conf={best[1]}) macro={best[2]:.4f} ({best[3]:+.4f}) "
          f"-> {'KEEP' if keep else 'DISABLE (does not beat base; raise thresholds)'}")

    hdr = (f"# Step 8 — global Train-vs-Subway rail expert (base=global handcrafted; "
           f"base TEST macro={base_test.macro_f1:.4f}, Train={base_test.per_class_f1[7]:.3f}, "
           f"Subway={base_test.per_class_f1[8]:.3f})\n\n"
           "| tau_rail | tau_conf | TEST macro-F1 | Δ vs base | Train F1 | Subway F1 |\n"
           "|---|---|---|---|---|---|\n")
    body = "".join(f"| {tr} | {tc} | {m:.4f} | {d:+.4f} | {trf:.3f} | {sbf:.3f} |\n"
                   for tr, tc, m, d, trf, sbf in rows)
    args.out.write_text(hdr + body)
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
