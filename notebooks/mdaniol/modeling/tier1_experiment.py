#!/usr/bin/env python3
"""Tier-1 experiment: macro-F1 post-hoc ablation on the MOMENT-fusion base (honest split).

Compares deployable post-hocs that survive the shuffled, no-smoothing test, selecting the
best on TUNE and confirming once on the locked TEST (per AI_GUIDELINES). Reuses the exact
v1 protocol (fit on User-1+val[FIT], calibrate on val[TUNE]); the only new piece is
label-shift prior adaptation (prior_adapt). Reports per-class F1 (esp. Run).

Variants (CLASSES 1..8, eval Bag/Hips/Torso):
  raw           argmax of raw aligned posteriors
  cal (=v1)     + per-class macro-F1 calibration (current submission)
  logit_adj     logit adjustment with the TRAIN prior (tau swept on TUNE; no test use)
  mlls          label-shift adapt to the prior estimated from each set's UNLABELED marginal
  cal+mlls      MLLS-adapt, then re-calibrate on the adapted TUNE
  [oracle]      adapt to the TRUE test prior — analysis-only upper bound (NOT deployable)

MLLS uses only the eval-set posteriors (model outputs on inputs), never labels — valid on a
shuffled test and simulates the real submission. Output -> TIER1_RESULTS.md + MLflow.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import fit_cal_eval, aligned_proba, calibrate, load_labels, CLASSES  # noqa: E402
from split import load_split_with_location_map  # noqa: E402
from submit_fusion import fuse  # noqa: E402
import prior_adapt as pa  # noqa: E402

FIT, TUNE, TEST = 0, 1, 2
BHT = ("Bag", "Hips", "Torso")
CLS = np.asarray(CLASSES)
RUN = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--emb", default="moment-small_V1")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol/TIER1_RESULTS.md")
    args = ap.parse_args()
    t0 = time.time()
    from shl2026 import track, evaluate_predictions

    Xtr = fuse(args.feat_dir, args.emb_root, args.emb, "train")
    Xva = fuse(args.feat_dir, args.emb_root, args.emb, "validation")
    ytr, _ = load_labels(args.feat_dir, "train"); yva, _ = load_labels(args.feat_dir, "validation")
    assign, loc_off = load_split_with_location_map(args.split, args.feat_dir)
    va_loc = np.empty(len(assign), dtype=object)
    for loc, (s, e) in loc_off.items():
        va_loc[s:e] = loc
    bht = np.isin(va_loc.astype(str), BHT)
    fit_m, tune_m, test_m = assign == FIT, (assign == TUNE) & bht, (assign == TEST) & bht
    Xfit = np.concatenate([Xtr, Xva[fit_m]]); yfit = np.concatenate([ytr, yva[fit_m]])
    ytune, ytest = yva[tune_m], yva[test_m]

    out = fit_cal_eval(f"base({args.emb})", Xfit, yfit, Xva[tune_m], ytune,
                       Xva[test_m], ytest, return_probs=True)
    clf, w = out["model"], out["weights"]
    raw_tu = aligned_proba(clf, Xva[tune_m], None)      # uncalibrated posteriors
    raw_te = aligned_proba(clf, Xva[test_m], None)
    cal_tu, cal_te = out["proba_tune"], out["proba_test"]   # = raw * w, normalized (v1)
    pi_train = pa.class_prior(yfit)
    print(f"[tier1] fit={len(yfit)} tune={len(ytune)} test={len(ytest)} | "
          f"pi_train Run={pi_train[RUN-1]:.3f}", flush=True)

    def ev(y, P):
        r = evaluate_predictions(y, CLS[P.argmax(1)])
        return r.macro_f1, r.per_class_f1[RUN], r

    # --- deployable variants: (tune_pred_proba, test_pred_proba) -------------------------
    variants = {}
    variants["raw"] = (raw_tu, raw_te)
    variants["cal(v1)"] = (cal_tu, cal_te)
    # logit adjustment: sweep tau on TUNE, lock on TEST
    taus = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    best_tau = max(taus, key=lambda t: ev(ytune, pa.logit_adjust(raw_tu, pi_train, t))[0])
    variants[f"logit_adj(t={best_tau})"] = (pa.logit_adjust(raw_tu, pi_train, best_tau),
                                            pa.logit_adjust(raw_te, pi_train, best_tau))
    # MLLS: estimate each set's prior from its OWN unlabeled marginal, then adapt
    pi_tu = pa.estimate_prior_mlls(raw_tu, pi_train); pi_te = pa.estimate_prior_mlls(raw_te, pi_train)
    mlls_tu, mlls_te = pa.adapt(raw_tu, pi_train, pi_tu), pa.adapt(raw_te, pi_train, pi_te)
    variants["mlls"] = (mlls_tu, mlls_te)
    # cal + mlls: re-calibrate macro-F1 multipliers on the MLLS-adapted TUNE
    w2 = calibrate(mlls_tu, CLS, ytune)
    variants["cal+mlls"] = (mlls_tu * w2 / (mlls_tu * w2).sum(1, keepdims=True),
                            mlls_te * w2 / (mlls_te * w2).sum(1, keepdims=True))

    rows = {}
    for name, (Ptu, Pte) in variants.items():
        mt, rt, _ = ev(ytune, Ptu); ms, rs, _ = ev(ytest, Pte)
        rows[name] = (mt, ms, rt, rs)
        print(f"  {name:16s} TUNE={mt:.4f} TEST={ms:.4f} | Run TUNE={rt:.3f} TEST={rs:.3f}", flush=True)
    # analysis-only oracle (true TEST prior) — upper bound, NOT deployable
    orc = pa.adapt(raw_te, pi_train, pa.class_prior(ytest))
    om, orun, _ = ev(ytest, orc)
    print(f"  {'[oracle prior]':16s} TEST={om:.4f} Run={orun:.3f}  (upper bound, not deployable)", flush=True)

    base = rows["cal(v1)"][1]
    best = max((n for n in rows if n != "cal(v1)"), key=lambda n: rows[n][0])  # select on TUNE
    keep = rows[best][1] > base + 0.001
    print(f"\n=== TIER1 DECISION ===\n  v1 TEST={base:.4f}; best deployable (by TUNE)={best} "
          f"-> TEST={rows[best][1]:.4f} (Δ{rows[best][1]-base:+.4f}) "
          f"-> {'ADOPT' if keep else 'keep v1'}; oracle-prior ceiling TEST={om:.4f}")

    hdr = (f"# Tier-1 macro-F1 post-hoc ({args.emb}, temporal split, Bag/Hips/Torso; select=TUNE, "
           f"lock=TEST). v1={base:.4f}; best deployable **{best}** TEST {rows[best][1]:.4f} "
           f"(oracle-prior ceiling {om:.4f}).\n\n"
           "| variant | TUNE macro | TEST macro | Run F1 TUNE | Run F1 TEST |\n|---|---|---|---|---|\n")
    body = "".join(f"| {'**'+n+'**' if n == best else n} | {a:.4f} | {b:.4f} | {c:.3f} | {d:.3f} |\n"
                   for n, (a, b, c, d) in rows.items())
    body += f"| _oracle-prior (n/d)_ | — | {om:.4f} | — | {orun:.3f} |\n"
    args.out.write_text(hdr + body)

    with track("mdaniol", run_name=f"tier1_{best}", seed=0, params_path=None,
               params={"emb": args.emb, "split": args.split.stem, "best": best, "tau": best_tau},
               tags={"phase": "tier1", "branch": "posthoc",
                     "decision": "ADOPT" if keep else "KEEP_V1"}) as run:
        run.log_metrics({"macro_f1": rows[best][1], "v1_macro_f1": base,
                         "delta_vs_v1": rows[best][1] - base, "oracle_prior_ceiling": om,
                         "run_f1_best": rows[best][3], "run_f1_v1": rows["cal(v1)"][3]})
        run.log_artifact(args.out)
    print(f"wrote {args.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
