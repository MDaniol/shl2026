#!/usr/bin/env python3
"""Location mixture-of-experts experiment (plan Steps 4-6), reuse-first.

Test location is UNKNOWN, so per-location models are *experts* combined by a
router, never hard-routed in the submission. This runner builds — under the
existing split protocol (FIT/TUNE/TEST from val_split.npy) — a global model and
Bag/Hips/Torso experts, trains a 3-class location router, and compares:

  global · oracle(true loc) · uniform avg · hard-router · soft-router ·
  global+soft(β∈{.25,.40,.50,.70} + adaptive)

on BOTH the selection split (TUNE) and the lock-test (TEST), reporting macro-F1,
per-class F1 (incl. Train/Subway), and the selection-lock gap. Gate G1 = oracle
gain over global; G2 = router reliability. Everything reuses existing pieces:
  probe_fusion.{load_feats,load_emb,load_labels,calibrate,aligned_proba}
  split.load_split_with_location_map · moe_combine.* · shl2026.{track,evaluate_predictions}

Usage:
    python moe_experiment.py --rep handcrafted
    python moe_experiment.py --rep fusion --emb mantisv2_V1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_feats, load_emb, load_labels, calibrate, aligned_proba  # noqa: E402
from split import load_split_with_location_map, LOCATIONS  # noqa: E402
import moe_combine as moe  # noqa: E402

EXPERT_LOCS = moe.EXPERT_LOCS         # ("Bag","Hips","Torso")
CLASSES = moe.CLASSES                 # 1..8
FIT, TUNE, TEST = 0, 1, 2


def loc_array(feat_dir: Path, split: str) -> np.ndarray:
    """Per-row location label, LOCATIONS-order (matches load_feats/load_emb/val_split)."""
    arrs = []
    for loc in LOCATIONS:
        n = len(pd.read_parquet(feat_dir / split / f"{loc}.parquet", columns=["label"]))
        arrs.append(np.full(n, loc))
    return np.concatenate(arrs)


def load_rep(rep: str, split: str, feat_dir: Path, emb_root: Path, emb: str | None) -> np.ndarray:
    if rep == "handcrafted":
        return load_feats(feat_dir, split)
    if rep == "emb":
        return load_emb(emb_root / emb, split)
    if rep == "fusion":
        return np.concatenate([load_emb(emb_root / emb, split), load_feats(feat_dir, split)], 1)
    raise ValueError(rep)


def _lgbm(num_class=8, n_est=2000, leaves=63):
    return lgb.LGBMClassifier(objective="multiclass", num_class=num_class, n_estimators=n_est,
                              learning_rate=0.05, num_leaves=leaves, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8,
                              class_weight="balanced", n_jobs=-1, verbosity=-1)


def fit_probs(Xfit, yfit, Xcal, ycal, evals: dict) -> dict:
    """Fit 8-class LGBM, calibrate on (Xcal,ycal), return aligned calibrated
    probabilities on each matrix in `evals` ({name: X})."""
    clf = _lgbm()
    clf.fit(Xfit, yfit, eval_set=[(Xcal, ycal)], eval_metric="multi_logloss",
            callbacks=[early_stopping(100)])
    w = calibrate(clf.predict_proba(Xcal), clf.classes_, ycal)
    return {k: aligned_proba(clf, X, w) for k, X in evals.items()}


def fit_router(Xfit, loc_fit, evals: dict) -> dict:
    """3-class Bag/Hips/Torso router; return posteriors q (n,3) aligned to EXPERT_LOCS."""
    le = {l: i for i, l in enumerate(EXPERT_LOCS)}
    y = np.array([le[l] for l in loc_fit])
    clf = _lgbm(num_class=len(EXPERT_LOCS), n_est=600, leaves=31)
    clf.fit(Xfit, y)
    col = {c: i for i, c in enumerate(clf.classes_)}

    def q(X):
        P = clf.predict_proba(X)
        out = np.zeros((len(P), len(EXPERT_LOCS)))
        for j in range(len(EXPERT_LOCS)):
            if j in col:
                out[:, j] = P[:, col[j]]
        return out / (out.sum(1, keepdims=True) + 1e-12)
    return {k: q(X) for k, X in evals.items()}


def onehot_loc(loc_arr) -> np.ndarray:
    le = {l: i for i, l in enumerate(EXPERT_LOCS)}
    o = np.zeros((len(loc_arr), len(EXPERT_LOCS)))
    for i, l in enumerate(loc_arr):
        o[i, le[l]] = 1.0
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--rep", choices=["handcrafted", "emb", "fusion"], default="handcrafted")
    ap.add_argument("--emb", default=None, help="<model>_<variant> for rep=emb/fusion")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split.npy")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol/MOE_RESULTS.md")
    args = ap.parse_args()
    t0 = time.time()
    if args.rep != "handcrafted" and not args.emb:
        ap.error("--emb required for rep=emb/fusion")

    # --- load representation + labels + per-row location & split ---------------
    Xtr = load_rep(args.rep, "train", args.feat_dir, args.emb_root, args.emb)
    Xva = load_rep(args.rep, "validation", args.feat_dir, args.emb_root, args.emb)
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, _ = load_labels(args.feat_dir, "validation")
    tr_loc = loc_array(args.feat_dir, "train")
    va_loc = loc_array(args.feat_dir, "validation")
    assign, _ = load_split_with_location_map(args.split, args.feat_dir)
    assert len(assign) == len(yva) == len(Xva), "split/val length mismatch"
    fit_m, tune_m, test_m = assign == FIT, assign == TUNE, assign == TEST

    # pooled selection (TUNE) and lock (TEST) matrices + their true locations
    Xtune, ytune, loc_tune = Xva[tune_m], yva[tune_m], va_loc[tune_m]
    Xtest, ytest, loc_test = Xva[test_m], yva[test_m], va_loc[test_m]
    evals = {"tune": Xtune, "test": Xtest}
    print(f"[moe] rep={args.rep} emb={args.emb} | fit={fit_m.sum()} tune={len(ytune)} "
          f"test={len(ytest)} (test locs: {sorted(set(loc_test))})", flush=True)

    # --- global model ----------------------------------------------------------
    Pg = fit_probs(np.concatenate([Xtr, Xva[fit_m]]), np.concatenate([ytr, yva[fit_m]]),
                   Xtune, ytune, evals)
    print(f"  [global] fitted ({time.time()-t0:.0f}s)", flush=True)

    # --- per-location experts (fit on loc data, eval on POOLED tune/test) -------
    Pe = {"tune": {}, "test": {}}
    for loc in EXPERT_LOCS:
        fm = fit_m & (va_loc == loc)
        Xfit_e = np.concatenate([Xtr[tr_loc == loc], Xva[fm]])
        yfit_e = np.concatenate([ytr[tr_loc == loc], yva[fm]])
        cm = tune_m & (va_loc == loc)
        probs = fit_probs(Xfit_e, yfit_e, Xva[cm], yva[cm], evals)
        Pe["tune"][loc], Pe["test"][loc] = probs["tune"], probs["test"]
        print(f"  [expert {loc}] fitted ({time.time()-t0:.0f}s)", flush=True)

    # --- location router (exclude Hand) ----------------------------------------
    rm_tr = np.isin(tr_loc, EXPERT_LOCS)
    rm_va = fit_m & np.isin(va_loc, EXPERT_LOCS)
    Q = fit_router(np.concatenate([Xtr[rm_tr], Xva[rm_va]]),
                   np.concatenate([tr_loc[rm_tr], va_loc[rm_va]]), evals)
    le = {l: i for i, l in enumerate(EXPERT_LOCS)}
    router_acc = float(np.mean(Q["test"].argmax(1) == np.array([le[l] for l in loc_test])))
    print(f"  [router] test accuracy = {router_acc:.4f}", flush=True)

    # --- build every config on tune + test -------------------------------------
    from shl2026 import track, evaluate_predictions

    def configs_for(split):
        E = moe.stack_experts(Pe[split]); g = Pg[split]; q = Q[split]
        qo = onehot_loc(loc_tune if split == "tune" else loc_test)
        c = {"global": g,
             "oracle": moe.soft_mix(E, qo),
             "uniform": moe.uniform_mix(E),
             "global+uniform": moe.global_fallback(g, moe.uniform_mix(E), 0.5),
             "hard_router": moe.hard_mix(E, q),
             "soft_router": moe.soft_mix(E, q)}
        for b in (0.25, 0.40, 0.50, 0.70):
            c[f"global+soft_b{b:.2f}"] = moe.global_fallback(g, moe.soft_mix(E, q), b)
        c["global+soft_adaptive"] = moe.global_fallback(g, moe.soft_mix(E, q), moe.adaptive_beta(q))
        return c

    cfg_tune, cfg_test = configs_for("tune"), configs_for("test")
    rows, results = [], {}
    for name in cfg_test:
        r_te = evaluate_predictions(ytest, moe.predict(cfg_test[name]))
        r_tu = evaluate_predictions(ytune, moe.predict(cfg_tune[name]))
        gap = r_tu.macro_f1 - r_te.macro_f1
        results[name] = {"test": r_te, "tune": r_tu, "gap": gap}
        rows.append((name, r_te.macro_f1, r_tu.macro_f1, gap,
                     r_te.per_class_f1[7], r_te.per_class_f1[8]))
        print(f"  {name:24s} TEST={r_te.macro_f1:.4f} TUNE={r_tu.macro_f1:.4f} "
              f"gap={gap:+.4f} Train={r_te.per_class_f1[7]:.3f} Subway={r_te.per_class_f1[8]:.3f}")
        with track("mdaniol", run_name=f"moe_{args.rep}_{name}", seed=0, params_path=None,
                   params={"rep": args.rep, "emb": args.emb, "config": name},
                   tags={"phase": "moe", "branch": "location"}) as run:
            run.log_eval(r_te, prefix="test_")
            run.log_eval(r_tu, prefix="tune_")
            run.log_metrics({"selection_lock_gap": gap, "router_acc": router_acc})

    # --- decision summary + results table --------------------------------------
    g_te = results["global"]["test"].macro_f1
    oracle_gain = results["oracle"]["test"].macro_f1 - g_te
    best = max((n for n in results), key=lambda n: results[n]["test"].macro_f1)
    print(f"\n=== DECISION (lock-test) ===")
    print(f"  global={g_te:.4f}  oracle={results['oracle']['test'].macro_f1:.4f} "
          f"(G1 gain {oracle_gain:+.4f})  router_acc={router_acc:.3f}")
    print(f"  best config on TEST: {best} = {results[best]['test'].macro_f1:.4f} "
          f"({'BEATS' if results[best]['test'].macro_f1 > g_te else 'does NOT beat'} global)")

    hdr = (f"# Location MoE results (rep={args.rep}, emb={args.emb}); "
           f"router test-acc={router_acc:.3f}; lock-test=TEST, selection=TUNE\n\n"
           "| config | TEST macro-F1 | TUNE macro-F1 | sel-lock gap | Train F1 | Subway F1 |\n"
           "|---|---|---|---|---|---|\n")
    body = "".join(f"| {n} | {te:.4f} | {tu:.4f} | {g:+.4f} | {tr:.3f} | {sb:.3f} |\n"
                   for n, te, tu, g, tr, sb in rows)
    args.out.write_text(hdr + body)
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
