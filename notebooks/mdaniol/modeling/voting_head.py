#!/usr/bin/env python3
"""Calibrated soft-voting head over the top-K fused FMs (the bake-off winners).

The bake-off ranks single fused FMs; this head asks the next question: does a
*calibrated soft-vote* of the best few beat the single best? Diversity, not
routing — different FM families make different mistakes, so averaging their
calibrated posteriors can lift macro-F1 with zero new representation.

Protocol mirrors probe_fusion.py --use-split EXACTLY (fit on User1+val[FIT],
calibrate/select on TUNE, lock once on TEST), so each FM's single-model column
here reproduces its BAKEOFF_SPLIT.md lgbm(emb+hc) value (built-in sanity check).

For each FM:  [emb ⊕ 520 handcrafted] -> fit_cal_eval(return_probs) -> calibrated,
class-aligned probas on TUNE & TEST. Then combine:
  - equal vote        : mean of the calibrated probas
  - weighted vote     : per-FM weights chosen on TUNE (coordinate ascent), locked on TEST
  - weighted + recal  : per-class recalibration of the voted mixture on TUNE
Decision (gated, select-on-TUNE): KEEP the vote only if it beats the best single
FM on the lock TEST by > +0.001 macro-F1.

Reuses: probe_fusion.{load_emb,load_feats,load_labels,fit_cal_eval,calibrate,
macro_f1,CLASSES}, moe_combine.{stack_experts-style stacking}, metrics.class_report.

    python voting_head.py --embs utica_V2,mantisv2_V1 --use-split \
        --split notebooks/mdaniol/modeling/artifacts/val_split_temporal.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import (load_emb, load_feats, load_labels, fit_cal_eval,  # noqa: E402
                          calibrate, macro_f1, CLASSES)
from metrics import class_report, save_report  # noqa: E402


def fused_probas(emb, emb_root, feat_dir, ytr, yva, Ftr, Fva, fm, tm, sm):
    """Single FM, fusion head, bake-off protocol -> calibrated class-aligned
    probas on TUNE and TEST plus its locked TEST report (reproduces BAKEOFF_SPLIT)."""
    Etr = load_emb(emb_root / emb, "train")
    Eva = load_emb(emb_root / emb, "validation")
    EFtr = np.concatenate([Etr, Ftr], 1)
    EFva = np.concatenate([Eva, Fva], 1)
    Xfit = np.concatenate([EFtr, EFva[fm]]); yfit = np.concatenate([ytr, yva[fm]])
    out = fit_cal_eval(f"vote:{emb}", Xfit, yfit, EFva[tm], yva[tm],
                       EFva[sm], yva[sm], return_probs=True)
    return out  # dict: proba_tune, proba_test, rep (TEST), rep_tune, ...


def weight_search(P, y, n_fm, grid=np.linspace(0.0, 2.0, 21), rounds=4):
    """Coordinate ascent on simplex-ish FM weights maximizing macro-F1 on TUNE.
    P: (K, n, 8) calibrated probas; returns weights (K,) summing to 1."""
    w = np.ones(n_fm)
    score = lambda ww: macro_f1(y, np.asarray(CLASSES)[
        np.einsum("knc,k->nc", P, ww / ww.sum()).argmax(1)])
    best = score(w)
    for _ in range(rounds):
        improved = False
        for k in range(n_fm):
            base_v = w[k]
            for v in grid:
                w2 = w.copy(); w2[k] = v
                if w2.sum() <= 0:
                    continue
                s = score(w2)
                if s > best:
                    best, base_v = s, v
            if base_v != w[k]:
                w[k] = base_v; improved = True
        if not improved:
            break
    return w / w.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--embs", default="utica_V2,mantisv2_V1",
                    help="comma list of <model>_<variant>, top-K from BAKEOFF_SPLIT.md")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path,
                    default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--out", type=Path,
                    default=root / "notebooks/mdaniol/VOTING_HEAD_RESULTS.md")
    args = ap.parse_args()
    t0 = time.time()
    embs = [e.strip() for e in args.embs.split(",") if e.strip()]
    cls = np.asarray(CLASSES)

    # shared FIT/TUNE/TEST masks (mirror probe_fusion --use-split exactly)
    assign = np.load(args.split)
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, _ = load_labels(args.feat_dir, "validation")
    Ftr, Fva = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
    assert len(assign) == len(yva), "val_split / validation length mismatch"
    fm, tm, sm = assign == 0, assign == 1, assign == 2
    ytune, ytest = yva[tm], yva[sm]
    print(f"[voting] embs={embs} fit=User1+val[{fm.sum()}] tune={tm.sum()} test={sm.sum()}")

    # per-FM calibrated probas (single-FM TEST macro reproduces the bake-off).
    # Also keep each FM's fitted model + calibration weights so the submission can REUSE
    # them (submit_vote --from-models) instead of refitting — and so §8 snapshots the model.
    singles, Ptu_list, Pte_list, fmodels = {}, [], [], []
    for e in embs:
        o = fused_probas(e, args.emb_root, args.feat_dir, ytr, yva, Ftr, Fva, fm, tm, sm)
        singles[e] = o["rep"]["macro_f1"]
        Ptu_list.append(o["proba_tune"]); Pte_list.append(o["proba_test"])
        fmodels.append((o["model"], o["weights"]))      # (fitted LGBM, per-class cal weights)
    Ptu = np.stack(Ptu_list, 0); Pte = np.stack(Pte_list, 0)        # (K, n, 8)
    best_single = max(singles, key=singles.get)
    base = singles[best_single]

    # --- combine -----------------------------------------------------------
    rows = []  # (name, TUNE macro, TEST macro, per_class_test report)

    def add(name, mix_tu, mix_te):
        r_tu = class_report(ytune, cls[mix_tu.argmax(1)])
        r_te = class_report(ytest, cls[mix_te.argmax(1)])
        rows.append((name, r_tu["macro_f1"], r_te["macro_f1"], r_te))

    for e in embs:                                                  # single-FM rows
        i = embs.index(e)
        add(f"single:{e}", Ptu[i], Pte[i])
    eq_tu, eq_te = Ptu.mean(0), Pte.mean(0)                         # equal vote
    add("vote:equal", eq_tu, eq_te)
    w = weight_search(Ptu, ytune, len(embs))                       # weighted vote (TUNE-selected)
    wv_tu = np.einsum("knc,k->nc", Ptu, w); wv_te = np.einsum("knc,k->nc", Pte, w)
    add("vote:weighted", wv_tu, wv_te)
    cw = calibrate(wv_tu, cls, ytune)                              # per-class recal of the mixture
    add("vote:weighted+recal", wv_tu * cw, wv_te * cw)

    # --- decision (select on TUNE; gate on TEST) ---------------------------
    vote_rows = [r for r in rows if r[0].startswith("vote:")]
    best_vote = max(vote_rows, key=lambda r: r[1])                 # select on TUNE macro
    keep = best_vote[2] > base + 0.001                             # gate: beat best single on TEST
    print(f"\n[decision] best single = {best_single} TEST={base:.4f}")
    print(f"  best vote (TUNE-selected) = {best_vote[0]} "
          f"TEST={best_vote[2]:.4f} (Δ{best_vote[2]-base:+.4f}) "
          f"weights={dict(zip(embs, np.round(w, 3)))} -> {'KEEP' if keep else 'DISABLE'}")

    # --- write result table, then snapshot to MLflow (rule §8) -------------
    hdr = (f"# Soft-voting head over {embs} (temporal split, calibrated).\n"
           f"best single = **{best_single}** TEST={base:.4f}; "
           f"best vote = **{best_vote[0]}** TEST={best_vote[2]:.4f} "
           f"(Δ{best_vote[2]-base:+.4f}) **{'KEEP' if keep else 'DISABLE'}** "
           f"(selected on TUNE).\n\n"
           "| config | TUNE macro | TEST macro | per-class F1 (TEST) |\n"
           "|---|---|---|---|\n")
    body = ""
    for name, ftu, fte, rep in rows:
        pc = " ".join(f"{k[:2]}={d['f1']:.2f}" for k, d in rep["per_class"].items())
        body += f"| {name} | {ftu:.4f} | {fte:.4f} | {pc} |\n"
    args.out.write_text(hdr + body)

    from shl2026 import track
    import joblib
    json_path = args.out.with_suffix(".json")
    save_report(json_path, {name: rep for name, _, _, rep in rows})
    # snapshot the fitted models + the locked weighted+recal vote params (rule §8) so the
    # submission can reuse them (no refit). Recipe = weighted+recal, which submit_vote applies.
    model_path = args.out.with_name(f"vote_models_{'+'.join(embs)}.joblib")
    joblib.dump({"embs": embs, "models": fmodels, "vote_weights": w, "recal": cw,
                 "split": args.split.stem, "tune_macro": float(best_vote[1])}, model_path)
    with track("mdaniol", run_name=f"voting_{'+'.join(embs)}", seed=0, params_path=None,
               params={"embs": ",".join(embs), "split": args.split.stem,
                       "weights": ",".join(f"{x:.3f}" for x in w),
                       "best_single": best_single,
                       "protocol": "fit(User1+val[FIT])->cal(TUNE)->lock(TEST)"},
               tags={"phase": "voting", "branch": "lightweight_head",
                     "decision": "KEEP" if keep else "DISABLE"}) as run:
        run.log_metrics({"macro_f1": best_vote[2],                 # bare = lock-test -> leaderboard
                         "best_single_macro_f1": base,
                         "delta_vs_best_single": best_vote[2] - base,
                         "vote_tune_macro_f1": best_vote[1]})
        for k, d in best_vote[3]["per_class"].items():
            run.log_metrics({f"test_f1_{k}": d["f1"]})
        run.log_artifact(args.out)
        run.log_artifact(json_path)
        run.log_artifact(model_path)                  # fitted models + vote params (§8)
        if args.split.exists():
            run.log_artifact(args.split)              # snapshot the split file (§8)
    print(f"wrote {args.out} + {model_path.name}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
