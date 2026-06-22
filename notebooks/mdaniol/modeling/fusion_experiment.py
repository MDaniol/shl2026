#!/usr/bin/env python3
"""Step 7 — branch-level probability fusion (plan), reuse-first & leakage-safe.

Combines the per-branch class probabilities (NOT one giant concat-feature model —
that is the report's failure mode). Branches each produce calibrated, class-aligned
probabilities; we then compare:

  * each branch alone,
  * weighted average (equal weights),
  * a logistic-regression STACKER on the branches' probabilities.

Leakage discipline (standard stacking): base branches are fit on FIT, produce
probabilities on TUNE/TEST; the stacker is fit on the **TUNE** probabilities only
and read once on TEST. Selection is on TUNE; TEST is the lock. Eval is Bag/Hips/
Torso, per-window (see GUARDRAILS.md / AI_GUIDELINES.md).

Branches (handcrafted by default; add the FM branch with --emb):
  global_hc : LightGBM on the 520 features (Branch A), calibrated   [fit_cal_eval]
  freqmag   : logistic regression on the freq+mag subset            [linear, predict_proba]
  fm        : LightGBM on emb / emb⊕handcrafted (optional, --emb)   [fit_cal_eval]

Usage:  python fusion_experiment.py            # global_hc + freqmag
        python fusion_experiment.py --emb mantisv2_V1 --fm-rep fusion
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_feats, load_emb, load_labels, fit_cal_eval, aligned_proba, CLASSES  # noqa: E402
from split import load_split_with_location_map  # noqa: E402
from freq_mag_branch import select_freq_mag  # noqa: E402

FIT, TUNE, TEST = 0, 1, 2
BHT = ("Bag", "Hips", "Torso")


def bht_masks(assign, va_loc):
    """FIT keeps all locations; TUNE/TEST are Bag/Hips/Torso only (mirror the test)."""
    bht = np.isin(va_loc.astype(str), BHT)
    return assign == FIT, (assign == TUNE) & bht, (assign == TEST) & bht


def linear_branch_probs(Xfit, yfit, Xtune, Xtest):
    """Calibration-free linear branch: standardized logistic regression -> aligned probs."""
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(Xfit, yfit)
    return aligned_proba(pipe, Xtune), aligned_proba(pipe, Xtest)


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split.npy")
    ap.add_argument("--emb", default=None, help="add an FM branch from this cached set")
    ap.add_argument("--fm-rep", choices=["emb", "fusion"], default="fusion",
                    help="FM branch input: embeddings, or embeddings⊕handcrafted")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol/FUSION_RESULTS.md")
    args = ap.parse_args()
    t0 = time.time()
    from shl2026 import track, evaluate_predictions

    # --- shared data + BHT split masks -----------------------------------------
    Ftr, Fva = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, _ = load_labels(args.feat_dir, "validation")
    assign, loc_off = load_split_with_location_map(args.split, args.feat_dir)
    va_loc = np.empty(len(assign), dtype=object)
    for loc, (s, e) in loc_off.items():
        va_loc[s:e] = loc
    fm_, tm_, sm_ = bht_masks(assign, va_loc)
    ytune, ytest = yva[tm_], yva[sm_]
    yfit = np.concatenate([ytr, yva[fm_]])
    print(f"[fusion] fit={fm_.sum()+len(ytr)} tune={len(ytune)} test={len(ytest)} (BHT)", flush=True)

    branches = {}   # name -> (P_tune (n_tune,8), P_test (n_test,8))

    # branch: global handcrafted (Branch A), calibrated
    out = fit_cal_eval("global_hc", np.concatenate([Ftr, Fva[fm_]]), yfit,
                       Fva[tm_], ytune, Fva[sm_], ytest, return_probs=True)
    branches["global_hc"] = (out["proba_tune"], out["proba_test"])

    # branch: freq+mag linear
    sel = select_freq_mag(list(__import__("pandas").read_parquet(
        args.feat_dir / "train" / "Bag.parquet").columns))
    branches["freqmag"] = linear_branch_probs(
        np.concatenate([Ftr[:, sel], Fva[fm_][:, sel]]), yfit, Fva[tm_][:, sel], Fva[sm_][:, sel])

    # branch: FM (optional)
    if args.emb:
        Etr, Eva = load_emb(args.emb_root / args.emb, "train"), load_emb(args.emb_root / args.emb, "validation")
        if args.fm_rep == "fusion":
            Xtr_fm = np.concatenate([Etr, Ftr], 1); Xva_fm = np.concatenate([Eva, Fva], 1)
        else:
            Xtr_fm, Xva_fm = Etr, Eva
        o = fit_cal_eval(f"fm_{args.emb}", np.concatenate([Xtr_fm, Xva_fm[fm_]]), yfit,
                         Xva_fm[tm_], ytune, Xva_fm[sm_], ytest, return_probs=True)
        branches[f"fm_{args.emb}"] = (o["proba_tune"], o["proba_test"])

    names = list(branches)
    cls = np.asarray(CLASSES)

    # --- fusion configs (deployable) -------------------------------------------
    Ptune = {n: branches[n][0] for n in names}
    Ptest = {n: branches[n][1] for n in names}
    configs_tune, configs_test = {}, {}
    for n in names:                                   # each branch alone
        configs_tune[n], configs_test[n] = Ptune[n], Ptest[n]
    configs_tune["weighted_avg"] = np.mean([Ptune[n] for n in names], axis=0)
    configs_test["weighted_avg"] = np.mean([Ptest[n] for n in names], axis=0)

    # logreg stacker: meta-features = concatenated branch probabilities.
    # Fit on TUNE only (base branches already fit on FIT) -> no leakage; read TEST once.
    Mtune = np.concatenate([Ptune[n] for n in names], axis=1)
    Mtest = np.concatenate([Ptest[n] for n in names], axis=1)
    stacker = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Mtune, ytune)
    configs_tune["logreg_stacker"] = aligned_proba(stacker, Mtune)
    configs_test["logreg_stacker"] = aligned_proba(stacker, Mtest)

    # --- evaluate; select on TUNE, confirm on TEST -----------------------------
    rows, results = [], {}
    for name in configs_test:
        r_tu = evaluate_predictions(ytune, cls[configs_tune[name].argmax(1)])
        r_te = evaluate_predictions(ytest, cls[configs_test[name].argmax(1)])
        gap = r_tu.macro_f1 - r_te.macro_f1
        results[name] = {"tune": r_tu, "test": r_te}
        rows.append((name, r_te.macro_f1, r_tu.macro_f1, gap, r_te.per_class_f1[7], r_te.per_class_f1[8]))
        print(f"  {name:18s} TEST={r_te.macro_f1:.4f} TUNE={r_tu.macro_f1:.4f} gap={gap:+.4f} "
              f"Train={r_te.per_class_f1[7]:.3f} Subway={r_te.per_class_f1[8]:.3f}", flush=True)
        with track("mdaniol", run_name=f"fusion_{name}", seed=0, params_path=None,
                   params={"branches": "+".join(names), "config": name, "split": args.split.stem},
                   tags={"phase": "fusion", "branch": "fusion"}) as run:
            run.log_eval(r_te, prefix="test_")
            run.log_eval(r_tu, prefix="tune_")
            run.log_metrics({"macro_f1": r_te.macro_f1, "selection_lock_gap": gap})

    g_te = results["global_hc"]["test"].macro_f1
    best = max(results, key=lambda n: results[n]["tune"].macro_f1)        # select on TUNE
    best_te = results[best]["test"].macro_f1
    print(f"\n=== FUSION DECISION ===\n  global_hc TEST={g_te:.4f}; best (selected on TUNE)="
          f"{best} -> TEST={best_te:.4f} ({'BEATS' if best_te > g_te else 'does NOT beat'} global_hc)")

    hdr = (f"# Step 7 — branch fusion (branches: {', '.join(names)}; eval Bag/Hips/Torso; "
           f"selection=TUNE, lock=TEST). Best (by TUNE): **{best}** "
           f"(TEST {best_te:.4f} vs global_hc {g_te:.4f}).\n\n"
           "| config | TEST macro-F1 | TUNE macro-F1 | sel-lock gap | Train F1 | Subway F1 |\n"
           "|---|---|---|---|---|---|\n")
    body = "".join(f"| {'**'+n+'**' if n == best else n} | {te:.4f} | {tu:.4f} | {g:+.4f} | {tr:.3f} | {sb:.3f} |\n"
                   for n, te, tu, g, tr, sb in rows)
    args.out.write_text(hdr + body)
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
