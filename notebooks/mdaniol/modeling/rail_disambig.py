#!/usr/bin/env python3
"""E-RAIL — gated Train↔Subway disambiguator (data-justified by ROBUSTNESS_DIAG: Subway→Train errors
are 99% top-2-recoverable, Train→Subway 80% — the model ranks them adjacent and just picks the wrong
one). A dedicated binary head re-splits ONLY the champion's within-pair {Train,Subway} probability
mass, leaving every other class untouched.

Discipline (vs the earlier DISABLEd rail expert): honest temporal+embargo; binary head fit on FIT only
(emb⊕520 restricted to Train/Subway); blend β selected on TUNE; TEST locked once; PAIRED-bootstrap
significance vs the champion. KEEP only if the paired Δ CI excludes 0 AND the lock-gap budget holds.

Method: champion calibrated proba P (n,8). Binary LightGBM → p = P(Train | {Train,Subway}) for every
window. Redistribute: pair_mass = P_Train+P_Subway is re-split by β·p + (1−β)·champion_split. Reuses
voting_head.{fused_probas,weight_search}, probe_fusion.{load_emb,load_feats,load_labels,calibrate,
macro_f1,CLASSES}, decision_rule.paired_bootstrap_diff, metrics.class_report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_emb, load_feats, load_labels, calibrate, macro_f1, CLASSES  # noqa: E402
from voting_head import fused_probas, weight_search  # noqa: E402
from decision_rule import paired_bootstrap_diff  # noqa: E402
from metrics import class_report  # noqa: E402

EPS = 1e-12
TRAIN, SUBWAY = 7, 8


def redistribute(P, p_pos, cls, pos, neg, beta):
    """Re-split the {pos,neg} pair mass of P (n,8) by beta*p_pos + (1-beta)*champion_split. Pair mass
    is preserved and all other classes are unchanged. p_pos = candidate P(pos | pair) per row."""
    ip, ineg = list(cls).index(pos), list(cls).index(neg)
    mass = P[:, ip] + P[:, ineg]
    champ_pos = P[:, ip] / (mass + EPS)
    new_pos = beta * p_pos + (1.0 - beta) * champ_pos
    Q = P.copy()
    Q[:, ip] = mass * new_pos
    Q[:, ineg] = mass * (1.0 - new_pos)
    return Q


def _norm(P):
    return P / (P.sum(1, keepdims=True) + EPS)


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--embs", default="utica_V2,mantisv2_V1", help="champion vote FMs")
    ap.add_argument("--bin-emb", default="utica_V2", help="FM whose emb⊕520 trains the binary head")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol" / "RAIL_DISAMBIG_RESULTS.md")
    args = ap.parse_args()
    cls = np.asarray(CLASSES)

    assign = np.load(args.split)
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, _ = load_labels(args.feat_dir, "validation")
    Ftr, Fva = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
    assert len(assign) == len(yva), "split / validation length mismatch"
    fm, tm, sm = assign == 0, assign == 1, assign == 2
    ytune, ytest = yva[tm], yva[sm]

    # --- champion calibrated proba (mult_reweight = the locked 0.8339 vote) -----------------------
    embs = [e.strip() for e in args.embs.split(",") if e.strip()]
    Ptu_list, Pte_list = [], []
    for e in embs:
        o = fused_probas(e, args.emb_root, args.feat_dir, ytr, yva, Ftr, Fva, fm, tm, sm)
        Ptu_list.append(o["proba_tune"]); Pte_list.append(o["proba_test"])
        print(f"[rail] champion fused {e}", flush=True)
    Ptu, Pte = np.stack(Ptu_list, 0), np.stack(Pte_list, 0)
    w = weight_search(Ptu, ytune, len(embs))
    cw = calibrate(np.einsum("knc,k->nc", Ptu, w), cls, ytune)
    P_tu = _norm(np.einsum("knc,k->nc", Ptu, w) * cw)
    P_te = _norm(np.einsum("knc,k->nc", Pte, w) * cw)

    # --- binary Train-vs-Subway head on emb⊕520 (FIT only) ----------------------------------------
    E_tr = load_emb(args.emb_root / args.bin_emb, "train")
    E_va = load_emb(args.emb_root / args.bin_emb, "validation")
    Xtr = np.concatenate([E_tr, Ftr], 1)
    Xva = np.concatenate([E_va, Fva], 1)
    Xfit = np.concatenate([Xtr, Xva[fm]]); yfit = np.concatenate([ytr, yva[fm]])
    Xtune, Xtest = Xva[tm], Xva[sm]
    mfit = np.isin(yfit, [TRAIN, SUBWAY]); mtune = np.isin(ytune, [TRAIN, SUBWAY])
    clf = lgb.LGBMClassifier(objective="binary", n_estimators=2000, learning_rate=0.05, num_leaves=63,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             class_weight="balanced", n_jobs=-1, verbosity=-1, random_state=0,
                             deterministic=True, force_col_wise=True)
    clf.fit(Xfit[mfit], (yfit[mfit] == TRAIN).astype(int),
            eval_set=[(Xtune[mtune], (ytune[mtune] == TRAIN).astype(int))],
            eval_metric="binary_logloss", callbacks=[early_stopping(100)])
    p_tu = clf.predict_proba(Xtune)[:, 1]                  # P(Train | {Train,Subway}) for every window
    p_te = clf.predict_proba(Xtest)[:, 1]
    # binary held-out check on the pair
    mtest = np.isin(ytest, [TRAIN, SUBWAY])
    bin_acc = float(((p_te[mtest] > 0.5).astype(int) == (ytest[mtest] == TRAIN)).mean())
    print(f"[rail] binary head pair-accuracy on TEST = {bin_acc:.4f} (n={int(mtest.sum())})", flush=True)

    # --- select β on TUNE, lock TEST once ---------------------------------------------------------
    champ_pred = cls[P_te.argmax(1)]
    champ_tu = macro_f1(ytune, cls[P_tu.argmax(1)])
    best_beta, best_m = 0.0, -1.0
    for beta in np.linspace(0.0, 1.0, 11):
        m = macro_f1(ytune, cls[redistribute(P_tu, p_tu, cls, TRAIN, SUBWAY, beta).argmax(1)])
        if m > best_m:
            best_m, best_beta = m, float(beta)
    rail_pred = cls[redistribute(P_te, p_te, cls, TRAIN, SUBWAY, best_beta).argmax(1)]

    r_c, r_r = class_report(ytest, champ_pred), class_report(ytest, rail_pred)
    d, dlo, dhi, p_le0 = paired_bootstrap_diff(ytest, rail_pred, champ_pred)
    champ_te, rail_te = r_c["macro_f1"], r_r["macro_f1"]
    keep = dlo > 0.0
    print(f"[rail] β*={best_beta:.1f} (TUNE {champ_tu:.4f}->{best_m:.4f}); TEST champ={champ_te:.4f} "
          f"rail={rail_te:.4f}; paired Δ={d:+.4f} CI[{dlo:+.4f},{dhi:+.4f}] p(Δ≤0)={p_le0:.3f} -> "
          f"{'KEEP (significant)' if keep else 'NOT significant'}", flush=True)

    def pc(r): return " ".join(f"{k[:2]}={v['f1']:.2f}" for k, v in r["per_class"].items())
    lines = [f"# E-RAIL Train↔Subway disambiguator over vote({'+'.join(embs)}) — β on TUNE, TEST locked.",
             f"binary head ({args.bin_emb}⊕520) pair-acc TEST={bin_acc:.4f}; β*={best_beta:.1f}.",
             f"champion TEST={champ_te:.4f}; rail TEST={rail_te:.4f}; **paired Δ={d:+.4f} "
             f"CI[{dlo:+.4f},{dhi:+.4f}] p(Δ≤0)={p_le0:.3f}** → **{'KEEP' if keep else 'DISABLE'}**.\n",
             "| model | TEST macro | per-class F1 (TEST) |", "|---|---|---|",
             f"| champion | {champ_te:.4f} | {pc(r_c)} |",
             f"| +rail (β={best_beta:.1f}) | {rail_te:.4f} | {pc(r_r)} |"]
    args.out.write_text("\n".join(lines) + "\n")
    for ln in lines:
        print(ln, flush=True)
    print(f"[rail] wrote {args.out}", flush=True)

    from shl2026 import track
    with track("mdaniol", run_name=f"rail_{'+'.join(embs)}", seed=0, params_path=None,
               params={"embs": ",".join(embs), "bin_emb": args.bin_emb, "split": args.split.stem,
                       "beta": best_beta, "protocol": "binary Train/Subway redistribute; β on TUNE, lock TEST"},
               tags={"phase": "hard-pair", "experiment": "E-RAIL",
                     "decision": "KEEP" if keep else "DISABLE"}) as run:
        run.log_metrics({"macro_f1": rail_te, "champion_test": champ_te, "paired_diff": d,
                         "paired_ci_lo": dlo, "paired_ci_hi": dhi, "bin_pair_acc": bin_acc,
                         "beta": best_beta})
        run.log_artifact(args.out)
    print("[rail] MLflow tracked.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
