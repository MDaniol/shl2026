#!/usr/bin/env python3
"""Representation ablation — does the 520 handcrafted (HC) block earn its keep, or is it an
overfitting / generalization liability (the team-lead's last-year concern)?

For one FM, fit the SAME LightGBM head (fit_cal_eval protocol) on three representations:
  - fm      : FM embedding only
  - hc      : 520 handcrafted features only
  - fusion  : emb ⊕ 520 (the champion's input)
and report, per variant: TEST macro-F1, TUNE macro-F1, the **selection-lock gap** (TUNE−TEST; the
overfitting thermometer), and per-class F1. Then, on the FUSION model, two overfitting probes:
  - **importance share** — fraction of total LightGBM importance on the HC columns vs the FM columns
    (HC dominating while fusion≈fm ⇒ HC used but not generalizing);
  - **top-k importance stability** — Jaccard of the top-k important features across two disjoint
    halves of FIT (low ⇒ unstable, dataset-specific features ⇒ overfitting risk).

Decision read-out: fusion≈fm ⇒ drop/shrink HC for robustness at no score cost; fusion≫fm ⇒ HC earns
its keep (and the gap tells how overfit-prone each is). Analysis on the honest temporal+embargo split,
TEST locked once. Reuses probe_fusion.{load_emb,load_feats,load_labels,fit_cal_eval,CLASSES}.
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
from probe_fusion import load_emb, load_feats, load_labels, fit_cal_eval, CLASSES  # noqa: E402

EPS = 1e-12


def importance_share(imp, n_emb):
    """(fm_share, hc_share) of total LightGBM feature importance; first n_emb cols are the embedding."""
    imp = np.asarray(imp, float); tot = imp.sum() + EPS
    return float(imp[:n_emb].sum() / tot), float(imp[n_emb:].sum() / tot)


def jaccard_topk(imp_a, imp_b, k):
    """Top-k important-feature-set overlap between two fits (importance-rank stability)."""
    a = set(np.argsort(-np.asarray(imp_a))[:k]); b = set(np.argsort(-np.asarray(imp_b))[:k])
    return len(a & b) / max(len(a | b), 1)


def _imp_fit(X, y, n_estimators=400):
    """A light deterministic LightGBM purely for an importance vector (no calibration)."""
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=n_estimators,
                             learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, class_weight="balanced", n_jobs=-1, verbosity=-1,
                             random_state=0, deterministic=True, force_col_wise=True)
    clf.fit(X, y)
    return clf.feature_importances_


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--emb", default="utica_V2")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol" / "ABLATION_REP.md")
    args = ap.parse_args()

    assign = np.load(args.split)
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, _ = load_labels(args.feat_dir, "validation")
    E_tr, E_va = load_emb(args.emb_root / args.emb, "train"), load_emb(args.emb_root / args.emb, "validation")
    F_tr, F_va = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
    assert len(assign) == len(yva) == len(E_va) == len(F_va), "split / validation length mismatch"
    fm_, tm_, sm_ = assign == 0, assign == 1, assign == 2
    n_emb = E_tr.shape[1]

    def variant(name, Xtr_blk, Xva_blk):
        Xfit = np.concatenate([Xtr_blk, Xva_blk[fm_]]); yfit = np.concatenate([ytr, yva[fm_]])
        o = fit_cal_eval(name, Xfit, yfit, Xva_blk[tm_], yva[tm_], Xva_blk[sm_], yva[sm_],
                         return_probs=True)
        return o

    reps = {
        "fm": variant("fm", E_tr, E_va),
        "hc": variant("hc", F_tr, F_va),
        "fusion": variant("fusion", np.concatenate([E_tr, F_tr], 1), np.concatenate([E_va, F_va], 1)),
    }
    te = {k: o["rep"]["macro_f1"] for k, o in reps.items()}
    tu = {k: o["rep_tune"]["macro_f1"] for k, o in reps.items()}

    # overfitting probes on the FUSION model
    fm_share, hc_share = importance_share(reps["fusion"]["model"].feature_importances_, n_emb)
    Xfit_fus = np.concatenate([np.concatenate([E_tr, F_tr], 1),
                               np.concatenate([E_va, F_va], 1)[fm_]])
    yfit_fus = np.concatenate([ytr, yva[fm_]])
    half = len(Xfit_fus) // 2
    imp_a = _imp_fit(Xfit_fus[:half], yfit_fus[:half])
    imp_b = _imp_fit(Xfit_fus[half:], yfit_fus[half:])
    stab = jaccard_topk(imp_a, imp_b, args.topk)

    add_over_fm = te["fusion"] - te["fm"]
    lines = [f"# Representation ablation — emb={args.emb} (temporal+embargo, TEST locked once).\n",
             "| rep | TEST macro | TUNE macro | gap (TUNE−TEST) | per-class F1 (TEST) |",
             "|---|---|---|---|---|"]
    for k in ("fm", "hc", "fusion"):
        pc = " ".join(f"{n[:2]}={d['f1']:.2f}" for n, d in reps[k]["rep"]["per_class"].items())
        lines.append(f"| {k} | {te[k]:.4f} | {tu[k]:.4f} | {tu[k]-te[k]:+.4f} | {pc} |")
    lines += [
        f"\n- **fusion − fm (HC's marginal value on TEST) = {add_over_fm:+.4f}.**",
        f"- **Fusion importance share:** FM={fm_share:.2f}, HC={hc_share:.2f} "
        f"({n_emb} emb cols vs {F_tr.shape[1]} HC cols).",
        f"- **Top-{args.topk} importance stability (Jaccard across 2 FIT halves) = {stab:.2f}** "
        f"(low ⇒ unstable / dataset-specific ⇒ overfitting risk).",
        "\n## Read-out",
        f"- {'fusion≈fm → HC adds little; DROP/SHRINK HC for robustness at no score cost.' if add_over_fm < 0.005 else 'fusion>fm → HC earns its keep.'}",
        f"- Widest selection-lock gap: **{max(tu, key=lambda k: tu[k]-te[k])}** "
        f"(most overfit-prone variant).",
    ]
    args.out.write_text("\n".join(lines) + "\n")
    for ln in lines:
        print(ln, flush=True)
    print(f"[ablation] wrote {args.out}", flush=True)

    from shl2026 import track
    with track("mdaniol", run_name=f"ablation_{args.emb}", seed=0, params_path=None,
               params={"emb": args.emb, "split": args.split.stem,
                       "protocol": "fm/hc/fusion ablation; honest temporal+embargo"},
               tags={"phase": "diagnostic", "experiment": "ablation-rep"}) as run:
        run.log_metrics({f"test_{k}": te[k] for k in te} | {f"tune_{k}": tu[k] for k in tu}
                        | {"hc_minus_none": add_over_fm, "fm_imp_share": fm_share,
                           "hc_imp_share": hc_share, "top_feat_stability": stab})
        run.log_artifact(args.out)
    print("[ablation] MLflow tracked.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
