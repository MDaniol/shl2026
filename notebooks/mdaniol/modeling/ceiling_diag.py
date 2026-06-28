#!/usr/bin/env python3
"""Phase-0 diagnostics — bound the achievable macro-F1 and characterize the failure modes BEFORE
spending GH200 time (completeness-critic, round-3 synthesis). Pure measurement on the LOCKED TEST:
nothing here selects a model or tunes a rule — it only characterizes the ceiling, so it is reported,
never fed back into selection.

Computes:
  1. Champion macro-F1 + **bootstrap 95% CI** (per-window resample) — so "low-single-digit gains" can be
     judged against noise (a candidate inside the CI is not a real win).
  2. **Top-2 oracle ceiling** — macro-F1 if we could always pick the true label whenever *either* champion
     FM ranks it top-2. Upper bound for any better fusion / decision rule on the SAME representations.
  3. **Confusion-collapse ceiling** — macro-F1 if Train↔Subway and Car↔Bus were perfectly separated
     (mispredictions WITHIN each pair corrected). Quantifies exactly how much the two caps cost.
  4. **FM error-correlation** (utica vs mantis): Q-statistic, disagreement, double-fault, per-class error
     overlap — does adding voters even have room to help, or are the champions already redundant? (This
     is the cheap check that explains the E-FMDIV dilution and gates every new-voter job.)

Reuses voting_head.{fused_probas,weight_search}, probe_fusion.{load_feats,load_labels,calibrate,
macro_f1,CLASSES}, metrics.class_report, split. Deterministic, pure CPU on cached probas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_feats, load_labels, calibrate, macro_f1, CLASSES  # noqa: E402
from voting_head import fused_probas, weight_search  # noqa: E402

EPS = 1e-12
NAME = {1: "Still", 2: "Walk", 3: "Run", 4: "Bike", 5: "Car", 6: "Bus", 7: "Train", 8: "Subway"}
PAIRS = [("Train", "Subway", 7, 8), ("Car", "Bus", 5, 6)]


def bootstrap_ci(y, pred, cls, B=1000, seed=0, alpha=0.05):
    """Per-window bootstrap CI of macro-F1 for a fixed (y, pred). Deterministic (seeded)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = macro_f1(y[idx], pred[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(macro_f1(y, pred)), float(lo), float(hi)


def top2_oracle(y, P_list, base_pred, cls):
    """Pred = true label if it is in the top-2 of ANY FM's probas, else the base (vote) prediction.
    Upper bound on what a better selector/fusion of these representations could reach."""
    in_top2 = np.zeros(len(y), bool)
    for P in P_list:
        order = np.argsort(-P, axis=1)[:, :2]                 # top-2 class indices
        top2_labels = cls[order]                              # (n,2) labels
        in_top2 |= (top2_labels == y[:, None]).any(1)
    return np.where(in_top2, y, base_pred)


def confusion_collapse(y, pred, pairs):
    """Pred with each confusable pair perfectly separated: where {pred,true} are the two members of a
    pair, correct pred to true. macro-F1 of this = ceiling if those pairs were solved."""
    out = pred.copy()
    for _, _, a, b in pairs:
        m = ((y == a) & (pred == b)) | ((y == b) & (pred == a))
        out[m] = y[m]
    return out


def error_correlation(y, pa, pb):
    """Diversity of two classifiers' errors: Yule's Q, disagreement, double-fault."""
    ca, cb = (pa == y), (pb == y)
    n11 = int((ca & cb).sum()); n00 = int((~ca & ~cb).sum())
    n10 = int((ca & ~cb).sum()); n01 = int((~ca & cb).sum())
    q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10 + EPS)
    return {"Q": float(q), "disagreement": float((n10 + n01) / len(y)),
            "double_fault": float(n00 / len(y)), "acc_a": float(ca.mean()), "acc_b": float(cb.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--embs", default="utica_V2,mantisv2_V1")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol" / "CEILING_DIAGNOSTICS.md")
    args = ap.parse_args()
    cls = np.asarray(CLASSES)

    assign = np.load(args.split)
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, _ = load_labels(args.feat_dir, "validation")
    Ftr, Fva = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
    assert len(assign) == len(yva), "split / validation length mismatch"
    fm, tm, sm = assign == 0, assign == 1, assign == 2
    ytune, ytest = yva[tm], yva[sm]
    embs = [e.strip() for e in args.embs.split(",") if e.strip()]
    Ptu_list, Pte_list = [], []
    for e in embs:
        o = fused_probas(e, args.emb_root, args.feat_dir, ytr, yva, Ftr, Fva, fm, tm, sm)
        Ptu_list.append(o["proba_tune"]); Pte_list.append(o["proba_test"])
        print(f"[ceiling] fused {e}", flush=True)
    Ptu, Pte = np.stack(Ptu_list, 0), np.stack(Pte_list, 0)
    w = weight_search(Ptu, ytune, len(embs))
    wv_tu = np.einsum("knc,k->nc", Ptu, w); wv_te = np.einsum("knc,k->nc", Pte, w)
    cw = calibrate(wv_tu, cls, ytune)
    champ_pred = cls[(wv_te * cw).argmax(1)]

    # 1. champion + bootstrap CI
    champ, lo, hi = bootstrap_ci(ytest, champ_pred, cls, B=args.boot)
    # 2. top-2 oracle ceiling (per-FM probas)
    oracle_pred = top2_oracle(ytest, list(Pte), champ_pred, cls)
    oracle = macro_f1(ytest, oracle_pred)
    # 3. confusion-collapse ceiling
    collapse = macro_f1(ytest, confusion_collapse(ytest, champ_pred, PAIRS))
    # 4. FM error-correlation (single-FM argmax)
    fm_preds = [cls[P.argmax(1)] for P in Pte]
    corr = error_correlation(ytest, fm_preds[0], fm_preds[1]) if len(embs) >= 2 else {}

    lines = [f"# Phase-0 ceiling diagnostics — vote({'+'.join(embs)}), locked TEST (analysis only).\n",
             f"- **Champion macro-F1 = {champ:.4f}**  (95% bootstrap CI [{lo:.4f}, {hi:.4f}], "
             f"B={args.boot}) → a candidate inside this CI is NOT a real win.",
             f"- **Top-2 oracle ceiling = {oracle:.4f}**  (Δ {oracle - champ:+.4f}) — max reachable by a "
             f"better selector/decision on these two FMs. Small Δ ⇒ representation-limited, GPU voters won't help.",
             f"- **Confusion-collapse ceiling = {collapse:.4f}**  (Δ {collapse - champ:+.4f}) — macro-F1 if "
             f"Train↔Subway & Car↔Bus were perfectly separated. This Δ is the entire prize of the hard-pair lane.",
             ""]
    if corr:
        lines.append(f"- **FM error-correlation (utica vs mantis):** Q={corr['Q']:.3f} "
                     f"disagreement={corr['disagreement']:.3f} double_fault={corr['double_fault']:.3f} "
                     f"(acc {corr['acc_a']:.3f}/{corr['acc_b']:.3f}). High Q + high double-fault ⇒ redundant "
                     f"champions ⇒ new voters likely DILUTE (the E-FMDIV mechanism).")
    args.out.write_text("\n".join(lines) + "\n")
    for ln in lines:
        print(ln, flush=True)
    print(f"[ceiling] wrote {args.out}", flush=True)

    from shl2026 import track
    with track("mdaniol", run_name=f"ceiling_{'+'.join(embs)}", seed=0, params_path=None,
               params={"embs": ",".join(embs), "split": args.split.stem, "boot": args.boot,
                       "protocol": "analysis-only ceiling/diagnostics on locked TEST"},
               tags={"phase": "diagnostic", "experiment": "P0-ceiling"}) as run:
        run.log_metrics({"champion": champ, "ci_lo": lo, "ci_hi": hi, "top2_oracle": oracle,
                         "collapse_ceiling": collapse} | ({f"corr_{k}": v for k, v in corr.items()} if corr else {}))
        run.log_artifact(args.out)
    print("[ceiling] MLflow tracked.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
