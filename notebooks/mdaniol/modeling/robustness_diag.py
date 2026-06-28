#!/usr/bin/env python3
"""Confusion-structure + GENERALIZABILITY diagnostics for the champion vote (analysis only on locked
TEST; selects nothing). Answers three questions that bound the hidden-test risk:

  1. CONFUSION STRUCTURE — where does the +0.097 top-2 oracle gap actually live? The 8x8 confusion
     matrix + for each error whether the true label is the model's RANK-2 class (recoverable by a
     top-2 disambiguator), ranked by confusion pair. Tells us if a general top-2 reranker is worth it
     or if it's only Train↔Subway.
  2. PER-LOCATION ROBUSTNESS — per-class F1 on Bag/Hips/Torso separately. The hidden test has NO
     location label, so a rule that wins on average but collapses on one placement is a hidden-test
     risk. (TEST is BHT-only by split construction.)
  3. PRIOR SENSITIVITY — macro-F1 of the SAME decision under a grid of class priors (esp. Run 1–10%),
     by reweighting the confusion rows. The hidden-test prior is unknown and macro-F1 is prior-
     sensitive; this quantifies how fragile our number is to a shifted test distribution.

Reuses voting_head.{fused_probas,weight_search}, probe_fusion.{load_feats,load_labels,calibrate,
macro_f1,CLASSES}. Deterministic, pure CPU on cached probas.
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


def confusion_counts(y, pred, cls):
    """(K,K) counts, rows = true class, cols = predicted class (in cls order)."""
    idx = {c: i for i, c in enumerate(cls)}
    C = np.zeros((len(cls), len(cls)), dtype=np.int64)
    for t, p in zip(y, pred):
        C[idx[t], idx[p]] += 1
    return C


def per_class_f1_from_confusion(C):
    """Per-class F1 + macro from a (true x pred) confusion-count matrix."""
    tp = np.diag(C).astype(float)
    fp = C.sum(0) - tp
    fn = C.sum(1) - tp
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    f1 = 2 * prec * rec / (prec + rec + EPS)
    return f1, float(f1.mean())


def top2_structure(y, pred, proba, cls):
    """For each error, is the true label the RANK-2 predicted class? Aggregate by (true,pred) pair.
    Returns rows sorted by error count: (true, pred, n_err, n_recoverable_at_top2)."""
    order = np.argsort(-proba, axis=1)
    rank2 = cls[order[:, 1]]
    err = pred != y
    agg = {}
    for t, p, r2, e in zip(y, pred, rank2, err):
        if not e:
            continue
        k = (int(t), int(p))
        a = agg.setdefault(k, [0, 0])
        a[0] += 1
        a[1] += int(r2 == t)                     # truth was the model's 2nd choice -> a reranker could fix it
    rows = sorted(([t, p, n, rec] for (t, p), (n, rec) in agg.items()), key=lambda r: -r[2])
    return rows


def prior_sensitivity(C, cls, target_priors):
    """Macro-F1 of the same decision under each target class-prior (reweight confusion ROWS by
    target/empirical). target_priors: list of (label, prior_vector) with prior summing to 1."""
    emp = C.sum(1) / (C.sum() + EPS)             # empirical true-class prior on TEST
    out = []
    for label, pri in target_priors:
        pri = np.asarray(pri, float); pri = pri / pri.sum()
        w = pri / (emp + EPS)
        _, macro = per_class_f1_from_confusion(C * w[:, None])
        out.append((label, float(macro)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--embs", default="utica_V2,mantisv2_V1")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol" / "ROBUSTNESS_DIAG.md")
    args = ap.parse_args()
    cls = np.asarray(CLASSES)

    assign = np.load(args.split)
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, lva = load_labels(args.feat_dir, "validation")
    Ftr, Fva = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
    assert len(assign) == len(yva), "split / validation length mismatch"
    fm, tm, sm = assign == 0, assign == 1, assign == 2
    ytune, ytest, loc_test = yva[tm], yva[sm], lva[sm]
    embs = [e.strip() for e in args.embs.split(",") if e.strip()]
    Ptu_list, Pte_list = [], []
    for e in embs:
        o = fused_probas(e, args.emb_root, args.feat_dir, ytr, yva, Ftr, Fva, fm, tm, sm)
        Ptu_list.append(o["proba_tune"]); Pte_list.append(o["proba_test"])
        print(f"[robust] fused {e}", flush=True)
    Ptu, Pte = np.stack(Ptu_list, 0), np.stack(Pte_list, 0)
    w = weight_search(Ptu, ytune, len(embs))
    cw = calibrate(np.einsum("knc,k->nc", Ptu, w), cls, ytune)
    proba_te = np.einsum("knc,k->nc", Pte, w) * cw
    pred = cls[proba_te.argmax(1)]

    C = confusion_counts(ytest, pred, cls)
    f1, macro = per_class_f1_from_confusion(C)
    rows = top2_structure(ytest, pred, proba_te, cls)

    lines = [f"# Confusion-structure + generalizability — vote({'+'.join(embs)}), locked TEST.\n",
             f"champion macro-F1 = {macro:.4f}; per-class F1: "
             + " ".join(f"{NAME[c][:2]}={f:.2f}" for c, f in zip(cls, f1)) + "\n",
             "## Top-2 recoverable confusions (true→pred, n_err, recoverable@rank-2)"]
    tot_err = sum(r[2] for r in rows); tot_rec = sum(r[3] for r in rows)
    for t, p, n, rec in rows[:10]:
        lines.append(f"- {NAME[t]}→{NAME[p]}: {n} errors, {rec} ({100*rec/max(n,1):.0f}%) had truth as rank-2")
    lines.append(f"\n**{tot_rec}/{tot_err} ({100*tot_rec/max(tot_err,1):.0f}%) of all errors are "
                 f"top-2-recoverable** → upper bound on a top-2 reranker's reach.\n")

    # per-location robustness
    lines.append("## Per-location robustness (Bag/Hips/Torso) — hidden test has no location label")
    for loc in ["Bag", "Hips", "Torso"]:
        m = loc_test == loc
        if m.sum() == 0:
            continue
        Cl = confusion_counts(ytest[m], pred[m], cls)
        fl, ml = per_class_f1_from_confusion(Cl)
        worst = sorted(zip(cls, fl), key=lambda x: x[1])[:3]
        lines.append(f"- **{loc}** (n={int(m.sum())}): macro={ml:.4f}; weakest: "
                     + ", ".join(f"{NAME[c]}={f:.2f}" for c, f in worst))
    spread = []
    for loc in ["Bag", "Hips", "Torso"]:
        m = loc_test == loc
        if m.sum():
            spread.append(per_class_f1_from_confusion(confusion_counts(ytest[m], pred[m], cls))[1])
    if spread:
        lines.append(f"\n**Per-location macro spread = {max(spread)-min(spread):.4f}** "
                     f"(large ⇒ placement-fragile ⇒ hidden-test risk).\n")

    # prior sensitivity
    lines.append("## Prior sensitivity (same decision, reweighted test prior) — hidden-test prior unknown")
    emp = C.sum(1) / C.sum()
    uniform = np.ones(len(cls)) / len(cls)
    grids = [("empirical", emp), ("uniform", uniform)]
    for rf in (0.01, 0.03, 0.05, 0.10):                          # vary Run fraction, rest proportional
        pri = emp.copy(); run_i = list(cls).index(3)
        pri[run_i] = 0.0; pri = pri / pri.sum() * (1 - rf); pri[run_i] = rf
        grids.append((f"Run={rf:.0%}", pri))
    for label, macro_p in prior_sensitivity(C, cls, grids):
        lines.append(f"- {label}: macro-F1 = {macro_p:.4f}")
    macros = [m for _, m in prior_sensitivity(C, cls, grids)]
    lines.append(f"\n**Macro-F1 range across priors = [{min(macros):.4f}, {max(macros):.4f}]** "
                 f"(width ⇒ fragility to a shifted hidden-test prior).\n")

    args.out.write_text("\n".join(lines) + "\n")
    for ln in lines:
        print(ln, flush=True)
    print(f"[robust] wrote {args.out}", flush=True)

    from shl2026 import track
    with track("mdaniol", run_name=f"robust_{'+'.join(embs)}", seed=0, params_path=None,
               params={"embs": ",".join(embs), "split": args.split.stem,
                       "protocol": "analysis-only confusion/per-location/prior on locked TEST"},
               tags={"phase": "diagnostic", "experiment": "P0-robustness"}) as run:
        run.log_metrics({"macro_f1": macro,
                         "loc_macro_spread": (max(spread) - min(spread)) if spread else 0.0,
                         "prior_macro_min": float(min(macros)), "prior_macro_max": float(max(macros)),
                         "top2_recoverable_frac": tot_rec / max(tot_err, 1)})
        run.log_artifact(args.out)
    print("[robust] MLflow tracked.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
