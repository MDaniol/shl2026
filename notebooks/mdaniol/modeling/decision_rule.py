#!/usr/bin/env python3
"""C1 — decision-layer experiment on the champion vote (registered, gated vs 0.8342 on locked TEST).

Question: does a macro-F1-aware **additive-logit** decision rule beat the current **multiplicative**
per-class reweight (`calibrate()`)? Two independent research lanes (har-dl decision-lane, har-dl
feature-lane) converged here, with a concrete prediction:

  The current rule is `argmax(p_k · w_k)` (multiplicative). For an UNDER-emitted class whose p_k→0
  (Run is predicted 1.2% vs 1.91% true), a finite multiplier barely moves it. The Bayes-optimal
  post-hoc form is **additive in log-space**: `argmax(log p_k + b_k)` (GLA, arXiv:2310.08106), which
  *can* raise a starved class. We select the 8 biases by JOINT coordinate-descent on TUNE (Lipton
  arXiv:1402.1892 — joint, never independent OvR, because argmax couples the classes).

Also evaluates SLD/Saerens transductive prior-correction (arXiv 10.1162/089976602753284446) as a
SECONDARY, **shift-test-gated** rule — flagged because it reads the global test-pool prior (verify the
challenge permits unsupervised whole-test statistics before shipping it; it uses NO per-window /
cross-window / temporal information, only the global class prior).

Protocol = identical to voting_head/submit_vote: refit each FM (emb⊕520→LightGBM) on FIT, per-class
calibrate on TUNE, weighted vote (weights on TUNE); then select every decision rule's params on TUNE
and **lock TEST once**. Deterministic, pure CPU. Reuses voting_head.{fused_probas,weight_search},
probe_fusion.{load_feats,load_labels,calibrate,macro_f1,CLASSES}, metrics.class_report, split.
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
from metrics import class_report  # noqa: E402

EPS = 1e-12


def _norm(P):
    return P / (P.sum(1, keepdims=True) + EPS)


def additive_bias_search(P, y, cls, grid=None, rounds=4):
    """Joint coordinate-descent over an 8-vector additive log-bias maximizing macro-F1 on (P, y).
    Decision = argmax(log P + b). JOINT: re-optimize each bias given the others (classes couple
    through the argmax), NOT independent one-vs-rest thresholds. Deterministic."""
    if grid is None:
        grid = np.linspace(-2.0, 2.0, 41)
    logP = np.log(_norm(P) + EPS)
    K = logP.shape[1]
    b = np.zeros(K)
    best = macro_f1(y, cls[(logP + b).argmax(1)])
    for _ in range(rounds):
        improved = False
        for k in range(K):
            cur = b[k]
            best_k = cur
            for v in grid:
                b[k] = v
                m = macro_f1(y, cls[(logP + b).argmax(1)])
                if m > best:
                    best, best_k = m, v
            b[k] = best_k
            improved |= (best_k != cur)
        if not improved:
            break
    return b


def sld_correct(P, src_prior, rounds=100, tol=1e-7):
    """Saerens-Latinne-Decaestecker EM: estimate the target (test) class prior from predicted probas
    P given the source prior, return (corrected probas, estimated target prior). Transductive global
    prior only. Returns the *gated* decision to caller."""
    src = np.asarray(src_prior, dtype=np.float64) + EPS
    tgt = src.copy()
    for _ in range(rounds):
        Q = _norm(P * (tgt / src))
        new = Q.mean(0)
        if np.abs(new - tgt).max() < tol:
            break
        tgt = new
    return _norm(P * (tgt / src)), tgt


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--embs", default="utica_V2,mantisv2_V1")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--bar", type=float, default=0.8342, help="champion TEST macro to beat")
    ap.add_argument("--sld-shift-thresh", type=float, default=0.02,
                    help="apply SLD only if max|tgt_prior - src_prior| exceeds this (shift-gate)")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol" / "DECISION_RULE_RESULTS.md")
    args = ap.parse_args()
    cls = np.asarray(CLASSES)

    # --- reproduce the champion vote's calibrated probas (mirror voting_head exactly) -----------
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
        print(f"[decision] fused {e}", flush=True)
    Ptu, Pte = np.stack(Ptu_list, 0), np.stack(Pte_list, 0)
    w = weight_search(Ptu, ytune, len(embs))
    wv_tu = _norm(np.einsum("knc,k->nc", Ptu, w))
    wv_te = _norm(np.einsum("knc,k->nc", Pte, w))
    print(f"[decision] vote weights={dict(zip(embs, np.round(w, 3)))}", flush=True)

    # --- decision rules: select params on TUNE, evaluate on TEST once --------------------------
    rules = {}

    def record(name, pred_tu, pred_te):
        r_tu, r_te = class_report(ytune, pred_tu), class_report(ytest, pred_te)
        rules[name] = (r_tu, r_te)
        pc = " ".join(f"{k[:2]}={d['f1']:.2f}" for k, d in r_te["per_class"].items())
        print(f"[decision] {name:18s} TUNE={r_tu['macro_f1']:.4f} TEST={r_te['macro_f1']:.4f} | {pc}",
              flush=True)

    record("baseline", cls[wv_tu.argmax(1)], cls[wv_te.argmax(1)])
    cw = calibrate(wv_tu, cls, ytune)                                   # current champion rule
    record("mult_reweight", cls[(wv_tu * cw).argmax(1)], cls[(wv_te * cw).argmax(1)])
    b = additive_bias_search(wv_tu, ytune, cls)                         # NEW: additive log-bias
    lt, le = np.log(wv_tu + EPS), np.log(wv_te + EPS)
    record("additive_logit", cls[(lt + b).argmax(1)], cls[(le + b).argmax(1)])
    # additive on top of the multiplicative recal (does the additive part add beyond mult?)
    b2 = additive_bias_search(wv_tu * cw, ytune, cls)
    record("mult+additive", cls[(np.log(wv_tu * cw + EPS) + b2).argmax(1)],
           cls[(np.log(wv_te * cw + EPS) + b2).argmax(1)])
    # SLD (secondary, shift-test-gated): src prior = TUNE predicted-proba mean
    src_prior = wv_tu.mean(0)
    _, tgt = sld_correct(wv_te, src_prior)
    shift = float(np.abs(tgt - src_prior).max())
    if shift > args.sld_shift_thresh:
        corr_te, _ = sld_correct(wv_te * cw, (wv_tu * cw).mean(0))
        record(f"sld_gated(shift={shift:.3f})", cls[(wv_tu * cw).argmax(1)], cls[corr_te.argmax(1)])
    else:
        print(f"[decision] SLD not applied (shift {shift:.3f} ≤ {args.sld_shift_thresh}); "
              f"flagged: verify rules permit whole-test prior adaptation before shipping.", flush=True)

    # --- gate + write ---------------------------------------------------------------------------
    champ = rules["mult_reweight"][1]["macro_f1"]
    best = max(rules, key=lambda n: rules[n][0]["macro_f1"])            # SELECT on TUNE
    best_test = rules[best][1]["macro_f1"]
    keep = best_test > args.bar + 0.001 and best != "mult_reweight"
    print(f"\n[decision] champion(mult_reweight) TEST={champ:.4f}; bar={args.bar:.4f}; "
          f"best-on-TUNE={best} TEST={best_test:.4f} -> {'KEEP' if keep else 'no improvement'}",
          flush=True)

    lines = [f"# C1 decision-layer over vote({'+'.join(embs)}) — select on TUNE, lock TEST once "
             f"(bar={args.bar}).",
             f"champion `mult_reweight` TEST={champ:.4f}; best-on-TUNE=`{best}` TEST={best_test:.4f} "
             f"**{'KEEP' if keep else 'no improvement'}**.\n",
             "| rule | TUNE macro | TEST macro | per-class F1 (TEST) |", "|---|---|---|---|"]
    for name, (r_tu, r_te) in rules.items():
        pc = " ".join(f"{k[:2]}={d['f1']:.2f}" for k, d in r_te["per_class"].items())
        lines.append(f"| {name} | {r_tu['macro_f1']:.4f} | {r_te['macro_f1']:.4f} | {pc} |")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[decision] wrote {args.out}", flush=True)

    from shl2026 import track
    with track("mdaniol", run_name=f"decision_{'+'.join(embs)}", seed=0, params_path=None,
               params={"embs": ",".join(embs), "split": args.split.stem, "bar": args.bar,
                       "protocol": "additive-logit joint-CD vs multiplicative; select TUNE, lock TEST"},
               tags={"phase": "decision", "experiment": "C1-decision-rule",
                     "decision": "KEEP" if keep else "DISABLE"}) as run:
        run.log_metrics({"macro_f1": best_test, "champion_test": champ}
                        | {f"{n}_test": rules[n][1]["macro_f1"] for n in rules}
                        | {f"{n}_tune": rules[n][0]["macro_f1"] for n in rules})
        run.log_artifact(args.out)
    print("[decision] MLflow tracked.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
