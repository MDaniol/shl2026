#!/usr/bin/env python3
"""E-COMBINE — leakage-clean soft-vote of OUR champion (v4) ⊕ the collaborator's frozen-DINoV2 pipeline.

Both are honest ~0.834 and structurally decorrelated (our frozen TS-FM vote over raw IMU vs his frozen
DINoV2 over STFT/CWT/GAF spectrogram images). The blend weight w is tuned ONLY on the slice held out by
BOTH models — (his `target_holdout`) ∩ (our `validation[TEST]`, which our champion never fit/calibrated
on) — then applied to the hidden test. v3/v4 fit calibrate() AND the additive bias on TUNE, so only our
TEST is clean for us → we intersect with TEST (not TUNE).

Alignment: our validation obs i -> (loc = LOCATIONS[i//NV], row = i%NV); his position_code {Bag0,Hand1,
Hips2,Torso3}. Hidden test: both in canonical official order -> 1:1 (sanity-checked by argmax-agreement).

Gate: KEEP the v5 blend iff its macro-F1 on the doubly-held-out slice beats max(ours, his) with a PAIRED
Δ CI excluding 0. Reuses voting_head.{fused_probas,weight_search}, probe_fusion.{load_feats,load_labels,
calibrate,macro_f1,CLASSES,LOCATIONS}, decision_rule.{additive_bias_search,paired_bootstrap_diff},
metrics.class_report, shl2026.write_submission.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import (load_feats, load_labels, calibrate, macro_f1, CLASSES, LOCATIONS)  # noqa: E402
from voting_head import fused_probas, weight_search  # noqa: E402
from decision_rule import additive_bias_search, paired_bootstrap_diff  # noqa: E402
from metrics import class_report  # noqa: E402

EPS = 1e-12
N_TEST = 92726
LOC_TO_HISPOS = {"Bag": 0, "Hand": 1, "Hips": 2, "Torso": 3}   # his §8 position_code


def _norm(P):
    return P / (P.sum(1, keepdims=True) + EPS)


def val_keys(n_val, sel_mask):
    """(row_index, his_position_code) for each selected validation observation, in our concat order
    (LOCATIONS-major). n_val = total validation obs; sel_mask picks the subset; NV = obs per location."""
    nv = n_val // len(LOCATIONS)
    idx = np.flatnonzero(sel_mask)
    return [(int(i % nv), LOC_TO_HISPOS[LOCATIONS[i // nv]]) for i in idx]


def tune_w(our_P, his_P, y, cls, grid=None):
    """Pick w∈[0,1] maximizing macro-F1 of argmax(w·our + (1-w)·his) on the shared held-out slice."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 41)
    best_w, best_m = 1.0, -1.0
    for w in grid:
        m = macro_f1(y, cls[(w * our_P + (1 - w) * his_P).argmax(1)])
        if m > best_m:
            best_m, best_w = m, float(w)
    return best_w, best_m


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--embs", default="utica_V2,mantisv2_V1")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--pdusza-dir", type=Path, required=True, help="his results root on group storage")
    ap.add_argument("--our-test-proba", type=Path,
                    default=root / "notebooks/mdaniol" / "preds_test_v4_vote.npy")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol" / "AGH_predictions_v5_combine.txt")
    args = ap.parse_args()
    cls = np.asarray(CLASSES)
    PD = args.pdusza_dir

    # --- our v4 champion proba on validation[TEST] (clean for us) -------------------------------
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
        print(f"[combine] our fused {e}", flush=True)
    Ptu, Pte = np.stack(Ptu_list, 0), np.stack(Pte_list, 0)
    w_v = weight_search(Ptu, ytune, len(embs))
    cw = calibrate(np.einsum("knc,k->nc", Ptu, w_v), cls, ytune)
    b = additive_bias_search(np.einsum("knc,k->nc", Ptu, w_v) * cw, ytune, cls)     # v4 additive
    our_test = _norm(np.einsum("knc,k->nc", Pte, w_v) * cw * np.exp(b))             # our v4 proba on val TEST
    test_keys = val_keys(len(yva), sm)                                              # (row,pos) per TEST obs
    our_map = {k: i for i, k in enumerate(test_keys)}

    # --- his holdout (adapted = the model that made the final preds) ----------------------------
    d = np.load(PD / "02_adaptation_selection/selected_untouched_target_holdout/predictions.npz")
    his_ri, his_pos, his_y, his_p = d["row_index"], d["position_code"], d["y_true"], d["probabilities"]

    # --- intersect: his holdout ∩ our TEST (doubly held-out) ------------------------------------
    our_rows, his_rows = [], []
    for j, key in enumerate(zip(his_ri.tolist(), his_pos.tolist())):
        if key in our_map:
            our_rows.append(our_map[key]); his_rows.append(j)
    our_rows, his_rows = np.array(our_rows), np.array(his_rows)
    ourP = our_test[our_rows]; hisP = _norm(his_p[his_rows]); y = ytest[our_rows]
    assert (his_y[his_rows] + 1 == y).all(), "label mismatch on shared slice — alignment is wrong!"
    print(f"[combine] doubly-held-out slice: n={len(y)} "
          f"(his holdout {len(his_y)} ∩ our TEST {len(ytest)})", flush=True)

    # decorrelation + per-model macro on the slice
    pa, pb = cls[ourP.argmax(1)], cls[hisP.argmax(1)]
    ca, cb = pa == y, pb == y
    q = ((ca & cb).sum() * (~ca & ~cb).sum() - (ca & ~cb).sum() * (~ca & cb).sum()) / \
        (((ca & cb).sum() * (~ca & ~cb).sum() + (ca & ~cb).sum() * (~ca & cb).sum()) + EPS)
    our_m, his_m = macro_f1(y, pa), macro_f1(y, pb)
    best_w, blend_m = tune_w(ourP, hisP, y, cls)
    blend_pred = cls[(best_w * ourP + (1 - best_w) * hisP).argmax(1)]
    base_pred = pa if our_m >= his_m else pb
    dlt, dlo, dhi, p_le0 = paired_bootstrap_diff(y, blend_pred, base_pred)
    keep = dlo > 0.0
    r_blend = class_report(y, blend_pred)
    print(f"[combine] slice macro: ours={our_m:.4f} his={his_m:.4f} blend(w={best_w:.2f})={blend_m:.4f} "
          f"| Q={q:.3f} | paired Δ(blend−best)={dlt:+.4f} CI[{dlo:+.4f},{dhi:+.4f}] -> "
          f"{'KEEP' if keep else 'no gain'}", flush=True)

    # --- blend the HIDDEN TEST (both official order) -> v5 -------------------------------------
    our_te = np.load(args.our_test_proba); his_te = _norm(np.load(
        PD / "03_final_all_labelled_ensemble/primary_adapted_ensemble_probabilities.npy"))
    assert our_te.shape == his_te.shape == (N_TEST, 8), f"test shape {our_te.shape} vs {his_te.shape}"
    agree = (our_te.argmax(1) == his_te.argmax(1)).mean()
    assert agree > 0.5, f"test argmax agreement {agree:.3f} too low — rows likely MISALIGNED, abort"
    print(f"[combine] hidden-test argmax agreement ours-vs-his = {agree:.3f} (alignment sane)", flush=True)
    blend_te = best_w * _norm(our_te) + (1 - best_w) * his_te
    pred = cls[blend_te.argmax(1)].astype(int)
    from shl2026 import write_submission
    rep = write_submission(pred, args.out)
    uniq, cnt = np.unique(pred, return_counts=True)
    print(f"[combine] v5 wrote {args.out} ok={rep.ok} rows={rep.n_rows} "
          f"dist={ {int(c): round(n/len(pred),3) for c,n in zip(uniq,cnt)} }", flush=True)

    lines = [f"# E-COMBINE — our v4 ⊕ collaborator DINoV2 (w on his_holdout ∩ our TEST, locked).",
             f"slice n={len(y)}; ours={our_m:.4f} his={his_m:.4f} **blend(w={best_w:.2f})={blend_m:.4f}** "
             f"Q={q:.3f}; paired Δ={dlt:+.4f} CI[{dlo:+.4f},{dhi:+.4f}] **{'KEEP v5' if keep else 'no gain'}**.",
             f"hidden-test argmax-agreement={agree:.3f}; v5 dist={ {int(c): round(n/len(pred),3) for c,n in zip(uniq,cnt)} }",
             "", "| model | slice macro | per-class F1 |", "|---|---|---|",
             f"| ours(v4) | {our_m:.4f} | " + " ".join(f"{k[:2]}={d2['f1']:.2f}" for k, d2 in class_report(y, pa)['per_class'].items()) + " |",
             f"| his | {his_m:.4f} | " + " ".join(f"{k[:2]}={d2['f1']:.2f}" for k, d2 in class_report(y, pb)['per_class'].items()) + " |",
             f"| blend | {blend_m:.4f} | " + " ".join(f"{k[:2]}={d2['f1']:.2f}" for k, d2 in r_blend['per_class'].items()) + " |"]
    md = root / "notebooks/mdaniol" / "COMBINE_PDUSZA_RESULTS.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"[combine] wrote {md}", flush=True)

    from shl2026 import track
    with track("mdaniol", run_name="combine_pdusza_v5", seed=0, params_path=None,
               params={"embs": ",".join(embs), "w": best_w, "split": args.split.stem,
                       "protocol": "soft-vote ours⊕his; w on his_holdout∩our_TEST; lock test"},
               tags={"phase": "combine", "experiment": "E-COMBINE",
                     "decision": "KEEP" if keep else "DISABLE"}) as run:
        run.log_metrics({"macro_f1": blend_m, "ours_slice": our_m, "his_slice": his_m, "w": best_w,
                         "Q": float(q), "paired_diff": dlt, "paired_ci_lo": dlo, "test_agreement": float(agree)})
        run.log_artifact(md); run.log_artifact(args.out)
    print("[combine] MLflow tracked.", flush=True)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
