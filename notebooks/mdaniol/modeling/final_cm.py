#!/usr/bin/env python3
"""FINAL all-class v5 confusion matrix on a shared held-out set (leakage-clean, no colleague re-run).

Idea (see paper Appendix C): to score the *fused* model on every class we need windows that BOTH lanes
held out. We reuse the collaborator's `target_holdout` (clean for Lane B, all 8 classes present) and simply
RE-FIT Lane A on `User-1 train + (validation ∖ his-holdout)`, predicting on his-holdout — so Lane A is clean
there too. Blend with the pre-locked w → an honest all-class v5 confusion matrix. Nothing on the vision
side is re-run; only our cached-embedding LightGBM heads are re-fit (~20 min CPU).

Cleanliness:
  * Lane A never fits/tunes/calibrates on the eval windows (they are the held-out TEST of the re-fit).
  * Lane B predictions are his own target_holdout (his model never trained on them).
  * w is pre-locked (tuned earlier on the doubly-held-out slice); by default we EXCLUDE those w-tuning
    windows from the eval set so w-selection is clean too, and restrict to Bag/Hips/Torso (real test has no
    Hand). Flags --include-wtune / --include-hand relax these.

Reuses probe_fusion / voting_head / decision_rule / combine_pdusza machinery + figures/plot_cm.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "figures"))
from probe_fusion import (load_feats, load_labels, calibrate, macro_f1, CLASSES, LOCATIONS)  # noqa: E402
from voting_head import fused_probas, weight_search  # noqa: E402
from decision_rule import additive_bias_search, paired_bootstrap_diff  # noqa: E402
from combine_pdusza import HIS_POSITION_NAMES, LOC_TO_HISPOS, _norm  # noqa: E402
from plot_cm import plot_cm, per_class_f1  # noqa: E402

EPS = 1e-12


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--embs", default="utica_V2,mantisv2_V1")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--his-holdout", type=Path, required=True,
                    help="his selected_untouched_target_holdout/predictions.npz")
    ap.add_argument("--w", type=float, default=0.575, help="pre-locked blend weight P=w*LaneA+(1-w)*LaneB")
    ap.add_argument("--out-dir", type=Path, default=root / "notebooks/mdaniol" / "figures")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--include-hand", action="store_true", help="keep Hand windows (default: BHT only)")
    ap.add_argument("--include-wtune", action="store_true",
                    help="keep the w-tuning windows in the eval set (default: exclude for clean w-selection)")
    ap.add_argument("--titles", action="store_true", help="bake titles into the figures")
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()
    cls = np.asarray(CLASSES)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    st = a.titles

    # --- load our validation labels/feats + the original split (for TUNE + w-tune windows) ---
    assign = np.load(a.split)
    ytr, _ = load_labels(a.feat_dir, "train")
    yva, _ = load_labels(a.feat_dir, "validation")
    Ftr, Fva = load_feats(a.feat_dir, "train"), load_feats(a.feat_dir, "validation")
    assert len(assign) == len(yva), "split / validation length mismatch"
    nv = len(yva) // len(LOCATIONS)

    # --- map his holdout (row_index, position_code) -> our validation observation index ---
    key_to_obs = {(i % nv, LOC_TO_HISPOS[LOCATIONS[i // nv]]): i for i in range(len(yva))}
    d = np.load(a.his_holdout)
    his_ri, his_pos, his_y, his_p = d["row_index"], d["position_code"], d["y_true"], d["probabilities"]
    obs_idx, his_rows = [], []
    for j in range(len(his_ri)):
        k = (int(his_ri[j]), int(his_pos[j]))
        o = key_to_obs.get(k)
        if o is not None:
            obs_idx.append(o); his_rows.append(j)
    obs_idx, his_rows = np.array(obs_idx), np.array(his_rows)
    assert len(obs_idx) >= 20000, f"only {len(obs_idx)} his-holdout windows mapped — alignment error"
    assert (his_y[his_rows] + 1 == yva[obs_idx]).all(), "label mismatch on mapped holdout — alignment wrong!"
    print(f"[final_cm] mapped {len(obs_idx)} of {len(his_ri)} his-holdout windows to our validation obs",
          flush=True)

    # --- re-fit Lane A holding out ALL his-holdout windows; predict on them ---
    sm = np.zeros(len(yva), bool); sm[obs_idx] = True          # eval / held-out (exclude from train)
    tm = (assign == 1) & ~sm                                   # tune = original TUNE minus his holdout
    fm = ~sm & ~tm                                             # fit = the rest of validation
    print(f"[final_cm] Lane-A re-fit split: fit={fm.sum()} tune={tm.sum()} test(sm)={sm.sum()}", flush=True)
    embs = [e.strip() for e in a.embs.split(",") if e.strip()]
    Ptu_list, Pte_list = [], []
    for e in embs:
        o = fused_probas(e, a.emb_root, a.feat_dir, ytr, yva, Ftr, Fva, fm, tm, sm)
        Ptu_list.append(o["proba_tune"]); Pte_list.append(o["proba_test"])
        print(f"[final_cm] re-fit fused {e}", flush=True)
    Ptu, Pte = np.stack(Ptu_list, 0), np.stack(Pte_list, 0)
    ytune = yva[tm]
    w_v = weight_search(Ptu, ytune, len(embs))
    cw = calibrate(np.einsum("knc,k->nc", Ptu, w_v), cls, ytune)
    b = additive_bias_search(np.einsum("knc,k->nc", Ptu, w_v) * cw, ytune, cls)      # v4 additive
    laneA_sm = _norm(np.einsum("knc,k->nc", Pte, w_v) * cw * np.exp(b))              # Lane A proba on sm obs

    # --- align Lane A (sm-obs order) back to the his-holdout order ---
    sm_obs = np.flatnonzero(sm)                                # sorted obs indices == rows of Pte/laneA_sm
    obs_to_row = {o: i for i, o in enumerate(sm_obs)}
    laneA = laneA_sm[np.array([obs_to_row[o] for o in obs_idx])]
    laneB = _norm(his_p[his_rows])
    y = yva[obs_idx].astype(int)                               # labels 1..8

    # --- build the final eval mask: BHT-only + exclude w-tuning windows (defaults) ---
    hand_code = HIS_POSITION_NAMES.index("Hand")
    keep = np.ones(len(y), bool)
    if not a.include_hand:
        keep &= his_pos[his_rows] != hand_code
    if not a.include_wtune:
        keep &= assign[obs_idx] != 2                          # original TEST windows were used to pick w
    laneA, laneB, y = laneA[keep], laneB[keep], y[keep]
    print(f"[final_cm] final shared eval set n={len(y)} "
          f"(BHT_only={not a.include_hand}, wtune_excluded={not a.include_wtune}); "
          f"classes present={sorted(set((y-1).tolist()))}", flush=True)

    # --- blend + metrics ---
    W = a.w
    blend = W * laneA + (1 - W) * laneB
    yi = y - 1
    pA, pB, pV5 = laneA.argmax(1), laneB.argmax(1), blend.argmax(1)
    fA, fB, fV5 = per_class_f1(yi, pA), per_class_f1(yi, pB), per_class_f1(yi, pV5)
    mA, mB, mV5 = fA.mean(), fB.mean(), fV5.mean()
    dlt, dlo, dhi, _ = paired_bootstrap_diff(y, cls[pV5], cls[pA if mA >= mB else pB])
    print(f"[final_cm] shared set (n={len(y)}, all 8 classes): LaneA={mA:.4f} LaneB={mB:.4f} "
          f"v5(w={W})={mV5:.4f} | paired Δ(v5-base)={dlt:+.4f} CI[{dlo:+.4f},{dhi:+.4f}]", flush=True)

    # --- figures: the final all-class v5 confusion matrix (+ per-lane for the same set) ---
    plot_cm(yi, pV5, f"v5 (blend, w={W}) — shared all-class held-out set",
            a.out_dir / "cm_v5_final.png", dpi=a.dpi, show_title=st)
    plot_cm(yi, pA, "Lane A (v4) — shared all-class held-out set",
            a.out_dir / "cm_laneA_final.png", dpi=a.dpi, show_title=st)
    plot_cm(yi, pB, "Lane B (vision) — shared all-class held-out set",
            a.out_dir / "cm_laneB_final.png", dpi=a.dpi, show_title=st)
    np.savez_compressed(a.out_dir / "final_cm_dump.npz",
                        y=y.astype(np.int16), laneA=laneA.astype(np.float32),
                        laneB=laneB.astype(np.float32), w=np.float32(W), classes=cls.astype(np.int16))

    CLASSES_N = ["St", "Wa", "Ru", "Bi", "Ca", "Bu", "Tr", "Su"]

    def pcline(tag, f, m):
        return f"| {tag} | {m:.4f} | " + " ".join(f"{CLASSES_N[k]}={f[k]:.2f}" for k in range(8)) + " |"
    md = a.out_dir / "FINAL_CM_RESULTS.md"
    md.write_text("\n".join([
        f"# Final all-class v5 confusion matrix — shared held-out set (n={len(y)}, all 8 classes)",
        "",
        "Shared set = collaborator target_holdout (clean for Lane B) with Lane A re-fit to hold it out "
        f"(clean for Lane A). w={W} pre-locked. BHT_only={not a.include_hand}, "
        f"w-tuning windows excluded={not a.include_wtune}.",
        "",
        "| model | macro-F1 (8-class) | per-class F1 |", "|---|---|---|",
        pcline("Lane A (v4)", fA, mA), pcline("Lane B (vision)", fB, mB), pcline("v5 (blend)", fV5, mV5),
        "",
        f"v5 vs base: paired-bootstrap Δ = {dlt:+.4f}, 95% CI [{dlo:+.4f}, {dhi:+.4f}]"
        + (" (excludes 0 → significant)." if dlo > 0 else " (CI includes 0)."),
    ]) + "\n")
    print(f"[final_cm] wrote {md} + cm_v5_final.png + cm_laneA/B_final.png + final_cm_dump.npz", flush=True)

    if a.mlflow:
        from shl2026 import track
        with track("mdaniol", run_name="final_cm_v5", seed=0, params_path=None,
                   params={"embs": ",".join(embs), "w": W, "split": a.split.stem,
                           "bht_only": not a.include_hand, "wtune_excluded": not a.include_wtune,
                           "protocol": "shared held-out = his target_holdout; Lane A re-fit to exclude it"},
                   tags={"phase": "final_cm", "experiment": "E-COMBINE"}) as run:
            run.log_metrics({"macro_f1": float(mV5), "laneA": float(mA), "laneB": float(mB),
                             "n": int(len(y)), "paired_diff": float(dlt), "paired_ci_lo": float(dlo)})
            run.log_artifact(md)
            for f in ("cm_v5_final.png", "cm_laneA_final.png", "cm_laneB_final.png", "final_cm_dump.npz"):
                p = a.out_dir / f
                if p.exists(): run.log_artifact(p)
        print("[final_cm] MLflow tracked.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
