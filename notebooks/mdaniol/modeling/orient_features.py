#!/usr/bin/env python3
"""Orientation-invariant features — target the TORSO placement fragility (ROBUSTNESS_DIAG: Torso
macro 0.766 vs Bag/Hips ~0.86, Bike=0.44). The hidden test has no location label, so features that are
invariant to the phone's orientation should generalize across placements better than the magnitude-only
520. Two physics-grounded, rotation-invariant blocks computed from the RAW acc/gyr/mag windows:

  gravity V/H  — decompose dynamic acc into vertical (along gravity) vs horizontal; RMS_v, RMS_h,
                 log-var ratio, gravity magnitude, gravity-direction dispersion (placement wobble).
  SO(3) invariants — acc·gyro angle, scalar triple product a·(ω×m), acc·gravity angle (mean+std).
                 These are invariant to ANY global rotation of the phone (proven by the test).

Experiment: does emb⊕520⊕orient beat emb⊕520 on the honest TEST, especially on TORSO? fit_cal_eval
protocol, TEST locked once, per-location reported. Reuses probe_fusion.{load_emb,load_feats,
load_labels,fit_cal_eval,split_locs,macro_f1,CLASSES,class-report via metrics}. Streams raw per
location (memory-safe).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_emb, load_feats, load_labels, fit_cal_eval, split_locs, macro_f1, CLASSES  # noqa: E402

EPS = 1e-12
ACC = ("Acc_x", "Acc_y", "Acc_z"); GYR = ("Gyr_x", "Gyr_y", "Gyr_z"); MAG = ("Mag_x", "Mag_y", "Mag_z")


def _angle(a, b):
    """Per-sample angle between two (n,3,T) vector streams → (n,T), rotation-invariant."""
    dot = (a * b).sum(1)
    return np.arccos(np.clip(dot / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + EPS), -1, 1))


def gravity_vh(acc):
    """acc (n,3,T) → (n,5) gravity vertical/horizontal features (rotation-invariant)."""
    g = acc.mean(2)
    gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + EPS)
    a_dyn = acc - acc.mean(2, keepdims=True)
    a_v = np.einsum("nct,nc->nt", a_dyn, gn)
    a_h = np.linalg.norm(a_dyn - a_v[:, None, :] * gn[:, :, None], axis=1)
    rms_v = np.sqrt((a_v ** 2).mean(1)); rms_h = np.sqrt((a_h ** 2).mean(1))
    logratio = np.log((a_v.var(1) + EPS) / (a_h.var(1) + EPS))
    grav_mag = np.linalg.norm(g, axis=1)
    sl = np.array_split(np.arange(acc.shape[2]), 10)
    gdir = np.stack([acc[:, :, s].mean(2) for s in sl], 1)
    gdir = gdir / (np.linalg.norm(gdir, axis=2, keepdims=True) + EPS)
    disp = 1.0 - np.linalg.norm(gdir.mean(1), axis=1)                # gravity-direction wobble
    return (np.stack([rms_v, rms_h, logratio, grav_mag, disp], 1),
            ["grav_rms_v", "grav_rms_h", "grav_logvar_vh", "grav_mag", "grav_dir_disp"])


def so3_invariants(acc, gyr, mag):
    """(n,3,T) acc/gyr/mag → (n,6) cross-sensor SO(3) invariants (mean+std), rotation-invariant."""
    th_ag = _angle(acc, gyr)
    triple = (acc * np.cross(gyr, mag, axis=1)).sum(1)               # a·(ω×m), invariant scalar
    th_acrg = _angle(acc, np.broadcast_to(acc.mean(2, keepdims=True), acc.shape))
    return (np.stack([th_ag.mean(1), th_ag.std(1), triple.mean(1), triple.std(1),
                      th_acrg.mean(1), th_acrg.std(1)], 1),
            ["so3_accgyr_mean", "so3_accgyr_std", "so3_triple_mean", "so3_triple_std",
             "so3_accgrav_mean", "so3_accgrav_std"])


def orient_block(acc, gyr, mag):
    a, an = gravity_vh(acc); b, bn = so3_invariants(acc, gyr, mag)
    return np.concatenate([a, b], 1).astype(np.float32), an + bn


def compute_orient(data_dir, split):
    """Streamed per-location orient features for SPLIT, concatenated in split_locs order."""
    blocks = []
    for loc in split_locs(split):
        df = pd.read_parquet(data_dir / split / f"{loc}.parquet", columns=list(ACC + GYR + MAG))
        acc = np.stack([np.stack(df[c].to_numpy()) for c in ACC], 1).astype(np.float64)
        gyr = np.stack([np.stack(df[c].to_numpy()) for c in GYR], 1).astype(np.float64)
        mag = np.stack([np.stack(df[c].to_numpy()) for c in MAG], 1).astype(np.float64)
        feats, names = orient_block(acc, gyr, mag)
        blocks.append(feats)
        print(f"  orient {split}/{loc}: {feats.shape}", flush=True)
    out = np.concatenate(blocks, 0)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out, names


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--emb", default="utica_V2")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--data-dir", type=Path, default=root / "dataset_parquet")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol" / "ORIENT_FEATURES_RESULTS.md")
    args = ap.parse_args()
    from metrics import class_report
    cls = np.asarray(CLASSES)

    assign = np.load(args.split)
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, lva = load_labels(args.feat_dir, "validation")
    E_tr, E_va = load_emb(args.emb_root / args.emb, "train"), load_emb(args.emb_root / args.emb, "validation")
    F_tr, F_va = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
    O_tr, onames = compute_orient(args.data_dir, "train")
    O_va, _ = compute_orient(args.data_dir, "validation")
    assert len(assign) == len(yva) == len(E_va) == len(F_va) == len(O_va), "length mismatch"
    fm, tm, sm = assign == 0, assign == 1, assign == 2
    ytune, ytest, loc_test = yva[tm], yva[sm], lva[sm]

    def evalrep(name, Xtr_blk, Xva_blk):
        Xfit = np.concatenate([Xtr_blk, Xva_blk[fm]]); yfit = np.concatenate([ytr, yva[fm]])
        o = fit_cal_eval(name, Xfit, yfit, Xva_blk[tm], ytune, Xva_blk[sm], ytest, return_probs=True)
        pred = cls[o["proba_test"].argmax(1)]
        per_loc = {loc: class_report(ytest[loc_test == loc], pred[loc_test == loc])["macro_f1"]
                   for loc in ("Bag", "Hips", "Torso") if (loc_test == loc).any()}
        return o["rep"]["macro_f1"], o["rep_tune"]["macro_f1"], per_loc

    base = np.concatenate([E_tr, F_tr], 1), np.concatenate([E_va, F_va], 1)
    aug = np.concatenate([E_tr, F_tr, O_tr], 1), np.concatenate([E_va, F_va, O_va], 1)
    b_te, b_tu, b_loc = evalrep("fusion", *base)
    a_te, a_tu, a_loc = evalrep("fusion+orient", *aug)

    lines = [f"# Orientation-invariant features ({len(onames)}) on TORSO fragility — emb={args.emb}.",
             f"orient block: {', '.join(onames)}\n",
             "| rep | TEST macro | TUNE macro | Bag | Hips | **Torso** |", "|---|---|---|---|---|---|",
             f"| fusion (emb⊕520) | {b_te:.4f} | {b_tu:.4f} | {b_loc.get('Bag',0):.3f} | "
             f"{b_loc.get('Hips',0):.3f} | {b_loc.get('Torso',0):.3f} |",
             f"| +orient | {a_te:.4f} | {a_tu:.4f} | {a_loc.get('Bag',0):.3f} | "
             f"{a_loc.get('Hips',0):.3f} | {a_loc.get('Torso',0):.3f} |",
             f"\n- **orient Δ macro = {a_te-b_te:+.4f}**; **Δ Torso = {a_loc.get('Torso',0)-b_loc.get('Torso',0):+.4f}** "
             f"(the placement-fragility target).",
             f"- {'KEEP — orient helps' if a_te > b_te + 0.001 else 'DISABLE — no honest gain'} "
             f"(gate on TEST + per-location)."]
    args.out.write_text("\n".join(lines) + "\n")
    for ln in lines:
        print(ln, flush=True)
    print(f"[orient] wrote {args.out}", flush=True)

    from shl2026 import track
    with track("mdaniol", run_name=f"orient_{args.emb}", seed=0, params_path=None,
               params={"emb": args.emb, "split": args.split.stem, "n_orient": len(onames),
                       "protocol": "emb⊕520 vs emb⊕520⊕orient; per-location; honest temporal+embargo"},
               tags={"phase": "feature", "experiment": "orient-invariant"}) as run:
        run.log_metrics({"test_fusion": b_te, "test_orient": a_te, "delta": a_te - b_te,
                         "torso_fusion": b_loc.get("Torso", 0.0), "torso_orient": a_loc.get("Torso", 0.0)})
        run.log_artifact(args.out)
    print("[orient] MLflow tracked.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
