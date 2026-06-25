#!/usr/bin/env python3
"""VehicleExpert (task #13) — generic gated subset corrector over any 8-class base.

Generalizes the rail expert: redistributes probability mass WITHIN a confusable
class subset, only when confidently warranted. Evaluated on the calibrated
MOMENT-fusion base under the conservative temporal split; thresholds selected on
TUNE, confirmed on TEST. Gated KEEP/DISABLE per the decision criteria.

Subsets: V4 {Car,Bus,Train,Subway}, SV5 {Still,+vehicles}, R2 {Train,Subway}.
Expert features = base subset-probabilities ⊕ magnetometer features. Leakage-safe:
base + expert fit on FIT, calibrate/select on TUNE, confirm on TEST.

Reuses: submit_fusion.fuse, probe_fusion.{fit_cal_eval,aligned_proba,load_*},
rail_expert.mag_indices, split.load_split_with_location_map, shl2026.evaluate_predictions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_labels, fit_cal_eval, aligned_proba  # noqa: E402
from split import load_split_with_location_map  # noqa: E402
from submit_fusion import fuse  # noqa: E402
from rail_expert import mag_indices  # noqa: E402

CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]
BHT = ("Bag", "Hips", "Torso")
FIT, TUNE, TEST = 0, 1, 2
SUBSETS = {"V4": [5, 6, 7, 8], "SV5": [1, 5, 6, 7, 8], "R2": [7, 8]}   # class ids


def train_subset_expert(P_fit, y_fit, Xmag_fit, classes):
    """Specialist over `classes`, trained only on those examples.
    Features = base probabilities of the subset ⊕ magnetometer features."""
    cidx = [c - 1 for c in classes]
    m = np.isin(y_fit, classes)
    feats = np.concatenate([P_fit[m][:, cidx], Xmag_fit[m]], axis=1)
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[v] for v in y_fit[m]])
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(classes), n_estimators=400,
                             learning_rate=0.03, num_leaves=15, min_child_samples=100,
                             reg_lambda=2.0, class_weight="balanced", n_jobs=-1,
                             verbosity=-1, random_state=0).fit(feats, y)
    col = {c: i for i, c in enumerate(clf.classes_)}
    order = [col[i] for i in range(len(classes))]                  # align to `classes` order
    return clf, cidx, order


def apply_subset_expert(P_base, Xmag, clf, cidx, order, tau_mass, tau_conf, lam):
    """Gated mass-preserving correction within the subset; returns labels 1..8."""
    n = len(P_base)
    sub = P_base[:, cidx]
    mass = sub.sum(1)
    base_norm = sub / (sub.sum(1, keepdims=True) + 1e-9)
    feats = np.concatenate([sub, Xmag], axis=1)
    pe = clf.predict_proba(feats)[:, order]                       # (n, |subset|), subset order
    conf = pe.max(1)
    fire = (mass > tau_mass) & (conf > tau_conf)
    out = P_base.copy()
    blended = (1 - lam) * base_norm + lam * pe
    rows = np.where(fire)[0]
    out[np.ix_(rows, cidx)] = mass[rows, None] * blended[rows]
    out = out / out.sum(1, keepdims=True)
    return (out.argmax(1) + 1), int(fire.sum())                   # labels 1..8


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--emb", default="moment-small_V1")
    ap.add_argument("--subset", choices=list(SUBSETS), default="V4")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol/VEHICLE_EXPERT_RESULTS.md")
    args = ap.parse_args()
    from shl2026 import track, evaluate_predictions
    import pandas as pd

    classes = SUBSETS[args.subset]
    cls8 = np.arange(1, 9)
    cols = list(pd.read_parquet(args.feat_dir / "train" / "Bag.parquet").columns)
    feat_names = [c for c in cols if c != "label"]                # load_feats order (no label)
    mc = [i for i, c in enumerate(feat_names) if c.startswith(("mag_mag__", "mag_rate_mag__"))]

    Xtr = fuse(args.feat_dir, args.emb_root, args.emb, "train")
    Xva = fuse(args.feat_dir, args.emb_root, args.emb, "validation")
    ytr, _ = load_labels(args.feat_dir, "train"); yva, _ = load_labels(args.feat_dir, "validation")
    # fused = [emb | 520 handcrafted]; mag cols sit in the handcrafted block.
    n_emb = Xtr.shape[1] - len(feat_names)
    mc_fused = [n_emb + i for i in mc]

    assign, loc_off = load_split_with_location_map(args.split, args.feat_dir)
    va_loc = np.empty(len(assign), dtype=object)
    for loc, (s, e) in loc_off.items():
        va_loc[s:e] = loc
    bht = np.isin(va_loc.astype(str), BHT)
    fit_m, tune_m, test_m = assign == FIT, (assign == TUNE) & bht, (assign == TEST) & bht

    Xfit = np.concatenate([Xtr, Xva[fit_m]]); yfit = np.concatenate([ytr, yva[fit_m]])
    out = fit_cal_eval(f"base({args.emb})", Xfit, yfit, Xva[tune_m], yva[tune_m],
                       Xva[test_m], yva[test_m], return_probs=True)
    Pt, Ps = out["proba_tune"], out["proba_test"]
    ytune, ytest = yva[tune_m], yva[test_m]
    Xmag_fit = np.concatenate([Xtr[:, mc_fused], Xva[fit_m][:, mc_fused]])
    Xmag_t, Xmag_s = Xva[tune_m][:, mc_fused], Xva[test_m][:, mc_fused]

    base_t = evaluate_predictions(ytune, cls8[Pt.argmax(1)])
    base_s = evaluate_predictions(ytest, cls8[Ps.argmax(1)])
    Pfit = aligned_proba(out["model"], Xfit, out["weights"])
    clf, cidx, order = train_subset_expert(Pfit, yfit, Xmag_fit, classes)

    def vmacro(res):                                              # vehicle-subset macro-F1
        return float(np.mean([res.per_class_f1[c] for c in classes]))

    rows = []
    for tm_ in (0.40, 0.50, 0.60, 0.70):
        for tc in (0.45, 0.55, 0.65, 0.75):
            for lam in (0.25, 0.50, 0.75):
                pt, _ = apply_subset_expert(Pt, Xmag_t, clf, cidx, order, tm_, tc, lam)
                ps, nf = apply_subset_expert(Ps, Xmag_s, clf, cidx, order, tm_, tc, lam)
                r_t, r_s = evaluate_predictions(ytune, pt), evaluate_predictions(ytest, ps)
                rows.append((tm_, tc, lam, r_t.macro_f1, r_s.macro_f1,
                             r_s.macro_f1 - base_s.macro_f1, vmacro(r_s) - vmacro(base_s), nf))
    best = max(rows, key=lambda r: r[3])                         # select on TUNE macro
    keep = best[5] > 0.003 or (best[6] > 0.010 and best[5] >= 0)
    print(f"\n[{args.subset}] base TEST macro={base_s.macro_f1:.4f} vehicle-macro={vmacro(base_s):.4f}")
    print(f"  best (TUNE-selected) tau_mass={best[0]} tau_conf={best[1]} lambda={best[2]}: "
          f"TEST macro={best[4]:.4f} (Δ{best[5]:+.4f}) vehicle Δ{best[6]:+.4f} fired={best[7]} "
          f"-> {'KEEP' if keep else 'DISABLE'}")

    # write the result table first, then track + SNAPSHOT it as an MLflow artifact (rule §8).
    hdr = (f"# VehicleExpert {args.subset} {classes} on MOMENT-fusion ({args.emb}), temporal split.\n"
           f"base TEST macro={base_s.macro_f1:.4f}, vehicle-macro={vmacro(base_s):.4f}; "
           f"**{'KEEP' if keep else 'DISABLE'}** (selected on TUNE).\n\n"
           "| tau_mass | tau_conf | lambda | TUNE macro | TEST macro | Δmacro | Δvehicle | fired |\n"
           "|---|---|---|---|---|---|---|---|\n")
    body = "".join(f"| {a} | {b} | {c} | {d:.4f} | {e:.4f} | {f:+.4f} | {g:+.4f} | {h} |\n"
                   for a, b, c, d, e, f, g, h in rows)
    args.out.write_text(hdr + body)

    with track("mdaniol", run_name=f"vexpert_{args.subset}", seed=0, params_path=None,
               params={"subset": args.subset, "emb": args.emb, "split": args.split.stem,
                       "tau_mass": best[0], "tau_conf": best[1], "lambda": best[2]},
               tags={"phase": "vexpert", "branch": "vehicle_expert",
                     "decision": "KEEP" if keep else "DISABLE"}) as run:
        run.log_metrics({"macro_f1": best[4],              # bare key -> team leaderboard
                         "base_macro_f1": base_s.macro_f1, "delta_macro": best[5],
                         "delta_vehicle_macro": best[6], "fired": best[7]})
        run.log_artifact(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
