#!/usr/bin/env python3
"""Two-phase, split-aware handcrafted trainer (the likely big win).

Phase A (honest estimate) — using val_split.npy (60/20/20 of validation):
  fit  = ALL User-1 train  +  validation[FIT]      (model learns Users 2&3 too)
  tune = validation[TUNE]                          (early-stop + threshold calibration)
  test = validation[TEST]  (held-out, Bag/Hips/Torso) -> unbiased macro-F1
  Also trains a User-1-only model to quantify the gain from training on Users 2&3.

Phase B (submission) — retrain on User-1 + ALL validation (nothing wasted),
  apply Phase-A calibration, write submission v3.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import joblib, numpy as np, pandas as pd, lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import f1_score
from metrics import class_report, print_report, save_report

LOCATIONS = ("Bag", "Hips", "Torso", "Hand")
TEST_LOCS = ("Bag", "Hips", "Torso")
CLASSES = list(range(1, 9))
CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]
FIT, TUNE, TEST = 0, 1, 2
N_SAMPLES = 500


def load_split(feat_dir, split):
    df = pd.concat([pd.read_parquet(feat_dir / split / f"{l}.parquet").assign(__loc=l)
                    for l in LOCATIONS], ignore_index=True)
    return df


def macro_f1(y, p):
    return f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)


def fit_lgb(Xtr, ytr, Xtu, ytu, n_est=3000):
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=n_est,
                             learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, class_weight="balanced", n_jobs=-1, verbosity=-1,
                             random_state=0, deterministic=True, force_row_wise=True)
    clf.fit(Xtr, ytr, eval_set=[(Xtu, ytu)], eval_metric="multi_logloss",
            callbacks=[early_stopping(150)])
    return clf


def calibrate(proba, classes, y):
    """Per-class multipliers maximizing macro-F1 (coordinate ascent)."""
    w = np.ones(len(classes)); grid = np.linspace(0.3, 3.0, 28)
    f = lambda ww: macro_f1(y, classes[(proba * ww).argmax(1)])
    for _ in range(4):
        improved = False
        for c in range(len(classes)):
            best_v, best = w[c], f(w)
            for v in grid:
                w2 = w.copy(); w2[c] = v
                s = f(w2)
                if s > best:
                    best, best_v = s, v
            if best_v != w[c]:
                w[c] = best_v; improved = True
        if not improved:
            break
    return w


def report(tag, y, p, loc):
    a = macro_f1(y, p); t = macro_f1(y[np.isin(loc, TEST_LOCS)], p[np.isin(loc, TEST_LOCS)])
    print(f"  {tag:34s} all={a:.4f}  test-loc={t:.4f}")
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]; here = Path(__file__).resolve().parent
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=here / "artifacts" / "val_split.npy")
    ap.add_argument("--out", type=Path, default=here / "AGH_predictions_v3.txt")
    ap.add_argument("--aug-train-dir", type=Path, default=None,
                    help="dir of Branch-A augmented TRAIN feature parquets "
                         "(augment_features.py --branch A). Adds a +AUG held-out row.")
    ap.add_argument("--use-aug-for-submission", action="store_true",
                    help="include augmented copies in the Phase-B final model "
                         "(decide AFTER seeing the Phase-A +AUG row).")
    args = ap.parse_args(); t0 = time.time()

    train = load_split(args.feat_dir, "train")
    val = load_split(args.feat_dir, "validation")
    assign = np.load(args.split)
    assert len(assign) == len(val), "split/val length mismatch"
    feat = [c for c in train.columns if c not in ("label", "__loc")]

    Xtr, ytr = train[feat].to_numpy(np.float32), train["label"].to_numpy()
    Xv, yv, locv = val[feat].to_numpy(np.float32), val["label"].to_numpy(), val["__loc"].to_numpy()
    fm, tm, sm = assign == FIT, assign == TUNE, assign == TEST

    Xtune, ytune = Xv[tm], yv[tm]
    Xtest, ytest, loctest = Xv[sm], yv[sm], locv[sm]

    print("=== Phase A: held-out estimate (test = Users2&3 held-out, Bag/Hips/Torso) ===")
    # baseline: User-1-only
    clf0 = fit_lgb(Xtr, ytr, Xtune, ytune)
    rep_u1 = class_report(ytest, clf0.predict(Xtest)); print_report("User1-only", rep_u1)
    # + validation fit
    Xfit = np.concatenate([Xtr, Xv[fm]]); yfit = np.concatenate([ytr, yv[fm]])
    clf = fit_lgb(Xfit, yfit, Xtune, ytune)
    rep_vf = class_report(ytest, clf.predict(Xtest)); print_report("User1 + val-fit", rep_vf)
    # + calibration (tuned on TUNE)
    w = calibrate(clf.predict_proba(Xtune), clf.classes_, ytune)
    pred_cal = clf.classes_[(clf.predict_proba(Xtest) * w).argmax(1)]
    rep_cal = class_report(ytest, pred_cal); print_report("User1 + val-fit + calib", rep_cal)
    held = rep_cal["macro_f1"]

    # + train-time augmentation (Branch A: scale/jitter/time-warp on User-1) ----
    rep_aug, Xaug, yaug = None, None, None
    if args.aug_train_dir is not None:
        aug_df = load_split(args.aug_train_dir, "")  # aug-train-dir/<LOC>.parquet
        Xaug, yaug = aug_df[feat].to_numpy(np.float32), aug_df["label"].to_numpy()
        Xfit_a = np.concatenate([Xfit, Xaug]); yfit_a = np.concatenate([yfit, yaug])
        clf_a = fit_lgb(Xfit_a, yfit_a, Xtune, ytune)
        w_a = calibrate(clf_a.predict_proba(Xtune), clf_a.classes_, ytune)
        pred_a = clf_a.classes_[(clf_a.predict_proba(Xtest) * w_a).argmax(1)]
        rep_aug = class_report(ytest, pred_a); print_report("User1 + val-fit + AUG + calib", rep_aug)
        d = rep_aug["macro_f1"] - held
        print(f"  >>> augmentation effect on held-out macro-F1: {d:+.4f} "
              f"({'KEEP' if d > 0 else 'DROP'} per AUGMENTATION_STRATEGY.md §8)")

    print(f"\n=== Phase B: final model on User1 + ALL validation -> submission ===")
    parts_X, parts_y = [Xtr, Xv], [ytr, yv]
    if args.use_aug_for_submission and Xaug is not None:
        print("  including augmented copies in the submission model")
        parts_X.append(Xaug); parts_y.append(yaug)
    Xfin = np.concatenate(parts_X); yfin = np.concatenate(parts_y)
    best_it = int(clf.best_iteration_ or 1500)
    final = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=best_it,
                               learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, class_weight="balanced", n_jobs=-1, verbosity=-1,
                               random_state=0, deterministic=True, force_row_wise=True)
    final.fit(Xfin, yfin)
    test = pd.read_parquet(args.feat_dir / "test" / "all.parquet")
    pred = final.classes_[(final.predict_proba(test[feat].to_numpy(np.float32)) * w).argmax(1)].astype(int)
    u, c = np.unique(pred, return_counts=True)
    print("  test pred dist:", {CLASS_NAMES[k-1]: round(n/len(pred), 3) for k, n in zip(u, c)})
    np.savetxt(args.out, np.repeat(pred[:, None], N_SAMPLES, axis=1), fmt="%d", delimiter=" ")
    # `model` = Phase-B submission model (saw all validation). `model_heldout` =
    # Phase-A model (User1 + val[FIT] only — never saw TEST/TUNE), required for an
    # HONEST robustness sweep on the held-out TEST slice (robustness_sweep.py).
    joblib.dump({"model": final, "model_heldout": clf, "feat_cols": feat, "weights": w},
                here / "artifacts" / "split_lgbm.joblib")
    save_report(here / "artifacts" / "split_results.json", {
        "heldout_user1_only": rep_u1,
        "heldout_user1_plus_valfit": rep_vf,
        "heldout_user1_plus_valfit_calibrated": rep_cal,
        "heldout_user1_plus_valfit_aug_calibrated": rep_aug,
        "used_aug_for_submission": bool(args.use_aug_for_submission and Xaug is not None),
        "best_iteration": best_it,
        "calibration_weights": {CLASS_NAMES[i]: round(float(w[i]), 3) for i in range(8)},
        "test_pred_distribution": {CLASS_NAMES[k-1]: int(n) for k, n in zip(u, c)},
    })
    print(f"  wrote {args.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
