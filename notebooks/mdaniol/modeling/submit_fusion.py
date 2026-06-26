#!/usr/bin/env python3
"""Final submission — MOMENT-fusion model (handcrafted ⊕ MOMENT-small_V1 -> LightGBM).

The first ≥0.80 recipe: in the conservative temporal split this fused model scored
~0.803 macro-F1 (vs handcrafted 0.786), and the gain survived the leakage-corrected
evaluation. This is the SIMPLE, robust deployment — one fused model, no test-time
router/experts (the soft-MoE +1.4 is a possible v2).

Phase B (correct for a final submission): train on ALL labelled data — User-1 train
+ ALL validation (Users 2&3) — then predict the unlabelled test. We never see test
labels; the expected score is the held-out estimate already measured (~0.80, bracket
[0.725, 0.803]). Deterministic (seeded). Writes a validated 92726x500 matrix via
shl2026.write_submission and appends a row to SUBMISSIONS.md.

Reuses: probe_fusion.{load_feats,load_emb,load_labels}, shl2026.write_submission.

Usage:
    python submit_fusion.py --emb moment-small_V1 --version v1 --team AGH
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_feats, load_emb, load_labels, calibrate, aligned_proba, CLASSES  # noqa: E402
from split import load_split_with_location_map  # noqa: E402

N_TEST = 92726
FIT, TUNE, TEST = 0, 1, 2
CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_HERE, text=True).strip()
    except Exception:
        return "unknown"


def fuse(feat_dir: Path, emb_root: Path, emb: str, split: str) -> np.ndarray:
    """Concatenate the 520 handcrafted features with the cached FM embeddings."""
    F = load_feats(feat_dir, split)
    E = load_emb(emb_root / emb, split)
    assert len(F) == len(E), f"{split}: feat {len(F)} vs emb {len(E)} length mismatch"
    return np.concatenate([E, F], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--emb", default="moment-small_V1")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--n-estimators", type=int, default=2000,
                    help="ceiling; early-stopping on the calibration slice picks the actual count.")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy",
                    help="split for the FIT(train)/TUNE(calibration) protocol that matched 0.803.")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--team", default="AGH")
    ap.add_argument("--heldout", default="~0.803 (temporal; bracket 0.725-0.803)",
                    help="held-out macro-F1 estimate for the SUBMISSIONS row + MLflow tag "
                         "(this recipe's bake-off lock value; the real test is NOT scored here). "
                         "e.g. utica_V2 -> '0.8157 (temporal lock)'.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from shl2026 import write_submission, track

    cfg = f"fusion({args.emb})+cal"
    out = args.out or (root / "notebooks/mdaniol" /
                       f"{args.team}_predictions_{args.version}_{args.emb}-fusion.txt")

    # Reproduce the evaluated 0.803 recipe: fit on User-1 + validation[FIT], early-stop
    # AND per-class calibrate on validation[TUNE] (calibration is ~+0.16 macro — it
    # rescues rare classes, esp. Run 0.20->~0.75), then predict the unlabelled test.
    Xtr_u1 = fuse(args.feat_dir, args.emb_root, args.emb, "train")
    Xva = fuse(args.feat_dir, args.emb_root, args.emb, "validation")
    ytr_u1 = load_labels(args.feat_dir, "train")[0]
    yva = load_labels(args.feat_dir, "validation")[0]
    assign, _ = load_split_with_location_map(args.split, args.feat_dir)
    assert len(assign) == len(yva), "split/validation length mismatch"
    fit_m, tune_m = assign == FIT, assign == TUNE

    Xfit = np.concatenate([Xtr_u1, Xva[fit_m]]); yfit = np.concatenate([ytr_u1, yva[fit_m]])
    Xcal, ycal = Xva[tune_m], yva[tune_m]
    print(f"[submit] fit X={Xfit.shape} (User-1 + val[FIT]); calibrate on val[TUNE] n={len(ycal)}",
          flush=True)
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=args.n_estimators,
                             learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, class_weight="balanced", n_jobs=-1,
                             verbosity=-1, random_state=0)
    clf.fit(Xfit, yfit, eval_set=[(Xcal, ycal)], eval_metric="multi_logloss",
            callbacks=[early_stopping(100)])
    w = calibrate(clf.predict_proba(Xcal), clf.classes_, ycal)     # per-class macro-F1 weights

    # --- predict the unlabelled test (calibrated) ------------------------------
    Xte = fuse(args.feat_dir, args.emb_root, args.emb, "test")
    assert len(Xte) == N_TEST, f"expected {N_TEST} test frames, got {len(Xte)}"
    pred = np.asarray(CLASSES)[aligned_proba(clf, Xte, w).argmax(1)].astype(int)
    uniq, cnt = np.unique(pred, return_counts=True)
    dist = {CLASS_NAMES[c - 1]: round(n / len(pred), 3) for c, n in zip(uniq, cnt)}
    print(f"[submit] test pred distribution: {dist}", flush=True)
    assert set(uniq).issubset(set(range(1, 9))), "labels outside 1..8"

    rep = write_submission(pred, out)                  # writes 92726x500 + validates
    print(f"[submit] wrote {out}  ok={rep.ok} rows={rep.n_rows} cols={rep.n_cols}", flush=True)
    if not rep.ok:
        print(f"[submit] VALIDATION FAILED: {rep.messages}", file=sys.stderr); return 1

    # --- append provenance row to SUBMISSIONS.md -------------------------------
    log = root / "notebooks/mdaniol/SUBMISSIONS.md"
    if not log.exists():
        log.write_text("# Submissions log\n\n"
                       "| version | date (UTC) | recipe | git SHA | file | held-out (conservative) | notes |\n"
                       "|---|---|---|---|---|---|---|\n")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    log.write_text(log.read_text() + (
        f"| {args.version} | {ts} | {cfg} | {_git_sha()} | {out.name} "
        f"| {args.heldout} | FIT-train + TUNE-calibrated; det |\n"))
    print(f"[submit] logged -> {log}", flush=True)

    # MLflow traceability (rule §8): snapshot the submission + log as a tracked run. We log the
    # held-out *estimate* as a tag and the predicted test class balance as metrics — NOT a bare
    # macro_f1 (the real test is unlabelled here, so there is no fresh measurement to leaderboard).
    with track("mdaniol", run_name=f"submit_{args.version}_{args.emb}", seed=0, params_path=None,
               params={"emb": args.emb, "version": args.version, "split": args.split.stem,
                       "team": args.team, "git_sha": _git_sha(),
                       "protocol": "PhaseB: fit(User1+val[FIT]) -> cal(val[TUNE]) -> predict test"},
               tags={"phase": "submission", "branch": "submit", "heldout_estimate": args.heldout}) as run:
        run.log_metrics({f"test_pred_frac_{CLASS_NAMES[c - 1]}": float(n / len(pred))
                         for c, n in zip(uniq, cnt)})
        run.log_artifact(out)            # the validated 92726x500 submission matrix
        run.log_artifact(log)            # SUBMISSIONS.md provenance row
    print(f"[submit] MLflow tracked (run submit_{args.version}_{args.emb}); artifacts snapshotted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
