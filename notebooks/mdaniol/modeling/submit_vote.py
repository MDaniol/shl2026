#!/usr/bin/env python3
"""Final submission — calibrated SOFT-VOTE of the top-2 fused FMs (E-VOTE-01, KEEP).

The vote (utica_V2 + mantisv2_V1, weighted + per-class recal) scored 0.8342 macro-F1 on
the temporal lock vs 0.8213 for the best single FM (+0.0129) — see VOTING_HEAD_RESULTS.md /
LIGHTWEIGHT_HEAD_PLAN E-VOTE-01. This ships that exact recipe in Phase B.

Phase B: for EACH FM, fit [emb ⊕ 520 handcrafted] -> LightGBM on User-1 + validation[FIT],
per-class calibrate on validation[TUNE]; weights + a final per-class recalibration are
selected on TUNE (identical to voting_head.py); then predict the unlabelled test. Never
sees test labels — expected score is the measured lock value (0.8342). Deterministic.

Reuses: submit_fusion.fuse, voting_head.weight_search, probe_fusion.{load_labels,calibrate,
aligned_proba,macro_f1,CLASSES}, split.load_split_with_location_map, shl2026.{write_submission,track}.

    python submit_vote.py --embs utica_V2,mantisv2_V1 --version v3 --team AGH \
        --heldout "0.8342 (temporal lock; E-VOTE-01 weighted+recal)"
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_labels, calibrate, aligned_proba, macro_f1, CLASSES  # noqa: E402
from submit_fusion import fuse, _git_sha, N_TEST, CLASS_NAMES  # noqa: E402
from split import load_split_with_location_map  # noqa: E402
from voting_head import weight_search  # noqa: E402

FIT, TUNE = 0, 1


def fit_one_fm(emb, feat_dir, emb_root, n_estimators, ytr, yva, fit_m, tune_m):
    """Fit one FM's fusion (Phase B), calibrate on TUNE -> class-aligned calibrated probas
    on TUNE and on the real (unlabelled) test."""
    Xtr = fuse(feat_dir, emb_root, emb, "train")
    Xva = fuse(feat_dir, emb_root, emb, "validation")
    Xte = fuse(feat_dir, emb_root, emb, "test")
    assert len(Xte) == N_TEST, f"{emb}: expected {N_TEST} test frames, got {len(Xte)}"
    Xfit = np.concatenate([Xtr, Xva[fit_m]]); yfit = np.concatenate([ytr, yva[fit_m]])
    Xcal, ycal = Xva[tune_m], yva[tune_m]
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=n_estimators,
                             learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, class_weight="balanced", n_jobs=-1,
                             verbosity=-1, random_state=0, deterministic=True, force_col_wise=True)
    clf.fit(Xfit, yfit, eval_set=[(Xcal, ycal)], eval_metric="multi_logloss",
            callbacks=[early_stopping(100)])
    w = calibrate(clf.predict_proba(Xcal), clf.classes_, ycal)
    return aligned_proba(clf, Xcal, w), aligned_proba(clf, Xte, w), ycal


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--embs", default="utica_V2,mantisv2_V1", help="top-K from E-VOTE-01")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--team", default="AGH")
    ap.add_argument("--heldout", default="0.8342 (temporal lock; E-VOTE-01 weighted+recal)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    from shl2026 import write_submission, track

    embs = [e.strip() for e in args.embs.split(",") if e.strip()]
    cls = np.asarray(CLASSES)
    cfg = f"vote({'+'.join(embs)})weighted+recal"
    out = args.out or (root / "notebooks/mdaniol" /
                       f"{args.team}_predictions_{args.version}_vote.txt")

    ytr = load_labels(args.feat_dir, "train")[0]
    yva = load_labels(args.feat_dir, "validation")[0]
    assign, _ = load_split_with_location_map(args.split, args.feat_dir)
    assert len(assign) == len(yva), "split/validation length mismatch"
    fit_m, tune_m = assign == FIT, assign == TUNE

    # per-FM calibrated probas (TUNE for selection, test for the actual prediction)
    Ptu_list, Pte_list, ycal = [], [], None
    for e in embs:
        Ptu, Pte, ycal = fit_one_fm(e, args.feat_dir, args.emb_root, args.n_estimators,
                                    ytr, yva, fit_m, tune_m)
        Ptu_list.append(Ptu); Pte_list.append(Pte)
        print(f"[vote] fit {e}: TUNE n={len(ycal)}, test probas {Pte.shape}", flush=True)
    Ptu = np.stack(Ptu_list, 0); Pte = np.stack(Pte_list, 0)        # (K, n, 8)

    # weights + per-class recal selected ON TUNE (identical to voting_head's kept variant)
    w = weight_search(Ptu, ycal, len(embs))
    wv_tu = np.einsum("knc,k->nc", Ptu, w)
    cw = calibrate(wv_tu, cls, ycal)
    tune_macro = macro_f1(ycal, cls[(wv_tu * cw).argmax(1)])
    print(f"[vote] weights={dict(zip(embs, np.round(w,3)))} TUNE macro(weighted+recal)={tune_macro:.4f}",
          flush=True)

    # predict the unlabelled test with the locked weights + recal
    wv_te = np.einsum("knc,k->nc", Pte, w)
    pred = cls[(wv_te * cw).argmax(1)].astype(int)
    assert len(pred) == N_TEST and set(np.unique(pred)).issubset(set(range(1, 9)))
    uniq, cnt = np.unique(pred, return_counts=True)
    dist = {CLASS_NAMES[c - 1]: round(n / len(pred), 3) for c, n in zip(uniq, cnt)}
    print(f"[vote] test pred distribution: {dist}", flush=True)

    rep = write_submission(pred, out)
    print(f"[vote] wrote {out} ok={rep.ok} rows={rep.n_rows} cols={rep.n_cols}", flush=True)
    if not rep.ok:
        print(f"[vote] VALIDATION FAILED: {rep.messages}", file=sys.stderr); return 1

    log = root / "notebooks/mdaniol/SUBMISSIONS.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    log.write_text(log.read_text() + (
        f"| {args.version} | {ts} | {cfg} | {_git_sha()} | {out.name} "
        f"| {args.heldout} | Phase-B; weights+recal on TUNE; det |\n"))
    print(f"[vote] logged -> {log}", flush=True)

    # MLflow traceability (rule §8): tracked run + artifact snapshot. Held-out is an ESTIMATE
    # (test unlabelled here) -> log as a tag + tune macro + class balance, not a bare macro_f1.
    with track("mdaniol", run_name=f"submit_{args.version}_vote", seed=0, params_path=None,
               params={"embs": ",".join(embs), "version": args.version, "split": args.split.stem,
                       "weights": ",".join(f"{x:.3f}" for x in w), "team": args.team,
                       "git_sha": _git_sha(), "protocol": "PhaseB vote: fit+cal each -> weights+recal on TUNE"},
               tags={"phase": "submission", "branch": "submit_vote", "heldout_estimate": args.heldout}) as run:
        run.log_metrics({"tune_macro_f1": float(tune_macro),
                         **{f"test_pred_frac_{CLASS_NAMES[c-1]}": float(n/len(pred))
                            for c, n in zip(uniq, cnt)}})
        run.log_artifact(out)
        run.log_artifact(log)
    print(f"[vote] MLflow tracked (run submit_{args.version}_vote); artifacts snapshotted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
