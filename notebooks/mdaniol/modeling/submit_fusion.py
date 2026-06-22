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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_feats, load_emb, load_labels  # noqa: E402

N_TEST = 92726
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
    ap.add_argument("--n-estimators", type=int, default=600,
                    help="Phase-B has no early-stop set; 600 ~ typical early-stop landing.")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--team", default="AGH")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from shl2026 import write_submission

    cfg = f"moment-fusion({args.emb})"
    out = args.out or (root / "notebooks/mdaniol" /
                       f"{args.team}_predictions_{args.version}_moment-fusion.txt")

    # --- Phase B: train on ALL labelled data (User-1 train + ALL validation) ----
    Xtr = np.concatenate([fuse(args.feat_dir, args.emb_root, args.emb, "train"),
                          fuse(args.feat_dir, args.emb_root, args.emb, "validation")])
    ytr = np.concatenate([load_labels(args.feat_dir, "train")[0],
                          load_labels(args.feat_dir, "validation")[0]])
    print(f"[submit] train X={Xtr.shape} (User-1 + ALL validation)", flush=True)
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=args.n_estimators,
                             learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, class_weight="balanced", n_jobs=-1,
                             verbosity=-1, random_state=0)
    clf.fit(Xtr, ytr)

    # --- predict the unlabelled test -------------------------------------------
    Xte = fuse(args.feat_dir, args.emb_root, args.emb, "test")
    assert len(Xte) == N_TEST, f"expected {N_TEST} test frames, got {len(Xte)}"
    pred = clf.predict(Xte).astype(int)
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
        f"| ~0.803 (temporal; bracket 0.725-0.803) | Phase-B all-val; argmax; det |\n"))
    print(f"[submit] logged -> {log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
