#!/usr/bin/env python3
"""Step 3 — embedding L2/PCA + linear heads (plan Phase 2a), reuse-first.

A SEPARATE experiment from the raw-embedding bake-off (probe_fusion --use-split),
which it leaves untouched. Per cached FM embedding set, under the existing split
protocol (FIT/TUNE/TEST from val_split.npy), fit a leakage-safe pipeline on FIT:

    L2-normalize -> StandardScaler -> PCA{64,128} -> {logistic regression, ridge}

and score on TUNE (selection) + TEST (lock): macro-F1, per-class (incl. Train/
Subway), selection-lock gap. Logged via shl2026.track; appended to EMB_PCA_RESULTS.md.
Cheap (linear heads on reduced dims) and a sanctioned representation the MoE/fusion
can reuse later. No FM re-inference — reads cached embeddings only.

Reuses: probe_fusion.{load_emb,load_labels}, split.load_split_with_location_map,
shl2026.{track,evaluate_predictions}.

Usage:
    python emb_pca_heads.py --emb mantisv2_V1 --pcas 64,128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_emb, load_labels  # noqa: E402
from split import load_split_with_location_map  # noqa: E402

FIT, TUNE, TEST = 0, 1, 2


def make_head(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    if name == "ridge":
        return RidgeClassifier(alpha=1.0, class_weight="balanced")
    raise ValueError(name)


def make_pipe(pca_k: int | None, head: str, n_features: int) -> Pipeline:
    steps = [("l2", Normalizer(norm="l2")), ("sc", StandardScaler())]
    if pca_k:
        k = min(pca_k, n_features)                 # clamp to available dims
        steps.append(("pca", PCA(n_components=k, random_state=0)))
    steps.append(("clf", make_head(head)))
    return Pipeline(steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--emb", required=True, help="<model>_<variant>, e.g. mantisv2_V1")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split.npy")
    ap.add_argument("--pcas", default="64,128", help="comma list; 0 = no PCA (full dim)")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol/EMB_PCA_RESULTS.md")
    args = ap.parse_args()
    t0 = time.time()

    from shl2026 import track, evaluate_predictions

    Etr = load_emb(args.emb_root / args.emb, "train")
    Eva = load_emb(args.emb_root / args.emb, "validation")
    ytr, _ = load_labels(args.feat_dir, "train")
    yva, _ = load_labels(args.feat_dir, "validation")
    assign, _ = load_split_with_location_map(args.split, args.feat_dir)
    assert len(assign) == len(yva) == len(Eva), "split/val length mismatch"
    fm, tm, sm = assign == FIT, assign == TUNE, assign == TEST

    Xfit = np.concatenate([Etr, Eva[fm]]); yfit = np.concatenate([ytr, yva[fm]])
    Xtune, ytune = Eva[tm], yva[tm]
    Xtest, ytest = Eva[sm], yva[sm]
    print(f"[emb-pca] {args.emb} dim={Etr.shape[1]} | fit={len(yfit)} tune={len(ytune)} "
          f"test={len(ytest)}", flush=True)

    pcas = [int(x) for x in args.pcas.split(",")]
    rows = []
    for pca_k in pcas:
        for head in ("logreg", "ridge"):
            pipe = make_pipe(pca_k or None, head, Etr.shape[1])
            pipe.fit(Xfit, yfit)
            r_te = evaluate_predictions(ytest, pipe.predict(Xtest))
            r_tu = evaluate_predictions(ytune, pipe.predict(Xtune))
            gap = r_tu.macro_f1 - r_te.macro_f1
            tag = f"{head}_pca{pca_k or 'full'}"
            rows.append((tag, r_te.macro_f1, r_tu.macro_f1, gap,
                         r_te.per_class_f1[7], r_te.per_class_f1[8]))
            print(f"  {tag:16s} TEST={r_te.macro_f1:.4f} TUNE={r_tu.macro_f1:.4f} "
                  f"gap={gap:+.4f} Train={r_te.per_class_f1[7]:.3f} "
                  f"Subway={r_te.per_class_f1[8]:.3f}", flush=True)
            with track("mdaniol", run_name=f"embpca_{args.emb}_{tag}", seed=0,
                       params_path=None,
                       params={"emb": args.emb, "pca": pca_k, "head": head, "dim": int(Etr.shape[1])},
                       tags={"phase": "emb", "branch": "fm_linear"}) as run:
                run.log_eval(r_te, prefix="test_")
                run.log_eval(r_tu, prefix="tune_")
                run.log_metrics({"selection_lock_gap": gap})

    hdr = f"\n## {args.emb} (dim {Etr.shape[1]})  — L2+StandardScaler+PCA -> linear head\n\n" \
          "| head | TEST macro-F1 | TUNE macro-F1 | sel-lock gap | Train F1 | Subway F1 |\n" \
          "|---|---|---|---|---|---|\n"
    body = "".join(f"| {t} | {te:.4f} | {tu:.4f} | {g:+.4f} | {tr:.3f} | {sb:.3f} |\n"
                   for t, te, tu, g, tr, sb in rows)
    head_doc = "# Step 3 — embedding L2/PCA linear heads (lock-test=TEST, selection=TUNE)\n"
    prev = args.out.read_text() if args.out.exists() else head_doc
    args.out.write_text((prev if prev.startswith("# Step 3") else head_doc) + hdr + body)
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
