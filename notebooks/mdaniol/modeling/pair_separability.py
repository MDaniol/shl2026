#!/usr/bin/env python3
"""A1 — pair-separability DIAGNOSTIC for the two macro-F1 caps (Train↔Subway, Car↔Bus).

Question it answers (decides whether the hard-pair program B1/B2 is worth building): *can a linear
probe on the SUBMISSION representation separate each confusable pair on HELD-OUT data?* If yes, the
separating information is in the frozen rep and the multiclass head is leaving it on the table
(→ build the Fisher feature B1 / the OvO specialist B2). If no, the encoder has collapsed the pair
and the ceiling is an information limit (→ DISABLE the hard-pair program; don't repeat the rail-expert
mistake).

Leakage-safe, mirrors the pipeline protocol EXACTLY (submit_vote/voting_head):
  - fit the binary probe ONLY on FIT = train(all) + validation[FIT];
  - evaluate on the locked TEST = validation[TEST] (BHT-only by split construction);
  - per-window, deterministic; never selects on TUNE/TEST.
Representation configurable: `fusion` (emb ⊕ 520 handcrafted — the submission rep, default), `emb`
(frozen FM only), `handcrafted` (520 only — the no-FM control).

Reuses probe_fusion.{load_emb,load_feats,load_labels,split_locs}, split.load_split_with_location_map,
shl2026.track. Pure sklearn LogisticRegression probe (standardized, class-balanced).

    python pair_separability.py --emb utica_V2 --rep fusion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import load_emb, load_feats, load_labels  # noqa: E402
from split import load_split_with_location_map  # noqa: E402

FIT, TUNE, TEST = 0, 1, 2
PAIRS = {"Train_vs_Subway": (7, 8), "Car_vs_Bus": (5, 6)}
NAME = {1: "Still", 2: "Walk", 3: "Run", 4: "Bike", 5: "Car", 6: "Bus", 7: "Train", 8: "Subway"}


def build_rep(feat_dir: Path, emb_root: Path, emb: str, split: str, rep: str) -> np.ndarray:
    """Submission-style features for a split: fusion = emb ⊕ 520 handcrafted (default)."""
    feats = load_feats(feat_dir, split)[0] if rep != "emb" else None
    e = load_emb(emb_root / emb, split) if rep != "handcrafted" else None
    if rep == "fusion":
        return np.concatenate([e, feats], axis=1)
    return e if rep == "emb" else feats


def probe_pair(Xfit, yfit, Xtest, ytest, pos, neg):
    """Binary linear probe (standardized, balanced LogisticRegression) for {pos,neg}: fit on FIT,
    score on TEST. Returns held-out balanced-accuracy + F1(pos) + support — the separability signal."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, f1_score

    mf, mt = np.isin(yfit, [pos, neg]), np.isin(ytest, [pos, neg])
    Xf, yf = Xfit[mf], (yfit[mf] == pos).astype(int)
    Xt, yt = Xtest[mt], (ytest[mt] == pos).astype(int)
    sc = StandardScaler().fit(Xf)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0,
                             random_state=0).fit(sc.transform(Xf), yf)
    pred = clf.predict(sc.transform(Xt))
    return {"bal_acc": float(balanced_accuracy_score(yt, pred)),
            "f1_pos": float(f1_score(yt, pred, pos_label=1)),
            "n_fit": int(mf.sum()), "n_test": int(mt.sum())}


def verdict(bal_acc: float) -> str:
    if bal_acc >= 0.80:
        return "SEPARABLE → info present; build B1/B2"
    if bal_acc <= 0.65:
        return "COLLAPSED → ceiling is an info limit; DISABLE hard-pair program"
    return "MARGINAL → decide by ROI"


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--emb", default="utica_V2", help="frozen FM whose rep we probe")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--rep", choices=["fusion", "emb", "handcrafted"], default="fusion")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol" / "PAIR_SEPARABILITY.md")
    args = ap.parse_args()
    from shl2026 import track

    # FIT = train(all) + validation[FIT]; TEST = validation[TEST]  (identical to submit_vote)
    Xtr = build_rep(args.feat_dir, args.emb_root, args.emb, "train", args.rep)
    Xva = build_rep(args.feat_dir, args.emb_root, args.emb, "validation", args.rep)
    ytr = load_labels(args.feat_dir, "train")[0]
    yva = load_labels(args.feat_dir, "validation")[0]
    assign, _ = load_split_with_location_map(args.split, args.feat_dir)
    assert len(assign) == len(yva) == len(Xva), "split/validation length mismatch"
    Xfit = np.concatenate([Xtr, Xva[assign == FIT]]); yfit = np.concatenate([ytr, yva[assign == FIT]])
    Xtest, ytest = Xva[assign == TEST], yva[assign == TEST]

    rows, lines = [], [f"# A1 pair-separability — emb={args.emb} rep={args.rep} "
                       f"(linear probe, FIT-fit / TEST-locked, BHT-only)\n",
                       "| pair | held-out bal-acc | F1(+) | n_fit | n_test | verdict |",
                       "|---|---|---|---|---|---|"]
    for name, (pos, neg) in PAIRS.items():
        r = probe_pair(Xfit, yfit, Xtest, ytest, pos, neg)
        v = verdict(r["bal_acc"])
        rows.append((name, r, v))
        lines.append(f"| {name} | {r['bal_acc']:.4f} | {r['f1_pos']:.4f} | {r['n_fit']} "
                     f"| {r['n_test']} | {v} |")
        print(f"[A1] {name}: bal_acc={r['bal_acc']:.4f} F1+={r['f1_pos']:.4f} "
              f"(n_fit={r['n_fit']} n_test={r['n_test']}) -> {v}", flush=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[A1] wrote {args.out}", flush=True)

    with track("mdaniol", run_name=f"pairsep_{args.emb}_{args.rep}", seed=0, params_path=None,
               params={"emb": args.emb, "rep": args.rep, "split": args.split.stem,
                       "protocol": "binary linear probe; FIT-fit, TEST-locked, BHT-only"},
               tags={"phase": "diagnostic", "experiment": "A1-pair-separability"}) as run:
        run.log_metrics({f"{n}_bal_acc": r["bal_acc"] for n, r, _ in rows}
                        | {f"{n}_f1_pos": r["f1_pos"] for n, r, _ in rows})
        run.log_artifact(args.out)
    print("[A1] MLflow tracked.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
