#!/usr/bin/env python3
"""Generate all honest paper figures from the E-COMBINE `--dump-eval` npz (+ his holdout npz).

Produces, into --out-dir:
  cm_laneA_v4_test.png     Lane A (our v4) confusion matrix on the internal TEST (n~15k, honest)
  cm_v5_blend_slice.png    v5 blend confusion matrix on the doubly-held-out slice (n~2.3k, honest; small-n)
  cm_pair_slice.png        Lane A vs Lane B on the SAME slice (side-by-side decorrelation figure)
  cm_laneB_holdout.png     Lane B (his DINoV2+MLP) on his full holdout (n~22k, honest)  [needs --his-holdout]
  perclass_f1_slice.png    grouped per-class F1: ours vs his vs v5 (slice) — shows WHERE fusion wins
  weight_sweep.png         macro-F1 vs blend weight w, optimum marked

All inputs are labelled held-out data — no hidden-test (unlabelled) and no inflated full-validation.
Numerics only from numpy; plotting via matplotlib (Agg). Reuses plot_cm.plot_cm.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cm import plot_cm, per_class_f1, CLASSES  # noqa: E402


def _to_idx(y, labels):
    """Map class labels (e.g. 1..8) to 0-based column indices aligned with `labels` order."""
    lab2idx = {int(l): i for i, l in enumerate(labels)}
    return np.array([lab2idx[int(v)] for v in y])


def perclass_bars(y_idx, preds: dict, out, title="Per-class F1 on the doubly-held-out slice"):
    K = len(CLASSES)
    names = list(preds.keys())
    f1s = {n: per_class_f1(y_idx, p, K) for n, p in preds.items()}
    x = np.arange(K); wbar = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    colors = {"ours (v4)": "#3b82f6", "his (Lane B)": "#22c55e", "v5 blend": "#a855f7"}
    for i, n in enumerate(names):
        ax.bar(x + i * wbar, f1s[n], wbar, label=f"{n} (macro {f1s[n].mean():.3f})",
               color=colors.get(n, None))
    ax.set_xticks(x + wbar * (len(names) - 1) / 2); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.02); ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)


def weight_sweep(wg, wm, w_star, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(wg, wm, "-o", ms=3, color="#a855f7")
    ax.axvline(w_star, ls="--", color="#666", label=f"w* = {w_star:.2f}")
    j = int(np.argmax(wm))
    ax.scatter([wg[j]], [wm[j]], color="#ef4444", zorder=5, label=f"max macro-F1 = {wm[j]:.4f}")
    ax.set_xlabel("blend weight w   (P = w·ours + (1−w)·his)"); ax.set_ylabel("macro-F1 (slice)")
    ax.set_title("Blend-weight sweep on the doubly-held-out slice")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)


def cm_pair(yA, pA, yB, pB, mA, mB, out):
    """Side-by-side confusion matrices on the same slice (Lane A vs Lane B)."""
    from plot_cm import confusion
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (y, p, tag, m) in zip(axes, [(yA, pA, "Lane A — our v4", mA), (yB, pB, "Lane B — DINoV2+MLP", mB)]):
        M = confusion(y, p); Mn = M / np.clip(M.sum(1, keepdims=True), 1, None)
        im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(8):
            for jj in range(8):
                v = Mn[i, jj]
                ax.text(jj, i, f"{v*100:.0f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
        ax.set_title(f"{tag}\nmacro-F1={m:.4f} (same slice, n={len(y)})", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True, help="npz from combine_pdusza.py --dump-eval")
    ap.add_argument("--his-holdout", type=Path, default=None,
                    help="his selected_untouched_target_holdout/predictions.npz (Lane B full holdout CM)")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(a.dump)
    labels = d["classes"]

    # --- Lane A (our v4) on the full internal TEST (honest, n~15k) ---
    yT = _to_idx(d["test_y"], labels); pT = d["test_ourP"].argmax(1)
    mT, _ = plot_cm(yT, pT, "Lane A (our v4: frozen TS-FMs + LightGBM vote) — internal TEST",
                    a.out_dir / "cm_laneA_v4_test.png")
    print(f"Lane A v4 (TEST n={len(yT)}) macro-F1={mT:.4f}")

    # --- the doubly-held-out slice: ours / his / v5 blend ---
    yS = _to_idx(d["slice_y"], labels)
    ourP, hisP, w = d["slice_ourP"], d["slice_hisP"], float(d["w"])
    pOur, pHis = ourP.argmax(1), hisP.argmax(1)
    pV5 = (w * ourP + (1 - w) * hisP).argmax(1)
    mOur = per_class_f1(yS, pOur).mean(); mHis = per_class_f1(yS, pHis).mean()
    mV5 = per_class_f1(yS, pV5).mean()
    plot_cm(yS, pV5, f"v5 blend (w={w:.2f}) — doubly-held-out slice  [small n]",
            a.out_dir / "cm_v5_blend_slice.png")
    cm_pair(yS, pOur, yS, pHis, mOur, mHis, a.out_dir / "cm_pair_slice.png")
    perclass_bars(yS, {"ours (v4)": pOur, "his (Lane B)": pHis, "v5 blend": pV5},
                  a.out_dir / "perclass_f1_slice.png")
    weight_sweep(d["w_grid"], d["w_macro"], w, a.out_dir / "weight_sweep.png")
    print(f"slice (n={len(yS)}) macro-F1: ours={mOur:.4f} his={mHis:.4f} v5={mV5:.4f} (w={w:.2f})")

    # --- Lane B on his full holdout (honest, n~22k) ---
    if a.his_holdout and a.his_holdout.exists():
        h = np.load(a.his_holdout, allow_pickle=True)
        yB = h["y_true"]; pB = h["probabilities"].argmax(1) if "probabilities" in h else h["y_pred"]
        mB, _ = plot_cm(yB, pB, "Lane B (frozen DINoV2 + gated MLP) — honest holdout",
                        a.out_dir / "cm_laneB_holdout.png")
        print(f"Lane B (holdout n={len(yB)}) macro-F1={mB:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
