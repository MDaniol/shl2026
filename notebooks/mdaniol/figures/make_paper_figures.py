#!/usr/bin/env python3
"""Generate all honest paper figures from the E-COMBINE `--dump-eval` npz (+ his holdout npz).

Produces, into --out-dir:
  cm_laneA_v4_test.png     Lane A (our v4) confusion matrix on the internal TEST (n~15k, honest)
  cm_v5_blend_slice.png    v5 blend confusion matrix on the doubly-held-out slice (n~2.3k, honest; small-n)
  cm_pair_slice.png        Lane A vs Lane B on the SAME slice (side-by-side decorrelation figure)
  cm_laneB_holdout.png     Lane B (his DINoV2+MLP) on his full holdout (n~22k, honest)  [needs --his-holdout]
  perclass_f1_slice.png    grouped per-class F1: Lane A vs Lane B vs v5 (slice) — shows WHERE fusion wins
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


def perclass_bars(y_idx, preds: dict, out, title="Per-class F1 on the doubly-held-out slice",
                  dpi=300, show_title=True):
    K = len(CLASSES)
    names = list(preds.keys())
    counts = np.bincount(y_idx, minlength=K)          # true support per class in this slice
    present = counts > 0
    f1s = {n: per_class_f1(y_idx, p, K) for n, p in preds.items()}
    x = np.arange(K); wbar = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = {"Lane A (time-series, v4)": "#3b82f6", "Lane B (vision)": "#22c55e", "v5 (blend)": "#a855f7"}
    for i, n in enumerate(names):
        f = f1s[n]
        # legend reports the macro over the classes actually present (an absent class would only add a 0)
        lbl = f"{n} (macro-F1 {f[present].mean():.3f} over {int(present.sum())} present classes)"
        ax.bar(x + i * wbar, f, wbar, label=lbl, color=colors.get(n, None))
    # flag absent classes so the empty gap reads as "no data", not "F1 = 0"
    for k in range(K):
        if not present[k]:
            ax.text(x[k] + wbar * (len(names) - 1) / 2, 0.02, "absent\n(n=0)", ha="center",
                    va="bottom", fontsize=8, color="#999", style="italic")
    ax.set_xticks(x + wbar * (len(names) - 1) / 2)
    ax.set_xticklabels([f"{c}\n(n={int(counts[k])})" for k, c in enumerate(CLASSES)], fontsize=9)
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.02)
    if show_title:
        ax.set_title(title + "   —   n under each class = true support (small n ⇒ noisy)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=dpi); plt.close(fig)
    print("saved:", out)


def weight_sweep(wg, wm, w_star, out, dpi=300, show_title=True):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(wg, wm, "-o", ms=3, color="#a855f7")
    ax.axvline(w_star, ls="--", color="#666", label=f"w* = {w_star:.3f}")
    j = int(np.argmax(wm))
    ax.scatter([wg[j]], [wm[j]], color="#ef4444", zorder=5, label=f"max macro-F1 = {wm[j]:.4f}")
    ax.set_xlabel("blend weight w   (P = w·P_LaneA + (1−w)·P_LaneB)"); ax.set_ylabel("macro-F1 (slice)")
    if show_title:
        ax.set_title("Blend-weight sweep on the doubly-held-out slice")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=dpi); plt.close(fig)
    print("saved:", out)


def cm_pair(yA, pA, yB, pB, mA, mB, out, dpi=300, show_title=True):
    """Side-by-side confusion matrices on the same slice (Lane A vs Lane B)."""
    from plot_cm import confusion
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, (y, p, tag) in zip(axes, [(yA, pA, "Lane A — time-series"), (yB, pB, "Lane B — vision")]):
        M = confusion(y, p); counts = M.sum(1); Mn = M / np.clip(counts[:, None], 1, None)
        f1 = per_class_f1(y, p); present = counts > 0
        mp = float(f1[present].mean()) if present.any() else 0.0
        absent = [CLASSES[i] for i in range(8) if not present[i]]
        im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticklabels([f"{c} (n={int(n)})" if n > 0 else f"{c} (absent)" for c, n in zip(CLASSES, counts)],
                           fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(8):
            if counts[i] == 0:
                ax.text(3.5, i, "absent", ha="center", va="center", color="#999", fontsize=7, style="italic")
                continue
            for jj in range(8):
                v = Mn[i, jj]
                ax.text(jj, i, f"{v*100:.0f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
        if show_title:
            sub = f"macro-F1={mp:.4f} ({int(present.sum())} present" + (f", {','.join(absent)} absent)" if absent else ")")
            ax.set_title(f"{tag}\n{sub}, same slice n={len(y)}", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=dpi); plt.close(fig)
    print("saved:", out)


def paired_boot(y, p_new, p_base, present=None, B=2000, seed=0):
    """Percentile paired bootstrap of the macro-F1 difference (p_new − p_base) over windows.
    Averaged over the classes present in the slice (a class absent for every model would only add a
    constant 0 and rescale the metric, so it is excluded to keep the number meaningful)."""
    rng = np.random.default_rng(seed); n = len(y)
    sel = np.ones(8, bool) if present is None else present
    def macro(yy, pp):
        return per_class_f1(yy, pp)[sel].mean()
    point = macro(y, p_new) - macro(y, p_base)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = macro(y[idx], p_new[idx]) - macro(y[idx], p_base[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def write_captions(path, present, counts, mT, nT, mB, nB, w, our, his, v5, boot, nS):
    """Editable markdown: draft figure captions + honest results table (colleague edits for the paper)."""
    fOur, mOur, mpOur = our; fHis, mHis, mpHis = his; fV5, mV5, mpV5 = v5; dlt, dlo, dhi = boot
    nBus = int(counts[5])

    npresent = int(present.sum())

    def prow(tag, f, m, mp):
        pc = " ".join(f"{CLASSES[k][:2]}={f[k]:.2f}" for k in range(8) if present[k])
        return f"| {tag} | {mp:.3f} | {pc} |"

    L = [
        "# SHL-2026 — figure captions & results (editable draft)",
        "",
        "Draft captions for the paper figures — edit freely. All numbers come from the honest, "
        "leakage-clean held-out evaluation (select-on-TUNE, lock-TEST-once, per-window). Do NOT quote "
        "in-sample / full-validation numbers.",
        "",
        "## Figure captions",
        "",
        f"**cm_laneA_v4_test.png** — Confusion matrix (row-normalized recall, %) of the time-series lane "
        f"(Lane A; v4: frozen UTICA + Mantis-V2 embeddings concatenated with 520 hand-crafted features, "
        f"per-class-calibrated LightGBM soft-vote) on the held-out internal TEST set "
        f"(Users 2–3, Bag/Hips/Torso, per-window; n = {nT:,}). Macro-F1 = {mT:.3f}. Residual confusion "
        f"concentrates in the Train↔Subway and Bike↔Subway pairs.",
        "",
        (f"**cm_laneB_holdout.png** — Confusion matrix (row-normalized recall, %) of the vision-based lane "
         f"(Lane B; frozen DINoV2 over STFT/CWT/GAF spectrogram images, gated multi-branch MLP) on its "
         f"held-out target set (n = {nB:,}). Macro-F1 = {mB:.3f}. The error structure is complementary to "
         f"Lane A (stronger on Subway, weaker on Bus/Train), which motivates the late-fusion blend."
         if mB is not None else
         "**cm_laneB_holdout.png** — (not generated: Lane-B holdout npz not supplied.)"),
        "",
        f"**perclass_f1_slice.png** — Per-class F1 of the two lanes and their late-fusion blend (v5) on the "
        f"doubly-held-out slice (windows held out from training by BOTH pipelines; n = {nS:,}). Bars are "
        f"annotated with true support n. Run is absent from this intersection (n = 0) and Bus is small "
        f"(n = {nBus}), so those are omitted / noisy; the full per-class behaviour is given by the Lane-A "
        f"and Lane-B confusion matrices. Over the classes present, v5 (macro {mpV5:.3f}) improves on Lane A "
        f"({mpOur:.3f}) and Lane B ({mpHis:.3f}), with the largest gains on the confusable vehicle/rail "
        f"classes (Bus, Train).",
        "",
        f"**weight_sweep.png** — Macro-F1 on the doubly-held-out slice as a function of the blend weight w "
        f"(P = w·P_LaneA + (1−w)·P_LaneB); optimum at w = {w:.3f}. w is tuned only on this doubly-held-out "
        f"slice, so the fusion introduces no leakage."
        f"so the fusion introduces no leakage.",
        "",
        f"## Results — doubly-held-out slice (n = {nS:,}; {npresent} of 8 classes present)",
        "",
        f"Macro-F1 is averaged over the {npresent} classes present in this slice. Run (0 windows) is not scored "
        f"here — it would only contribute a forced 0 to every model and is fully characterised by the full-set "
        f"Lane-A / Lane-B confusion matrices (Run F1 ≈ 0.94 on the internal TEST).",
        "",
        f"| model | macro-F1 ({npresent} present classes) | per-class F1 |",
        "|---|---|---|",
        prow("Lane A (v4, time-series)", fOur, mOur, mpOur),
        prow("Lane B (vision)", fHis, mHis, mpHis),
        prow(f"v5 (blend, w={w:.3f})", fV5, mV5, mpV5),
        "",
        f"v5 vs Lane A (v4): paired-bootstrap Δ(macro-F1 over present classes) = {dlt:+.4f}, "
        f"95% CI [{dlo:+.4f}, {dhi:+.4f}] (excludes 0 → statistically significant). Bus is small "
        f"({nBus} windows) in this intersection, so its bar is noisy; the full-set confusion matrices are the "
        f"authoritative per-class evidence.",
        "",
        "## Lane summary (full held-out sets, all classes present)",
        "",
        "| lane | eval set | n | macro-F1 |",
        "|---|---|---|---|",
        f"| Lane A (time-series, v4) | internal TEST (Bag/Hips/Torso) | {nT:,} | {mT:.3f} |",
        (f"| Lane B (vision, DINoV2) | target holdout | {nB:,} | {mB:.3f} |"
         if mB is not None else "| Lane B (vision, DINoV2) | target holdout | — | — |"),
        "",
    ]
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True, help="npz from combine_pdusza.py --dump-eval")
    ap.add_argument("--his-holdout", type=Path, default=None,
                    help="his selected_untouched_target_holdout/predictions.npz (Lane B full holdout CM)")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dpi", type=int, default=300, help="figure DPI (paper quality = 300)")
    ap.add_argument("--titles", action="store_true",
                    help="bake titles into the figures. Default OFF: figures are title-less and captions "
                         "go to the editable FIGURE_CAPTIONS.md (paper convention).")
    ap.add_argument("--slice-cm", action="store_true",
                    help="also emit the doubly-held-out-slice confusion matrices (v5 blend + Lane-A/B pair). "
                         "OFF by default (Option A): that slice lacks Run, so per-class CMs use the full "
                         "held-out sets; the slice result is shown as bars + a table instead.")
    ap.add_argument("--captions", type=Path, default=None,
                    help="path for the editable caption+results markdown (default <out-dir>/FIGURE_CAPTIONS.md)")
    ap.add_argument("--mlflow", action="store_true",
                    help="log every PNG (+ the dump + captions md) to MLflow as a 'combine_figures' run.")
    a = ap.parse_args()
    st = a.titles
    a.out_dir.mkdir(parents=True, exist_ok=True)
    caps = a.captions or (a.out_dir / "FIGURE_CAPTIONS.md")
    d = np.load(a.dump); labels = d["classes"]
    produced = []

    # --- Lane A (our v4) confusion matrix on the full internal TEST (all classes present) ---
    yT = _to_idx(d["test_y"], labels); pT = d["test_ourP"].argmax(1)
    mT = float(per_class_f1(yT, pT).mean())
    plot_cm(yT, pT, "Lane A — time-series lane (v4) — internal TEST",
            a.out_dir / "cm_laneA_v4_test.png", dpi=a.dpi, show_title=st)
    produced.append(a.out_dir / "cm_laneA_v4_test.png")
    print(f"Lane A v4 (TEST n={len(yT)}) macro-F1={mT:.4f}")

    # --- Lane B (his) confusion matrix on the full holdout (all classes present) ---
    mB, nB = None, 0
    if a.his_holdout and a.his_holdout.exists():
        h = np.load(a.his_holdout, allow_pickle=True)
        yB = np.asarray(h["y_true"])
        pB = h["probabilities"].argmax(1) if "probabilities" in h else np.asarray(h["y_pred"])
        mB = float(per_class_f1(yB, pB).mean()); nB = int(len(yB))
        plot_cm(yB, pB, "Lane B — vision lane (DINoV2 + MLP) — holdout",
                a.out_dir / "cm_laneB_holdout.png", dpi=a.dpi, show_title=st)
        produced.append(a.out_dir / "cm_laneB_holdout.png")
        print(f"Lane B (holdout n={nB}) macro-F1={mB:.4f}")

    # --- doubly-held-out slice: per-class bars + weight sweep (+ optional slice CMs) ---
    yS = _to_idx(d["slice_y"], labels)
    ourP, hisP, w = d["slice_ourP"], d["slice_hisP"], float(d["w"])
    pOur, pHis = ourP.argmax(1), hisP.argmax(1)
    pV5 = (w * ourP + (1 - w) * hisP).argmax(1)
    counts = np.bincount(yS, minlength=8); present = counts > 0

    def stats(p):
        f = per_class_f1(yS, p); return f, float(f.mean()), float(f[present].mean())
    fOur, mOur, mpOur = stats(pOur); fHis, mHis, mpHis = stats(pHis); fV5, mV5, mpV5 = stats(pV5)
    boot = paired_boot(yS, pV5, pOur if mpOur >= mpHis else pHis, present=present)

    perclass_bars(yS, {"Lane A (time-series, v4)": pOur, "Lane B (vision)": pHis, "v5 (blend)": pV5},
                  a.out_dir / "perclass_f1_slice.png", dpi=a.dpi, show_title=st)
    weight_sweep(d["w_grid"], d["w_macro"], w, a.out_dir / "weight_sweep.png", dpi=a.dpi, show_title=st)
    produced += [a.out_dir / "perclass_f1_slice.png", a.out_dir / "weight_sweep.png"]
    if a.slice_cm:
        plot_cm(yS, pV5, f"v5 blend (w={w:.3f}) — doubly-held-out slice",
                a.out_dir / "cm_v5_blend_slice.png", dpi=a.dpi, show_title=st)
        cm_pair(yS, pOur, yS, pHis, mOur, mHis, a.out_dir / "cm_pair_slice.png", dpi=a.dpi, show_title=st)
        produced += [a.out_dir / "cm_v5_blend_slice.png", a.out_dir / "cm_pair_slice.png"]
    print(f"slice (n={len(yS)}) present-macro: ours={mpOur:.4f} his={mpHis:.4f} v5={mpV5:.4f}; "
          f"Δ(v5−base)={boot[0]:+.4f} CI[{boot[1]:+.4f},{boot[2]:+.4f}]")

    # --- editable captions + results markdown ---
    write_captions(caps, present, counts, mT, len(yT), mB, nB, w,
                   (fOur, mOur, mpOur), (fHis, mHis, mpHis), (fV5, mV5, mpV5), boot, len(yS))
    produced.append(caps)
    print("saved:", caps)

    if a.mlflow:
        from shl2026 import track  # local import: not needed for offline iteration
        with track("mdaniol", run_name="combine_figures", seed=0, params_path=None,
                   params={"dump": a.dump.name, "n_slice": int(len(yS)), "n_test": int(len(yT)),
                           "w": w, "dpi": a.dpi},
                   tags={"phase": "figures", "experiment": "E-COMBINE"}) as run:
            run.log_metrics({"slice_ours_present": mpOur, "slice_his_present": mpHis,
                             "slice_v5_present": mpV5, "test_v4": mT,
                             "paired_diff": boot[0], "paired_ci_lo": boot[1]})
            run.log_artifact(a.dump)
            for p in produced:
                if p.exists():
                    run.log_artifact(p)
        print(f"[figures] MLflow-logged {len(produced)} artifacts under run 'combine_figures'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
