# HASCA/SHL 2026 — Doable 1-Week Plan (revised)

**Status**: Actionable
**Supersedes**: `hasca_2026_challenge_plan.md` (Mistral Vibe draft) — keep that for its literature scan; this file is what we build.
**Date**: 2026-06-15
**Time left**: predictions due **30.06.2026** (~15 days, target a frozen submission by **28.06** for buffer), HASCA paper due **04.07.2026**.

---

## 1. Ground truth (verified against `dataset_parquet/` + challenge rules)

| Fact | Value | Consequence for design |
|------|-------|------------------------|
| Channels | **9**: Acc/Gyr/**Mag** × x,y,z | Use all 9. No GPS exists. |
| Window | **500 samples @ 100 Hz** (5 s) | Fixed; no resampling needed. |
| Train | user **1 only**, 4 locations (Bag/Hips/Torso/Hand), 196 072 windows/loc | **Single user → no user-DANN possible.** |
| Validation | users **2 & 3**, 4 locations, 28 789 windows/loc | This *is* our user-independent dev signal. Use it as the model-selection oracle. |
| Test | 92 726 windows, 9 ch, **no labels, merged locations, frames shuffled** | Window-internal features only; **no cross-window smoothing**. |
| Labels | per-frame 1–8; **constant within window** (0.24% mixed) | Predict **one label/window**, replicate ×500. |
| Imbalance | class 3 (Run) ≈ **4.3%**, others 12–16% | **Class-weighted loss / sampling**; report **macro-F1**, not accuracy. |
| Metric | **macro-F1** (SHL standard) | Select, early-stop, and tune on macro-F1. |
| Submission | `teamName_predictions.txt`, **92 726 × 500** matrix | Lock the writer + shape-check on day 1. |
| Rule | FMs **frozen**; only lightweight heads trainable | Cache embeddings once; train heads on top. |

**The real domain shift to defend against is phone *location* (4 train → 3 test, no Hand), not user.** Drop all cross-user adversarial machinery (DANN-over-users, DGDATA, EEG-ADG) — it cannot work with one training user.

---

## 2. Strategy: one frozen FM, done well + a classical safety net

Cut the 4-branch (TS+Vision+Language+Physics) X-Fi fusion. In one week it won't converge or be ablatable. Instead:

- **Track A — Safety net (must-have, Day 1):** handcrafted/physics features → LightGBM/RF. Guarantees a *valid, decent* submission before any FM work. Historically competitive on SHL and robust to location shift.
- **Track B — Primary FM (the contribution):** **MOMENT** (frozen, motion-pretrained, native embedding mode) over the 9 channels → cached embeddings → lightweight head. MOMENT is the right FM here; Chronos is a *forecaster* and does not cleanly emit a classification embedding.
- **Track C — Stretch (only if A+B are frozen and submitted):** spectrogram → frozen vision FM (DINOv2 or CLIP-ViT) as a *second* embedding; late-fuse with B. This is the honest novelty: **frozen-FM embeddings + physics fusion, single-user→multi-user, location-robust.**

Drop for now (paper "future work"): SensorLLM, X-Fi cross-modal attention, generative DA, graph nets.

---

## 3. One-week schedule

**Day 1 — Pipeline + safety-net submission.**
Loader over `dataset_parquet`; handcrafted features per window per channel (mean/std/energy, gravity-removed acc norm, jerk, dominant freq + spectral entropy via FFT, correlation across axes — all window-internal). Train LightGBM on pooled 4-location train, **validate on users 2&3**, optimise **macro-F1** with class weights. Wire `submit.py` → 92 726×500 writer + shape/format assertions. **Produce a real submission file today.**

**Day 2 — Cache frozen FM embeddings.**
Run MOMENT (frozen, bf16, batched) over train/val/test. Per-channel univariate encode → concat (or MOMENT multivariate) → store embeddings to the shared cache. This is the expensive step; do it once. Log to MLflow.

**Day 3 — Lightweight head on FM embeddings.**
MLP (LayerNorm + dropout 0.3) → 8 classes, AdamW, class-weighted CE, early-stop on **val macro-F1**. Compare against Track A.

**Day 4 — Fusion + robustness.**
Concatenate physics features to FM embedding (cheapest big win). Per-**location** val breakdown (esp. Hand vs the 3 test locations); confusion matrix; address Run + the still/vehicle confusions. Decide single-model vs A+B ensemble.

**Day 5 — Stretch (Track C) or harden.**
If A+B solid: add spectrogram+vision FM embedding and late-fuse. Else: calibration, seed averaging, finalise config.

**Day 6 — Freeze & submit.**
Regenerate test predictions from the frozen best config, sanity-check class distribution vs train prior, write matrix, **submit (buffer before 30.06).**

**Day 7+ — Paper.**
3–6 pp ACM sigconf from MLflow-logged runs: method, the single-user/location-shift framing, ablations (physics on/off, FM vs classical, fusion), error analysis.

---

## 4. Pitfalls (this challenge specifically)

- **Global normalisation only** (fit scaler on train, apply to val/test) — never per-user/per-window-leaking stats.
- **Macro-F1 everywhere** — accuracy will lie because of Run (4.3%).
- **No cross-window post-processing** — test is shuffled; any smoothing is illegal and won't generalise.
- **Use all 9 channels** including Mag (the draft dropped it).
- **Keep heads genuinely lightweight** — frozen FM, no backbone grads, to stay rule-compliant.
- **Validate on users 2&3 from the start** — it's the only honest proxy for the test distribution.
- Route every run through the existing **Parquet → MLflow → embedding-cache** stack for reproducibility (FAIR/Zenodo target).

---

## 5. Minimum viable outcome vs stretch

- **MVP (by Day 3–4):** valid submission + FM-head beating the classical baseline on user-independent macro-F1.
- **Target:** A+B fusion, location-robust, clean ablations.
- **Stretch:** + vision-FM branch (Track C) for a stronger novelty claim.

Honest accuracy/F1 targets are **unknown** under the new frozen-FM constraint — do **not** carry the draft's ">92%". Report what the val set gives.
</content>
</invoke>
