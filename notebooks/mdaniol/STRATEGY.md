# SHL 2026 — Master Strategy

**Date**: 2026-06-17 · Predictions due **30.06** (~13 days) · Paper **04.07**.
Consolidates: feature design, model selection, preprocessing, and visual-branch
research (all in `feature_extraction/*.md`). Two objectives: **leaderboard macro-F1**
AND **paper novelty** (challenge explicitly rewards new/underutilized frozen FMs).

## LOCKED DECISIONS (2026-06-20, after SHL-2025 precedent analysis — see `SHL_2025_LESSONS.md`)
1. **Vision branch elevated to co-primary** — frozen ViT on IMU→RGB image (HELP-style) was the best *frozen-legal* 2025 result (81.4%).
2. **Expand FM bake-off** — add **BIOT, CBraMod** (biosignal transformers the 2025 winner used on IMU) to MantisV2/UTICA/UniMTS/LIMU-BERT/NormWear; test more as time allows.
3. **MVP-first** — Day-1 handcrafted + LightGBM → valid submission + first macro-F1; then vision + FM branches in parallel.
4. **Paper novelty hook = (b)** cross-modal frozen ensemble (TS-FM + IMU-image-ViT + handcrafted) with a **trainable fusion adapter**.
Also: **nonlinear heads** (LightGBM/RF/MLP), not linear probe (frozen-MOMENT ablation: RF 0.378 ≫ LR 0.241); **drop LLM/sensor-language** path (failed in 2025); optimize for **novelty + paper** as much as F1; realistic frozen target **≈81%** (beat 80.3% baseline).

## Architecture: 3 branches → fusion head
Everything is **frozen FM + lightweight trainable head** (rule-compliant). The
branches are parallel; they meet at one trainable head. Evidence (TiViT, TimesFM-
fusion, SHL-2025 UT-IR) is unanimous: **fusion beats any single branch.**

```
raw 9×500 window
 ├─ A. Handcrafted features (520, validated) ───────────────┐
 ├─ B. Frozen TS/IMU-FM embeddings (cached) ────────────────┤→ standardize → concat → MLP head → 8 classes
 └─ C. Frozen vision-FM on spectrograms (secondary) ────────┘     (+ optional stacking ensemble)
```

## Branch A — Handcrafted features (the safety net & likely strongest single branch)
- 520 features, independently validated, SLURM-ready (`feature_extraction/`).
- **Won SHL 2024** in this style (handcrafted + tree ML). Do **not** under-invest.
- Head: **LightGBM** (Track A standalone) + the standardized vector into the fusion head.

## Branch B — Frozen time-series / IMU foundation models (the novelty core)
Evaluate these (weights verified, `fm_input.py` packers ready); pick by val macro-F1:
| Model | Why | Note |
|---|---|---|
| **MantisV2** (2602.17868) | top pick: novel 2026, **9-ch incl. Mag**, HAR evidence, MIT weights | channel-independent |
| **UTICA** (2603.01348) | most novel paradigm (DINOv2-style self-distillation) | Mantis backbone |
| **UniMTS** (2410.19818) | IMU/motion-native, built for location/orientation shift | **6-ch, no Mag** |
| **LIMU-BERT** | IMU-native, **uses magnetometer** | bespoke scaling, 20 Hz |
| **NormWear** (2412.09758) | **cross-channel attention** (models Acc↔Gyr↔Mag) | physiology-pretrained |
| MOMENT | baseline only (used 2025) | not novel |
- ❌ **RelCon** disqualified (no public weights, accel-only).
- Process embeddings: pool → standardize (train stats) → optional PCA → fuse. (Allowed; only backbone frozen.)

## Branch C — Frozen vision FM on IMU spectrograms (secondary, novel)
- **Spectrogram (per-channel + RGB-grouped) → frozen OpenCLIP ViT (or DINOv2) → trainable MLP head.**
- Evidence: TiViT (frozen ViT beats MOMENT/Mantis on UCR; **fusion best**), LVMs useful for TSC, spectrograms proven on SHL.
- **Complementary, not dominant** → build *after* A+B, fuse in. Match encoder normalization (not ImageNet); budget a trainable adapter for the domain gap.

## Priority & timeline (decisive)
| Days | Step | Output |
|---|---|---|
| **1–2** | Run feature extraction (all 9 files) → **LightGBM (Track A)** | first **val macro-F1** + **valid submission file** (de-risks deadline) |
| **3–5** | `extract_embeddings.py` bake-off: MantisV2, UTICA, UniMTS, LIMU-BERT, NormWear → probe → **fuse with features** | best FM(s); fused macro-F1 |
| **6–8** | Branch C: spectrogram → frozen OpenCLIP/DINOv2 → fuse **if it adds** | visual branch decision |
| **9–11** | Ensemble + ablations (input variants, z-norm, channel-combiner) → freeze best | final config |
| **12–13** | Predict test, format 92 726×500, sanity-check, **submit (buffer before 30.06)** | submission |
| **14+** | Paper (3–6 pp) from logged runs | HASCA paper |

**Rule:** A valid submission exists by Day 2; everything after only improves it.

## The novelty story (for the paper)
**A cross-modal frozen-FM ensemble for transportation-mode recognition**: novel 2026
TS-classification FMs (**MantisV2, UTICA**) + IMU-native FMs (**UniMTS, LIMU-BERT**) +
**frozen vision FM on spectrograms** (OpenCLIP/DINOv2) — *interfaced* via principled
per-model input contracts (`fm_input.py`) and *fused* with physics/handcrafted
features at a lightweight head. None of the prior SHL editions combined IMU-native +
vision frozen FMs this way; the multi-view fusion (shown to win by TiViT) is itself
the contribution. All backbones frozen; only heads/adapters trained.

## Risks & hedges
- **Frozen < fine-tuned** (Mantis says so) → head + Track A carry load; that's why A is co-equal.
- **No SHL-transport macro-F1 published for any FM** → all picks are priors; **bake-off decides**.
- **TSFM spectral-shift failure** (2511.05619) → vision + handcrafted branches hedge it.
- **Vision domain gap** → fuse (don't rely standalone); trainable adapter; match normalization.
- **Over-planning is the current risk** → execute Days 1–2 now.

## Decision on visual models (your question)
**Yes, include a visual branch — at priority 3 (after handcrafted + TS/IMU FMs), as a fused complement, using spectrograms → frozen OpenCLIP/DINOv2.** It is novel for SHL and evidence says fusion helps; it is *not* a replacement and should not delay the Day-1–2 baseline.

## References
`feature_extraction/`: FEATURES_TABLE.md, DESIGN_DECISIONS.md, RE_VALIDATION_2026.md,
MODEL_SELECTION_2026.md, VISUAL_STRATEGY.md, PREPROCESSING_PLAN.md, MODEL_SHORTLIST.md.
</content>
