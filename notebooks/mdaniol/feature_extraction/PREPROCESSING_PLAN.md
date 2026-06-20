# FM-Input Preprocessing — Design Reference (literature-backed)

Source: deep-research over 22 primary sources, 21 adversarially-verified claims
(4 refuted, listed at bottom). Question: *what preprocessing of the raw IMU signal
helps vs hurts when the FM backbone is **frozen** and pretrained on a different
distribution?* Context: SHL 2026, 9-ch Acc/Gyr/Mag, 500 samples @ 100 Hz.

## The one principle
Split preprocessing into **MANDATORY alignment** (match each FM's input contract —
get this wrong and frozen embeddings silently degrade) and **OPTIONAL domain**
preprocessing (gravity removal, orientation invariance) that is **not universally
good and must be ablated per backbone**. Normalization benefit is **model-specific**
— there is no single best scheme across frozen TS-FMs [2512.02833].

---

## A. MANDATORY per-model input contract (non-negotiable)

| Model | Channels | Units / scaling | Length / rate | Internal norm | External norm to apply |
|---|---|---|---|---|---|
| **MOMENT** (2402.03885) | 9, as **9 independent univariate** series | any (RevIN handles scale) | **T=512**; subsample if longer, **zero-pad on the LEFT** if shorter (500→pad to 512) | **RevIN** (reversible instance norm) before patching | **none** — external z-score is redundant |
| **Mantis** (2502.15637) | 9 as **independent univariate** (≤10 ch → no adapter needed) | any (internal z-score) | 512 via interpolation *(exact 500-handling: verify on the checkpoint — "must interp" was unconfirmed)* | **per-instance, per-channel z-score** in forward pass | **none** — and *avoid* external scaling (it also feeds a raw Multi-Scaled Scalar Encoder, so external scaling perturbs that branch) |
| **UniMTS** (2410.19818) | **6 only** (acc_x,y,z, gyro_x,y,z) — **DROP magnetometer** | **acc in m/s²** | **10 s** window (5 s auto-padded); pass `--original_sampling_rate 100` | — | convert acc → m/s²; supply sampling rate |
| **LIMU-BERT** (SenSys'21) | 6 or 9 | **bespoke: acc⁄9.8, mag→L2-unit×2, gyro untouched** (→ ~[-8,8]); **standard z-score HURTS** | **20 Hz / 120 samples**; resample 100→20 Hz (5 s→100 samples vs expected 120 — handle the mismatch) | — | **LIMU scaling only** (≈ +5.78% over raw; generic z-score "very low accuracy") |

**Hard takeaways**
- **Do NOT apply a generic z-score globally.** It's *redundant* for MOMENT/Mantis (internal norm) and *actively harmful* for LIMU-BERT.
- **UniMTS can't use the magnetometer** — our strongest transport cue (rail/road) is unavailable to it; weigh that against its location-shift design.
- **LIMU-BERT needs resampling to 20 Hz** and its exact scaling; the 100-vs-120 window length is an implementation detail to resolve.

---

## B. OPTIONAL domain preprocessing — build as channel variants, ABLATE

All reuse the **already-validated** `shl_features.derive_streams()` (it outputs the
time-series streams, not just features). Feed as extra/replacement **channels**
(channel-independence makes this natural for MOMENT/Mantis).

| Variant | Channels fed | Rationale | Literature | Expectation |
|---|---|---|---|---|
| **V0 raw** | 9 raw axes | baseline | — | reference |
| **V1 body-acc** | gravity-removed acc + gyr + mag | emphasize motion (UCI-HAR style) | gravity/forward frame [MDPI 23:5845] | may be **redundant** where RevIN already removes DC — ablate |
| **V2 invariant** | magnitude channels (\|acc\|,\|gyr\|,\|mag\|) + inter-vector angles | orientation/position invariance | 4 invariant feats → ~80% LOSO [PMC9313140] | likely helps cross-user/position — ablate vs raw |
| **V3 reference-frame** | gravity/forward 3×3 rotation transform | consistent frame across orientations | [MDPI 23:5845] | **partial** gains only (full-recovery claim refuted) |
| **V4 + derived** | V0/V1 **plus** jerk, mag-rate channels | extra HAR cues as channels | channel augmentation | cheap; ablate |

**Model-specific notes**
- **UniMTS** already learns rotation invariance via pretraining augmentation → input-side rotation transforms (V2/V3) likely **redundant for it** — ablate but don't expect gains.
- **MOMENT/Mantis** apply internal instance-norm → explicit **gravity removal may be partly redundant** (open question; logically implied, not directly tested) — V1 is the key ablation.
- **LIMU-BERT** has a fixed input contract → keep domain variants minimal (it expects raw-ish acc/gyro/mag in its scaling).

---

## C. Other preprocessing axes
- **Resampling/length**: per the contract table. MOMENT left-pad 500→512; Mantis interpolate; UniMTS pad to 10 s; LIMU-BERT decimate 100→20 Hz.
- **Filtering/denoising** (band-pass to motion band, detrend): **not validated** by a confirmed source for frozen FMs; risk = moving signal off the pretraining manifold. Treat as low-priority, ablate only if time.
- **Spectrogram / vision-FM (AST, DINOv2)** preprocessing (log-mel vs STFT, per-channel→RGB, ImageNet/AudioSet norm, resize): **NOT resolved** by this review (only secondary sources) — remains an open item for Track C; needs its own check before building.

---

## D. Plan of record
1. **Implement the MANDATORY contract per model exactly** (table A) — this is the correctness floor. Especially: no generic z-score; LIMU scaling; UniMTS 6-ch+m/s²; MOMENT left-pad.
2. **Build V0–V4 as channel variants** from validated `derive_streams`.
3. **Treat normalization + domain preprocessing as ablation axes in the Day-2 bake-off** — pick per (model × variant) by **val macro-F1** on users 2&3, with per-location breakdown. No fixed choice; the data decides [2512.02833].
4. **Manage expectations**: frozen use is *suboptimal vs fine-tuning* (Mantis) and fine-tuning is forbidden → the **head + optional preprocessing carry the load**. This is also why **Track A (handcrafted) + fusion** matters.

---

## Refuted (do NOT cite these numbers)
- RevIN "−89% MASE" / "+44% over others" (0-3) — only the *model-specific* conclusion survives.
- Reference-frame transform recovers "~100% of orientation-induced loss" (0-3) — gains are **partial**.
- "Mantis MUST interpolate 500→512" (1-2) — documented mechanism, but verify on the checkpoint.

## Open / unresolved (flagged honestly)
- Internal normalization + input contracts of **Chronos / MOIRAI / TimesFM** — not confirmed here.
- **SensorLM** (2506.09108), **"Time to Embed"** (2505.14543), **TimesFM-fusion** (UbiComp'25) recommendations — not resolved.
- **arXiv:2604.02711** (HAR-FM survey): this run flagged the ID as implausible, but given today is 2026-06, "2604" = Apr 2026 is valid and a prior verification confirmed it exists — re-check directly before citing.
- Whether gravity removal is **redundant under internal RevIN/instance-norm** — plausible, not directly tested → **V1 ablation answers it empirically**.
- **Spectrogram→frozen-ViT** preprocessing — unresolved; needs a dedicated check for Track C.

## Key sources
2512.02833 (normalization is model-specific) · 2402.03885 MOMENT (RevIN, T=512, channel-indep) · 2502.15637 Mantis (internal z-score, channel-indep, frozen suboptimal) · 2410.19818 + repo UniMTS (6-ch, m/s², 10 s, rotation-aug) · LIMU-BERT paper+repo (acc/9.8, mag-unit×2, 20 Hz/120, z-score hurts) · PMC9313140 (orientation-invariant feats → LOSO) · MDPI 23:5845 (gravity/forward frame).
</content>
