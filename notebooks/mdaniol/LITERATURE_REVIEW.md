# SHL 2026 — Literature Review (arXiv-focused)

**Compiled**: 2026-06-15 · **Method**: deep-research harness (22 sources → 98 claims → 25 adversarially verified, 3 refuted, 17 synthesized). Every quantitative claim below was 3-vote checked; refuted items are listed explicitly so they are *not* relied on.
**Purpose**: choose which frozen foundation models + methods to evaluate for the SHL 2026 frozen-FM, user-independent, 9-channel transport-mode task.

> ⚠️ **Global caveat**: all FM accuracy numbers below come from UCR/UEA or general motion benchmarks — **not** transportation-mode data, and mostly **accuracy, not macro-F1** (the SHL ranking metric). Treat them as priors for the Day-2 bake-off, not as expected SHL scores.

---

## TL;DR — what to evaluate (priority order)

| Rank | Model | Year | arXiv | Why | Frozen-rule fit | Public weights |
|---|---|---|---|---|---|---|
| **1** | **UniMTS** | 2024 | 2410.19818 (NeurIPS'24) | IMU-native; *explicitly engineered* for cross-device-location + cross-orientation generalization — our exact shift. Zero-shot frozen via text-label matching. | ✅ true zero-shot | ✅ HF `xiyuanz/UniMTS` + GitHub |
| **2** | **Mantis** | 2025 | 2502.15637 (ICML'25) | Strongest *frozen-encoder* TSFM on UCR/UEA (0.7816 acc, beats MOMENT/NuTime); 8M ViT; channel adapters for 9-ch; len-512 (our 500 interpolates near-lossless). | ✅ frozen encoder + RF/linear head | ✅ HF + `vfeofanov/mantis` |
| **3** | **LIMU-BERT** | 2021 | SenSys'21 | Lightweight, MIT-licensed frozen IMU encoder; ingests 6- or 9-ch Acc/Gyr/Mag. | ✅ frozen encoder + GRU/MLP head | ✅ `dapowan/LIMU-BERT-Public` (mostly 6-ch checkpoints) |
| **4** | **MOMENT** | 2024 | 2402.03885 (ICML'24) | Built for limited-supervision: zero-shot embeddings + linear probing on frozen backbone. | ✅ frozen backbone + head | ✅ **verified** HF `AutonLab/MOMENT-1-large`, **MIT**, `pip install momentfm` (only the CC-BY *license* claim was wrong) |
| Stretch | AST / DINOv2 / SigLIP on spectrograms | — | Vision-FM branch (Day 5). | ✅ off-the-shelf frozen | ✅ (not researched in depth here) |
| Defer | IMG2IMU, IMU2CLIP, SensorLLM, UniTS | see below | Each needs custom pretraining / alignment / task-tuning → tension with strict frozen rule. | ⚠️ partial | mixed |

**Action**: Day-2 bake-off = **UniMTS + Mantis + LIMU-BERT + MOMENT** (frozen embeddings → cheap probe → val macro-F1, users 2&3, per-location breakdown). Carry the best IMU-native + best TS-FM into the fused pipeline.

---

## 1. Time-series foundation models (frozen embedding extractors)

- **Mantis** (2502.15637, ICML'25) — 8M-param patch+ViT encoder, contrastively pretrained on ~1.89M univariate series (corpus size flagged ambiguous in their repo). **Designed to emit embeddings from a frozen encoder.** Frozen-encoder + RF head: **0.7816 avg accuracy over 159 UCR/UEA datasets**, beating MOMENT (0.7663) and NuTime (0.7639); best on 74/159. Input fixed at **512** (interpolated); multivariate via channel-independent encoding *or* a lightweight channel adapter (R^{d×t}→R^{dnew×t}). → **Top TS-FM pick.** *Caveats: self-reported; accuracy not macro-F1; not transport data.*
- **MOMENT** (2402.03885, ICML'24) — explicitly built/evaluated for limited-supervision (zero-shot embeddings, linear probing with frozen backbone). Strong fit to the frozen-head pattern. ⚠️ **The specific claim of CC-BY-4.0 weights on HF (`AutonLab/MOMENT-1-large`) was REFUTED (0-3 vote) — verify weight availability and license directly before committing.**
- **UniTS** (2403.00131, NeurIPS'24) — unified multi-task model (forecast/classify/anomaly/impute) via task tokenization; 38 datasets incl. activity sensors. ⚠️ Its **few-shot/prompt transfer claim was REFUTED (1-2)** — frozen-only suitability for SHL unproven. Lower priority.
- **NOT covered by surviving claims** (remain open — research before relying): Chronos / Chronos-2, MOIRAI / MOIRAI-MoE, TimesFM, Timer / Time-MoE, GPT4TS, NuTime (beyond the comparison number), UTICA. *Note from earlier manual checks this session: Chronos (2403.07815) and Chronos-2 (2510.15821) exist but are forecasters — awkward as embedders.*

## 2. IMU / wearable-native foundation models (best domain fit)

- **UniMTS** (2410.19818, NeurIPS'24) — **the standout.** Motion-time-series FM with **true zero-shot frozen classification** (contrastive text-label matching). Built for our shift: **spatio-temporal graph nets for cross-device-location generalization** + **rotation-invariant augmentation for cross-orientation robustness**; targets cross-*dataset* generalization. Reports +340% zero-shot / +16.3% few-shot / +9.2% full-shot over baselines on 18 datasets. ⚠️ *340% is relative over weak zero-shot baselines, not absolute F1; magnetometer support unclear (motion = accel/gyro-style) — may need handling for our 9-ch Mag.* → **Top IMU-native pick.**
- **LIMU-BERT** (SenSys'21) — BERT-style SSL IMU encoder, frozen + lightweight GRU/MLP head; **MIT license**; supports F=6 or 9 (Acc/Gyr/Mag). ⚠️ *Shipped checkpoints pretrained on HHAR/UCI/MotionSense/Shoaib (activity, mostly 6-ch); authors note "magnetometer adds little"; transport-mode transfer untested — a ready 9-ch checkpoint may not exist out of the box.*
- **Oxford UK-Biobank SSL** (2206.02909) — ResNet18 SSL on >700k person-days; public (`oxwearables/ssl-wearables`). ⚠️ **Accelerometer-only, wrist-domain** → weaker fit than UniMTS/LIMU-BERT; its frozen-transfer F1-gain claim was **REFUTED (1-2)**.
- **Additional candidates from the survey** (scan for public weights): **RelCon** (2411.18822, ICLR'25, Apple — relative contrastive, 1B segments / 87,376 participants), **SelfHAR** (accel-only SSL, teacher-student), **oneHAR** (universal IMU embeddings, cross-dataset transfer — notable for our setting), **MASTER** (masked cross-sensor modeling). *Public-weight availability not individually verified.*

## 3. Vision FMs on spectrograms

- **IMG2IMU** (2209.00945) — IMU→spectrogram→ViT works (+9.6 pp avg F1 over sensor-trained baselines, 4 tasks), **but requires custom sensor-aware contrastive image pretraining** — NOT plug-and-play frozen. → Lower priority under the rule/timeline.
- **WatchHAR** (2509.04736) — found but not deeply verified; earlier manual check: audio+inertial, on-device, not explicitly spectrogram-based.
- **AST / DINOv2 / CLIP / SigLIP** as off-the-shelf frozen encoders on IMU spectrograms — **not researched in depth here**; these remain the cleanest *rule-compliant* vision option for the Day-5 stretch (AST is spectrogram-native; DINOv2 has the strongest frozen features). Validate empirically.

## 4. LLMs / sensor-language models

- **SensorLLM** (2410.10624, EMNLP'25) — two-stage: Sensor-Language Alignment (auto trend descriptions, no labels) + **Task-Aware Tuning (uses labels)**. The tuning stage is in tension with the strict frozen rule. "No human annotations" applies only to alignment. → Defer.
- **IMU2CLIP** (2210.14395, Meta/FAIR) — aligns IMU↔video↔text in frozen CLIP space, **but the IMU encoder itself is trained** and downstream benefits from fine-tuning; released encoder is egocentric-video domain (far from smartphone transport). → Partial fit; defer.

## 5. Transportation mode recognition & SHL history

- **SHL 2026 ≈ SHL 2020 (user-independent)**: train on **User1 across 4 positions** (Bag/Hips/Torso/Hand), test on **User2/User3** at fewer positions. The three-year review (Frontiers 2021, 10.3389/fcomp.2021.713719) and JSI lessons (PMC9145859) are directly mineable. **Confirmed input spec matches: 9-ch Acc/Gyr/Mag @100 Hz, 5 s = 500 samples.**
- ⚠️ **Biggest divergence: per-window SHUFFLED test** → **cross-window temporal smoothing (HMM/voting) is FORBIDDEN**, even though it was the single biggest gain for many past SHL winners. Feature/window pipelines transfer; post-processing tricks do not. (Also: Hand dropped at test → 3 positions.)
- Other sources surfaced (verify before citing): SHL'24 summary (ACM 10.1145/3675094.3678456), attention-MIL TMD (2404.15323), and 2310.11087, 1910.04739.

## 6. Cross-user / cross-position generalization (frozen features)

- Classic **DANN over user IDs is impossible** with a single training user (confirmed by setup). Exploit the **position** axis instead — which is exactly what **UniMTS** bakes in (graph nets + rotation-invariant aug).
- **Rotation/orientation-invariant aggregation + per-window z-normalization** (2407.11048, SHL'24 missing-modality) is the verified, rule-compatible robustness lever.
- Test-time adaptation that relies on temporal order is out (shuffled test). Within-window invariance is the play.
- Additional sources to mine: 2406.18569, 2207.11221, 2508.01894, 2503.13542, 2307.04516 (cross-user/position DG + handcrafted features).

## 7. Handcrafted / physics features (classical baseline)

- Covered in depth in `IMPLEMENTATION_PLAN.md §2` (subband energy/ratio, FFT-peak cadence, magnetometer rail-vs-road, quantiles; derived streams ×65-feature bank). The survey confirms **lightweight heads/adapters on frozen features are the field-standard retargeting mechanism** — i.e., a probe/MLP/RF on embeddings is both compliant and endorsed.
- catch22 / tsfresh specifics were requested but **not covered by surviving claims** — treat the §2 plan (grounded in Wang et al. 2019 + JSI 2022) as the authoritative feature spec.

## 8. Survey anchor

- **"Foundation Models Defining a New Era in Sensor-based HAR"** (arXiv:**2604.02711**, Bian et al., Apr 2026) — **verified to exist** (unusual `2604` prefix confirmed real). Taxonomy: unimodal / multimodal / cross-modal FMs + three LLM forms; endorses lightweight adapters/projection heads. → Use as the related-work organizing reference for the HASCA paper.

---

## Refuted claims (do NOT rely on)
1. **MOMENT public weights on HF under CC-BY-4.0** — REFUTED (0-3). Re-verify directly.
2. **UniTS few-shot/prompt transfer** relevant to frozen/single-user — REFUTED (1-2).
3. **UK-Biobank SSL frozen-transfer F1 gains (2.5–100%)** — REFUTED (1-2).

## Open questions (drive next steps)
1. Do **public 9-channel (incl. Mag) frozen checkpoints** exist for UniMTS / LIMU-BERT, or is a Mag adapter needed (without violating the frozen rule)?
2. What is the **macro-F1** of frozen UniMTS/Mantis/MOMENT embeddings + lightweight head on SHL transport modes under cross-user, cross-position, shuffled eval? → **answered empirically by the Day-2 bake-off.**
3. Which **uncovered TS-FMs** (Chronos-2, MOIRAI-MoE, TimesFM, Time-MoE, NuTime, GPT4TS, UTICA, SensorLM) emit fixed frozen embeddings, support multivariate 9-ch, and have permissive licenses?
4. Strongest **within-window, orientation/position-invariant** handcrafted set to benchmark against / fuse with FM embeddings (shuffling forbids temporal features).

## Key arXiv IDs (verified this session)
2410.19818 UniMTS · 2502.15637 Mantis · 2402.03885 MOMENT · 2403.00131 UniTS · 2206.02909 UK-Biobank SSL · 2411.18822 RelCon · 2209.00945 IMG2IMU · 2210.14395 IMU2CLIP · 2410.10624 SensorLLM · 2407.11048 rotation-invariant/missing-modality · 2604.02711 HAR-FM survey · 2404.15323 attention-MIL TMD.
</content>
