# SHL 2026 — Foundation Models Catalog

**Compiled**: 2026-06-15 · Every FM surfaced across `LITERATURE_REVIEW.md`, `IMPLEMENTATION_PLAN.md`, the Mistral draft, and this session's research.
**Use**: pick bake-off candidates. Feasibility is judged against our constraints — **frozen-FM rule, 9-ch Acc/Gyr/Mag, 500-sample window, single training user, location shift, shuffled test (no temporal smoothing), macro-F1, ~1-week budget.**

**Feasibility legend**: 🟢 High (frozen embedder, public weights, fits our input) · 🟡 Medium (promising but a real gap to close) · 🔴 Low / Defer (rule tension, no weights, or poor domain/input fit).
⚠️ All FM benchmark numbers are **accuracy on UCR/UEA or generic motion data — not SHL macro-F1**. Treat as priors; the Day-2 bake-off decides.

---

## A. Time-series foundation models (generic)

| Model | Year | Description | Feasibility for SHL 2026 | Weights | Link | Key risks |
|---|---|---|---|---|---|---|
| **Mantis** | 2025 | 8M ViT, contrastive; built to emit embeddings from a **frozen** encoder | 🟢 strongest frozen-probe TSFM (0.7816 UCR); len-512, 9-ch via channel adapter | ✅ HF + GitHub | [2502.15637](https://arxiv.org/abs/2502.15637) · [repo](https://github.com/vfeofanov/mantis) | domain-agnostic (no IMU/transport prior); acc≠macro-F1 |
| **MOMENT** | 2024 | 385M T5 masked auto-encoder; zero-shot embeddings + linear probing | 🟢 designed for frozen-backbone + head | ✅ **verified** HF `AutonLab/MOMENT-1-large`, **MIT**, `pip install momentfm` | [2402.03885](https://arxiv.org/abs/2402.03885) | generic (no IMU prior); 512 context |
| **UTICA** | 2026 | DINOv2-style self-distillation on Mantis tokenizer | 🟡 **new best frozen probe 0.794 UCR**; same eval protocol as Mantis | ⚠️ very new — verify release | [2603.01348](https://arxiv.org/abs/2603.01348) | weeks old; weights may not be out |
| **MantisV2** | 2026 | Mantis successor; synthetic data + test-time strategies to close zero-shot gap | 🟡 drop-in upgrade to Mantis if released | ⚠️ verify | [2602.17868](https://arxiv.org/abs/2602.17868) | very new; weights unverified |
| **NuTime** | 2023 | 2M transformer, numerically multi-scaled embedding | 🟡 frozen-probe baseline (0.7639 UCR, < Mantis) | likely public | [2310.07402](https://arxiv.org/abs/2310.07402) | weaker than Mantis; generic |
| **UniTS** | 2024 | Unified multi-task (forecast/classify/anomaly/impute) via task tokens | 🔴 frozen/few-shot transfer claim **refuted (1-2)** | research | [2403.00131](https://arxiv.org/abs/2403.00131) | frozen suitability unproven |
| **Chronos** | 2024 | LLM-style tokenized TS **forecaster** | 🔴 forecaster — no clean classification embedding | ✅ HF | [2403.07815](https://arxiv.org/abs/2403.07815) | wrong task head; awkward as embedder |
| **Chronos-2** | 2025 | Universal/multivariate forecaster | 🔴 same — forecaster | ✅ HF | [2510.15821](https://arxiv.org/abs/2510.15821) | as above |
| **Chronicle** | 2026 | Multimodal joint language + time-series | 🔴 heavy; multimodal scope beyond need | verify | [2605.20268](https://arxiv.org/abs/2605.20268) | overkill for 1-week |
| MOIRAI / MOIRAI-MoE | 2024 | Multivariate forecaster (Salesforce) | 🔴 forecaster; **not researched** here | ✅ likely | gap — verify | embedding use unproven |
| TimesFM | 2024 | Decoder forecaster (Google) | 🔴 forecaster; **not researched** | ✅ | gap — verify | as above |
| Time-MoE / Timer | 2024 | MoE/decoder forecasters | 🔴 forecaster; **not researched** | partial | gap | as above |
| GPT4TS | 2023 | GPT-2 partially fine-tuned for TS | 🔴 needs fine-tuning (rule tension) | ✅ | gap | not frozen-clean |
| "Time to Embed" | 2025 | *Method*: channel descriptions to unlock FM embeddings for multivariate TS | 🟡 technique to apply to our 9-ch, not a model | n/a | [2505.14543](https://arxiv.org/abs/2505.14543) | method only |

## B. IMU / wearable-native foundation models (best domain fit)

| Model | Year | Description | Feasibility for SHL 2026 | Weights | Link | Key risks |
|---|---|---|---|---|---|---|
| **LIMU-BERT** | 2021 | BERT-style SSL IMU encoder; frozen + light GRU/MLP head | 🟢 **best input fit** — single-device, 6/9-ch Acc/Gyr/Mag, MIT | ✅ public (mostly 6-ch ckpts) | [repo](https://github.com/dapowan/LIMU-BERT-Public) · [paper](https://dl.acm.org/doi/10.1145/3485730.3485937) | trained on daily activities not transport; Mag underused |
| **UniMTS** | 2024 | Motion FM; graph nets (cross-location) + rotation-invariant aug; zero-shot via text labels | 🟡 motivation = our exact shift, **but** joint-graph assumes multi-location; **no Mag**; locomotion-trained → weak on vehicle classes | ✅ HF + GitHub | [2410.19818](https://arxiv.org/abs/2410.19818) | single-phone collapses its edge; vehicle-class zero-shot risk |
| **SensorLM** | 2025 | "Language of wearable sensors" — sensor-language FM (Google) | 🟡 sensor-native, novel | ⚠️ verify release | arXiv 2025 — verify ID | alignment may need tuning |
| **RelCon** | 2025 | Motion FM, relative-contrastive, 1B segments/87k participants (Apple); SOTA HAR+gait | 🔴 high relevance but **weights likely closed**; accel-only | ⚠️ likely not public | [2411.18822](https://arxiv.org/abs/2411.18822) | no Gyr/Mag; availability |
| **oneHAR** | 2025 | LLM-assisted **universal cross-dataset** IMU HAR (CDNet) | 🔴 generalization-focused but research code | research | [doi 3749509](https://dl.acm.org/doi/10.1145/3749509) | weight availability |
| **Wonderwall** | 2026 | Virtual-to-real FM for IMU HAR | 🔴 very new | ⚠️ verify | [doi 3789688](https://dl.acm.org/doi/10.1145/3789688) | unproven, weights |
| Oxford UK-Biobank SSL | 2022 | ResNet18 SSL on >700k person-days wrist accel | 🔴 **accel-only, wrist**; transfer-gain claim **refuted (1-2)** | ✅ `oxwearables/ssl-wearables` | [2206.02909](https://arxiv.org/abs/2206.02909) | poor input fit (1-ch, wrist) |
| Bio-inspired SSL (wrist) | 2026 | Wrist-accelerometer SSL | 🔴 wrist accel-only; very new | verify | [2603.10961](https://arxiv.org/abs/2603.10961) | input fit; new |
| SelfHAR / MASTER | 2021/24 | Accel SSL (teacher-student) / masked cross-sensor | 🔴 catalog-only; weights unverified | partial | via survey | availability |
| LIMU-BERT-X | 2025 | Commercial LIMU-BERT, 1.43M hrs / 60k subjects | 🔴 **commercial — no public weights** | ❌ | MobiCom'25 | not released |

## C. Vision FMs on spectrograms (Track C stretch)

| Model | Year | Description | Feasibility for SHL 2026 | Weights | Link | Key risks |
|---|---|---|---|---|---|---|
| **AST** | 2021 | Audio Spectrogram Transformer — **spectrogram-native** ViT | 🟡 best modality match for IMU→spectrogram; off-the-shelf frozen | ✅ | [2104.01778](https://arxiv.org/abs/2104.01778) | domain = audio; needs spectrogram design |
| **DINOv2** | 2023 | Self-supervised ViT, strongest frozen image features | 🟡 robust frozen embeddings on spectrogram images | ✅ | [2304.07193](https://arxiv.org/abs/2304.07193) | natural-image domain gap |
| CLIP | 2021 | Image-text contrastive ViT | 🟡 frozen features; used in 2025 (less novel) | ✅ | [2103.00020](https://arxiv.org/abs/2103.00020) | domain gap; not novel |
| SigLIP | 2023 | Sigmoid-loss CLIP variant | 🟡 alt to CLIP | ✅ | [2303.15343](https://arxiv.org/abs/2303.15343) | domain gap |
| **IMG2IMU** | 2022 | IMU→spectrogram + **custom** sensor-aware image pretraining | 🔴 requires custom pretraining → rule tension | research | [2209.00945](https://arxiv.org/abs/2209.00945) | not plug-and-play frozen |
| SPECTRA | 2026 | Spectral-informed efficient HAR net (edge) | 🔴 a model to train, not a frozen FM | research | [2603.26482](https://arxiv.org/abs/2603.26482) | not an FM in our sense |
| WatchHAR | 2025 | On-device real-time HAR (audio+inertial) | 🔴 on-device system, not a frozen FM | research | [2509.04736](https://arxiv.org/abs/2509.04736) | not a reusable encoder |

## D. LLM / sensor-language / cross-modal

| Model | Year | Description | Feasibility for SHL 2026 | Weights | Link | Key risks |
|---|---|---|---|---|---|---|
| **SensorLLM** | 2024 | Aligns LLM with motion sensors (2-stage: align + **task-aware tuning**) | 🔴 task-tuning stage uses labels → rule tension | ✅ GitHub | [2410.10624](https://arxiv.org/abs/2410.10624) | tuning conflicts with frozen rule; heavy |
| **IMU2CLIP** | 2022 | Aligns IMU↔video↔text into frozen CLIP space | 🔴 IMU encoder is **trained**; egocentric domain | ✅ FAIR | [2210.14395](https://arxiv.org/abs/2210.14395) | wrong domain; not frozen IMU side |
| X-Fi | 2024 | Modality-invariant multimodal human-sensing FM | 🔴 heavy cross-modal fusion; overkill | research | [2410.10167](https://arxiv.org/abs/2410.10167) | scope/time |
| Flamingo | 2022 | Visual-language few-shot model | 🔴 used in 2025; heavy; not IMU | ✅ (OF) | [2204.14198](https://arxiv.org/abs/2204.14198) | not novel; heavy |
| BERT | 2018 | Bidirectional text transformer | 🔴 used in 2025; not sensor-native | ✅ | [1810.04805](https://arxiv.org/abs/1810.04805) | not novel; needs tokenization hacks |

## E. Reference (not a candidate)

| Item | Year | Description | Link |
|---|---|---|---|
| HAR-FM Survey | 2026 | "Foundation Models Defining a New Era in Sensor-based HAR" — taxonomy + candidate catalog | [2604.02711](https://arxiv.org/abs/2604.02711) |
| Rotation-invariant / missing-modality | 2024 | SHL'24 winner-style; z-norm + rotation-invariant aggregation | [2407.11048](https://arxiv.org/abs/2407.11048) |

---

## Bottom line — bake-off shortlist (≤6, by feasibility)
1. 🟢 **LIMU-BERT** (2021) — best input fit, ready frozen IMU encoder.
2. 🟢 **Mantis** (2025) — strongest frozen-probe generic TSFM.
3. 🟢 **MOMENT** (2024) — frozen embeddings *(verify weights first)*.
4. 🟡 **UniMTS** (2024) — high-variance domain bet; verify single-phone + Mag handling.
5. 🟡 **UTICA or MantisV2** (2026) — drop-in Mantis upgrade *if weights released*.
6. 🟡 **SensorLM** (2025) — sensor-native novelty *if weights released*.

Vision (Track C, later): **AST** and/or **DINOv2** on spectrograms.
**Gate before caching:** load-and-shape check — does each accept 500-sample, 9-ch (or defined Acc/Gyr/Mag subset, with input-side Mag adapter) and emit a window embedding? Anything failing this is infeasible regardless of paper claims.

> Not exhaustive — the survey (2604.02711) is the full catalog. IDs for AST/DINOv2/CLIP/SigLIP/SensorLM and the un-researched forecasters (MOIRAI/TimesFM/Time-MoE/GPT4TS) should be confirmed before citing in the paper.
</content>
