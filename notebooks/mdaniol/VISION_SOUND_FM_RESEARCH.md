# Vision / Sound Foundation Models for IMU — deep research synthesis (2026-06-27)

4 parallel arXiv research agents (audio-spectrogram FMs · vision FMs · TS-as-image HAR
literature · cross-modal/native-IMU FMs), each with a hard **verify-the-weights** mandate.
Goal: a frozen vision/sound FM as an **orthogonal diversity voter** alongside our temporal-FM
vote (UTICA+MantisV2, 0.834). Constraint: frozen-only, public weights, single-location 9-ch
phone IMU @100 Hz / 5 s.

## Headline: a frozen vision/sound FM as a diversity voter is EVIDENCE-BACKED (not speculative)
- **TiViT (arXiv:2506.08641, 2025)** — frozen CLIP/DINOv2/SigLIP ViT + *linear probe only* beats
  dedicated time-series FMs on UCR-128 (TiViT-CLIP 81.3% > Mantis 80.1% > Moment 79.0%), and the
  **ensemble** TiViT+Mantis = **83.0%** with representation alignment only **0.32–0.34** → genuinely
  decorrelated. Direct validation of our exact plan. *CLIP > DINOv2; use INTERMEDIATE layers (40–70%
  depth), not the final block.*
- **IMG2IMU (arXiv:2209.00945)** — frozen ImageNet on IMU spectrograms beats sensor-SSL by +9.6 F1,
  from-scratch by +26.8.
- **LIGO (arXiv:1706.07446)** — frozen ImageNet features + logreg → 95%+ on spectrograms.
- **SHL history:** NO prior challenge entry (2018–2020) used any ImageNet/audio pretrained backbone →
  genuinely novel for HASCA. (Spectrogram-CNN entries peaked ~0.88; handcrafted+DL+HMM winners ~0.94.)

## Three evidence-backed diversity axes (weights all confirmed downloadable)

### Axis A — NATIVE-IMU frozen encoders (most novel, cheapest, no rendering; domain-gap risk)
| model | arXiv | weights | input | note |
|---|---|---|---|---|
| **PRIMUS** | 2411.15127 | Zenodo 15147513 (622 MB, **CC-BY**) | 6-ch acc+gyr, 5 s, **50 Hz**, ~1.4 M params | purpose-built frozen+linear-probe IMU encoder; contract closest to ours (100→50 Hz) |
| **ImageBind-IMU** | 2305.05665 | fbaipublicfiles (4.8 GB, CC-BY-NC) | `[B,6,2000]` acc+gyr → 1024-d | native multimodal FM; trivial frozen forward |
| risk | — | — | — | BOTH trained ONLY on **Ego4D head-mounted Aria** IMU → never phone/transport. Validate standalone (per-class incl. Run/vehicles) before the vote — head-motion bias could confuse stationary/vehicle. |

### Axis B — FROZEN VISION ViT on IMU-as-image (STRONGEST published evidence; reuses our spectrogram)
| model | arXiv | weights (HF/timm) | note |
|---|---|---|---|
| **CLIP / OpenCLIP ViT-L/H** | 2103.00020 | `openai/clip-vit-large-patch14`, `laion/CLIP-ViT-H-14-laion2B` | TiViT's BEST backbone; use vision tower + intermediate layer |
| **DINOv2-reg ViT-L** | 2304.07193 / 2309.16588 | `facebook/dinov2-with-registers-large` | only one with *direct* frozen-TS-transfer evidence; registers = clean pooled features |
| **SigLIP2-SO400M** | 2502.14786 | `google/siglip2-so400m-patch14-384` | strongest generic frozen probe (norm = [-1,1] footgun) |
| **EVA-02-L MIM / ConvNeXt-V2-L** | 2303.11331 / 2301.00808 | `timm/eva02_large_patch14_448.mim_m38m`, `timm/convnextv2_large.fcmae` | architecturally orthogonal → max *added* diversity (EVA=ViT, ConvNeXt=CNN) |
| encoding | — | — | transport ⇒ **spectrogram/STFT** (frequency cues: engine/cadence/rail); also A/B the **trivial 2D reshape** (TiViT's winner) and **GADF** (best general-IMU encoding). MTF = trap on IMU. |

### Axis C — FROZEN AUDIO FM on IMU spectrogram (our AST branch; novel, higher-risk: no IMU precedent)
| model | arXiv | weights | note |
|---|---|---|---|
| **AST** | 2104.01778 | `MIT/ast-finetuned-audioset-10-10-0.4593` | BUILT (E-AST-01); spectrogram-bypass is HF-native |
| **Dasheng** | 2406.06992 | `mispeech/dasheng-base` | general SSL audio *encoder* (pooled output by default) → diverse from supervised AST |
| **PANNs CNN14** | 1912.10211 | `qiuqiangkong/audioset_tagging_cnn` | a CNN → orthogonal to our all-transformer vote; 2048-d embedding API |
| caveat | — | — | NO arXiv benchmark of a frozen audio FM on IMU/transport spectrograms → exploratory; vision (Axis B) is lower-risk. |

## Load-bearing implementation caveats (from the literature)
1. **Use INTERMEDIATE ViT layers** (TiViT: 40–70% depth), NOT the penultimate/final — naive extraction underperforms.
2. **Probe head matters** (Bird-MAE 2504.12880): a bare linear probe can trail fine-tuning badly; **our LightGBM downstream is already a strong non-linear probe → in our favor.**
3. **Per-channel render**, robust-scaled; CLIP-pretrained > ImageNet/DINOv2 per TiViT; mind per-family normalization.
4. Frozen trails full fine-tuning (which we can't do) → expect a **complement, not a standalone winner**; keep only if it improves the vote on the BHT lock (+ no per-class regression).

## RULED OUT (no weights / no IMU modality / wrong front-end)
AURA-MFM, IMU2CLIP, COMODO (no/unconfirmed weights) · OmniBind, LanguageBind, Point-Bind (no IMU
modality) · Whisper, Wav2Vec2, HuBERT, WavLM (waveform front-end, NOT spectrogram-bypassable) ·
(earlier: oneHAR, MASTER, SensorLM, LSM, UniMTS).

## Recommendation (ranked by evidence × ROI)
1. **Finish E-AST-01** (built) — first spectrogram-FM data point, ~free.
2. **Axis B — frozen vision ViT on our IMU spectrogram, intermediate layer** (CLIP-ViT-L or DINOv2-reg-L)
   — STRONGEST evidence (TiViT), reuses `imu_log_spectrogram` (resize 224 + RGB-replicate). The
   highest-confidence new diversity voter.
3. **Axis A — PRIMUS** (then ImageBind) — most NOVEL (native IMU, no rendering, tiny), cheapest to run;
   validate standalone first (domain gap).
Then re-vote {utica_V2, mantisv2_V1, + survivors}; keep only lock-test improvers.
