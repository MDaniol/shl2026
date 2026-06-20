# SHL 2026 — Model Selection Reference (frozen FMs, weight-verified)

Source: deep-research, 22 sources, **25/25 claims confirmed, 0 refuted**. Focus:
newest frozen FMs with **downloadable public weights**, for 9-ch IMU transport-mode,
maximizing macro-F1 **and** novelty (2025 already used MOMENT/Chronos/Flamingo/BERT).

## Headline change
**MantisV2 is the top *evaluable* pick** — it post-dates 2025 (novel), has public
MIT weights, natively ingests **all 9 channels incl. magnetometer** (per-channel
encode→concat), and has **direct HAR evidence**: 0.9013 acc on smartphone UCI-HAR,
0.7562 avg over 7 HAR sets, **all zero-shot/frozen**. Demote MOMENT/Chronos to baselines.

## Ranked shortlist (all weights verified downloadable now)

| Rank | Model | arXiv (date) | Modality | Frozen embed | 9-ch incl. Mag? | Weights | Novel? |
|---|---|---|---|---|---|---|---|
| **1** | **MantisV2** | 2602.17868 (Feb 2026) | general TS-classification, 4.2M (→2.2M) | ✅ extract→classify | ✅ per-channel | HF `paris-noah/MantisV2` (MIT) | ✅ |
| **2** | **Mantis / Mantis+** | 2502.15637 (Feb 2025) | general TS, 8M | ✅ `MantisTrainer.transform` | ✅ per-channel (+optional adapter) | HF `paris-noah/Mantis-8M`, `MantisPlus` | ✅ |
| **3** | **UTICA** | 2603.01348 (Mar 2026) | TS, **DINOv2-style self-distillation** (most novel paradigm) | ✅ | ✅ (inherits Mantis backbone) | HF `fegounna/Utica` | ✅✅ |
| **4** | **UniMTS** | 2410.19818 (NeurIPS'24) | **IMU/motion-native** | ✅ zero-shot | ❌ **6-ch only (acc+gyr, NO Mag)** | HF `xiyuanz/UniMTS` | ✅ |
| **5** | **NormWear** | 2412.09758 (Dec 2024) | multivariate wearable, **channel-AWARE attention** (models cross-channel) | ✅ `get_embedding` | ✅ arbitrary channels | HF `mosaic-laboratory/normwear` | ✅ |
| **6** | **LIMU-BERT** | SenSys'21 (verified prior run) | IMU-native, uses Mag | ✅ frozen + head | ✅ 9-ch, bespoke scaling | `dapowan/LIMU-BERT-Public` (MIT) | ✅ (not in prior SHL) |
| alt | **Forecasting-FM-as-classifier** | 2510.26777 (Oct 2025) | TimesFM 2.0 / MOIRAI / TiRex frozen→features | ✅ | varies | public | ✅ (if not Chronos) |

## ❌ Disqualified — do NOT spend time
- **RelCon** (2411.18822, Apple): **weights cannot be released** (README confirms) + **accelerometer-only**. Fails both hard requirements.

## Key technical notes
- **Channel handling**: Mantis-family + UTICA are **channel-independent by default** (encode each of 9 channels, concatenate). They *can* take the magnetometer (just another channel) but **don't model Acc↔Gyr↔Mag interaction natively** unless you add a trainable `LinearChannelCombiner` adapter (allowed — backbone stays frozen). **NormWear's channel-aware attention models cross-channel interaction natively** → good ensemble complement.
- **Input length**: Mantis-family resize to **512** (our 500 → interpolate); UniMTS pad to 10 s; LIMU-BERT resample 100→20 Hz. (See `fm_input.py`.)
- **No published SHL transport macro-F1 exists for any** — HAR benchmarks (UCI-HAR/HHAR) are the closest proxy. The bake-off must measure it.

## Diversity logic (for the ensemble)
- **Channel-independent TS**: MantisV2 (lead), UTICA (novel paradigm)
- **IMU/motion-native**: UniMTS (6-ch), LIMU-BERT (9-ch incl. Mag)
- **Cross-channel attention**: NormWear
- **Baselines (not novel)**: MOMENT, Chronos
That spans 4 distinct inductive biases → strong ensemble.

## Recommended evaluation set (top 5–6)
**MantisV2 → UTICA → UniMTS → LIMU-BERT → NormWear**, with **MOMENT as baseline**.
MantisV2/UTICA = novelty + 9-ch + HAR evidence; UniMTS/LIMU-BERT = domain-native;
NormWear = cross-channel diversity. Pick winners by **val macro-F1** (users 2&3,
per-location), then fuse with the 520 handcrafted features.

## Unverified in this run (check before trusting IDs)
SensorLM (2506.09108), HAR-FM survey (2604.02711), TIC-FM (2602.00620), Bio-PM
(2603.10961), SelfPAB, oneHAR, Wonderwall, MASTER, AURA-MFM, NuTime, UniTS,
SensorLLM (2410.10624), AST/DINOv2/SigLIP specifics. LIMU-BERT was verified in a
prior run. (Vision-FM strategy is the subject of a separate deep-research run.)

## Execution note (2026-06-20, from cloning the repos)
- **MantisV2 / UTICA / Mantis8M / MOMENT** — load + extract embeddings cleanly on
  the MacBook (MPS, ~255–270 win/s). MantisV2→(n,9,256)→flatten 2304; MOMENT→(n,512).
  Loading quirks resolved: pin `transformers==4.44.2`/`huggingface_hub==0.25.2`;
  Mantis `from_pretrained` is an **instance** method (call on `MantisV2(device=...)`,
  not the class); UTICA = `Mantis8M` arch + `load_state_dict(strict=False)`.
- **LIMU-BERT** — ⚠️ released checkpoints are **per-dataset, small-scale**
  (`pretrain_base_{uci,hhar,motionsense,shoaib}_20_120`), mostly **6-channel**,
  20 Hz/120. **No single large general checkpoint** → weak as a "foundation model".
  **Deprioritized** (usable only as a small-data frozen encoder if at all).
- **UniMTS** — HF weights exist but need its custom graph-net model class + 6-ch
  (acc+gyr), m/s² input → moderate custom loader; queue after the core bake-off.
- **NormWear / BIOT / CBraMod** — not yet loaded; later.

## Caveat
Weight links verified mid-2026 — re-confirm before the sprint. No SHL-specific
macro-F1 published → all rankings are priors; the empirical bake-off decides.
</content>
