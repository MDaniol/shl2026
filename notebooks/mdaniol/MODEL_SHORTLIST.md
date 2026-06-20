# SHL 2026 — Model Shortlist (6 to try) + sources, weights, usefulness, supporting papers

**Compiled**: 2026-06-15 · Code/weights availability **verified this session** (HF + GitHub fetched). Supersedes the tentative shortlist in `FOUNDATION_MODELS_CATALOG.md` where they conflict.

**Selection rule**: must have **downloadable code + frozen weights now**, accept our window (≈500 samples / resamplable), emit a fixed embedding, and add modality diversity. Two candidates were **dropped on verification**: **SensorLM** (arXiv:2506.09108 — Google, trained on proprietary Fitbit/Pixel data, **no public weights**) and **UTICA** (arXiv:2603.01348 — **no code/weights released**). MantisV2 takes the "newest TSFM" slot since its weights *are* out.

---

## The 6

| # | Model | Family | Get code | Get weights | Install | License |
|---|---|---|---|---|---|---|
| 1 | **MOMENT** | Generic TS | [github moment](https://github.com/moment-timeseries-foundation-model/moment) | HF `AutonLab/MOMENT-1-large` (0.3B) | `pip install momentfm` | **MIT** ✅ |
| 2 | **Mantis / MantisV2** | Generic TS | [github vfeofanov/mantis](https://github.com/vfeofanov/mantis) | HF `paris-noah/Mantis-8M`, `paris-noah/MantisV2` | `pip install mantis-tsfm` | **Apache-2.0** ✅ |
| 3 | **UniMTS** | IMU-native | [github xiyuanzh/UniMTS](https://github.com/xiyuanzh/UniMTS) | HF `xiyuanz/UniMTS` | repo env | ⚠️ unstated — check |
| 4 | **LIMU-BERT** | IMU-native | [github dapowan/LIMU-BERT-Public](https://github.com/dapowan/LIMU-BERT-Public) | in repo `saved/` folder | repo env | **MIT** ✅ |
| 5 | **AST** | Vision (spectrogram) | [github YuanGongND/ast](https://github.com/YuanGongND/ast) | HF `MIT/ast-finetuned-audioset-10-10-0.4593` | `transformers` | BSD/MIT ✅ |
| 6 | **DINOv2** | Vision (general) | [github facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | HF `facebook/dinov2-base` | `transformers` | Apache-2.0 ✅ |

> 1–4 = core embedding track (Day 2–4). 5–6 = spectrogram stretch (Day 5, Track C). All weights confirmed reachable except UniMTS license (verify before paper).

---

## Usefulness assessment (for our exact case)

**1. MOMENT — 🟢 high, lowest-friction.**
`momentfm` with `task_name='embedding'` → frozen embeddings in a few lines; univariate (run each of 9 channels → concat). Context 512 (pad/interpolate our 500). MIT. *Useful as the reliable generic-TS baseline; no IMU/transport prior, so expect mid-pack until fused with physics features.*

**2. Mantis / MantisV2 — 🟢 high, best frozen-probe pedigree.**
`model.transform(X)` → frozen features; interpolate 500→512; multichannel via built-in channel adapter (or per-channel concat). Strongest reported frozen-encoder TSFM on UCR/UEA; **use the MantisV2 checkpoint** (released, lighter, closes zero-shot gap). *Likely the top generic-TS performer; cheap (8M).*

**3. UniMTS — 🟡 high-variance domain bet.**
Built for our cross-location/orientation shift; HF weights; zero-shot (`evaluate.py`) + linear-probe (`finetune.py`). **Critical limits:** **accelerometer+gyroscope only (6-ch) — drops magnetometer**, which we established is decisive for rail-vs-road (Train/Subway). 10 s windows (pad our 5 s); needs acc in m/s². *Worth trying for the location-shift story, but losing Mag may cap it on the vehicle classes; license unstated.*

**4. LIMU-BERT — 🟡 best input fit, but config mismatch.**
MIT; `embedding.py` → `.npy` frozen embeddings; **supports 9-ch Acc/Gyr/Mag** (the only shortlist model that does). **Risk:** released checkpoints default to **20 Hz / 120-sample** windows (and small activity datasets), not 100 Hz / 500 — we must **resample 100→20 Hz** to match, and the pretraining domain (daily activities) is off-transport. *Use as the IMU-native, magnetometer-aware candidate; treat resampling + domain gap as the main risks.*

**5. AST — 🟡 stretch, spectrogram-native.**
HuggingFace `transformers` one-liner; ViT pretrained on AudioSet spectrograms → strongest *modality match* for IMU→log-mel spectrograms. *Good novelty branch; design the spectrogram (per-channel → 3-band image) carefully.*

**6. DINOv2 — 🟡 stretch, strongest frozen image features.**
`transformers` `AutoModel`; robust frozen ViT features on spectrogram images. *Alternative/complement to AST; natural-image domain gap is the risk.*

**Net ranking for the bake-off priors:** MantisV2 ≈ MOMENT (safe generic) > LIMU-BERT (Mag-aware, resample risk) > UniMTS (domain bet, no Mag) > AST/DINOv2 (stretch). The data decides on Day 2.

---

## Integration gotchas (save debugging time)
- **Length**: MOMENT/Mantis want 512 → interpolate 500→512 (near-lossless). UniMTS wants 10 s → pad 5 s. LIMU-BERT → resample 100→20 Hz (~100 samples ≈ its 120 window).
- **Channels**: MOMENT/Mantis are **univariate** → encode 9 channels independently and concat (or Mantis adapter). UniMTS = **6-ch only**. LIMU-BERT = 6 or 9.
- **Magnetometer**: only LIMU-BERT ingests it natively. For MOMENT/Mantis it's just 3 more univariate channels. For UniMTS it's unusable → optionally add a tiny **input-side** Mag projection feeding a separate head (keeps backbones frozen).
- **Normalization**: global train-fit scaler; UniMTS expects acc in m/s².
- **Rule compliance**: freeze all backbones; only the probe/MLP head (+ optional input-side channel adapter) is trained.

---

## Related papers to boost traction & maximize output (HASCA paper)

**Method / frozen-FM adaptation (cite to justify the design):**
- **Generalized Prompt Tuning: Adapting Frozen *Univariate* TS FMs for *Multivariate* data** — [2411.12824](https://arxiv.org/abs/2411.12824). *Directly supports our 9-channel use of univariate MOMENT/Mantis.*
- **Time to Embed: channel descriptions for multivariate FM embeddings** — [2505.14543](https://arxiv.org/abs/2505.14543).
- **Mantis** (frozen linear-probe protocol) — [2502.15637](https://arxiv.org/abs/2502.15637); **MantisV2** — [2602.17868](https://arxiv.org/abs/2602.17868).
- **HAR-FM survey** (taxonomy, "lightweight heads are field-standard") — [2604.02711](https://arxiv.org/abs/2604.02711).

**Domain / SHL-specific (cite to ground features & robustness):**
- **Wang et al., Enabling Reproducible Research… SHL** (IEEE Access 2019) — feature MI/MRMR, **magnetometer = rail/road discriminator**, user-independent baselines.
- **JSI, What Actually Works… SHL 2019/2020** (Sensors 2022, PMC9145859) — derived streams, domain-shift tactics, *why temporal smoothing is unavailable here*.
- **Magnitude & Rotation-Invariant, missing modalities** — [2407.11048](https://arxiv.org/abs/2407.11048) — z-norm + rotation-invariant aggregation.
- **SHL three-year review** (Frontiers 2021) — confirms 2026 ≈ 2020 user-independent setup.

**Cross-modality / spectrogram justification (for Track C):**
- **IMG2IMU** — [2209.00945](https://arxiv.org/abs/2209.00945) — IMU→spectrogram→ViT works (+9.6 pp).
- **When Tabular FMs Transfer Across Modalities** — [2606.02106](https://arxiv.org/abs/2606.02106) — frozen cross-modality transfer feasibility.

**Novelty framing (cite as the gap we fill):**
- SensorLLM [2410.10624], UniMTS [2410.19818], SensorLM [2506.09108], X-Fi [2410.10167] — position our work as **frozen multi-FM embeddings + physics fusion, single-user→multi-user, location-robust, no-temporal-smoothing** — a combination none of these target for SHL transport modes.

---

## Immediate next step
First cell of `03_fm_embeddings.ipynb` = **load-and-shape gate**: for each of the 6, load weights, feed one real SHL window (handle length/channel adaptation per the gotchas), assert a finite embedding comes out. Drop any that fail before spending GPU on full caching.
</content>
