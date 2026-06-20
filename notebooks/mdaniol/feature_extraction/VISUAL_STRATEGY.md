# SHL 2026 — Visual-Representation Strategy (frozen vision FMs on IMU images)

Source: deep-research, 23 sources, **22/25 claims confirmed, 3 refuted**. Question:
*would IMU→image→frozen vision FM beat time-series FMs / handcrafted, and how?*

## Verdict
**Yes — viable, novel, and evidence-backed, but as a COMPLEMENTARY branch, not a
replacement.** Every unanimous result shows vision features **win via FUSION**, not
standalone dominance. Build it, but **prioritize it below the handcrafted + TS-FM
baseline.**

## Strongest evidence
- **TiViT** (arXiv:2506.08641, NeurIPS 2025): time-series→image → **frozen OpenCLIP ViT** + trainable linear head → **SOTA on UCR (81.3 > MOMENT 79.0, Mantis 80.1)**; **fusion is best: TiViT+Mantis 83.0 (+3% avg)**. Code public (`ExplainableML/TiViT`). This is the *direct* frozen-FM-on-images proof. (Encoding = segment-stacked grayscale, not spectrogram; not tested on SHL → extrapolation.)
- **"From Images to Signals: Are LVMs useful for TS?"** (arXiv:2505.24030, 2025): first large principled study — **LVMs are useful for TS classification** (frozen encoder + linear probe), struggle on forecasting. Matches our frozen+head constraint.
- **SHL-specific** (PMC9460376, Sensors 2022): ResNet+ViT on **spectrum/CWT images recognizes all 8 SHL transport modes** (ViT alone 81%, full 93%). ⚠️ trained end-to-end, **not frozen** — proves images work on SHL, not that *frozen* works.
- **IMG2IMU** (arXiv:2209.00945): spectrograms transfer vision knowledge, **strongest in low-data**; ⚠️ uses custom pretraining, not off-the-shelf frozen.
- **AST** (arXiv:2104.01778): ImageNet→spectrogram works, ⚠️ via **fine-tune+init**, and normalizes **mean 0/std 0.5, NOT ImageNet stats**.

## Encoding ranking (for our signals)
1. **Spectrogram / log-mel / CWT** — best-supported for locomotion/transport (IMG2IMU, PMC9460376, 2026 survey: time-frequency "especially salient for locomotion").
2. **Segment-stacked grayscale (TiViT recipe)** — exact published frozen-FM recipe, lowest implementation risk.
3. **GAF/GADF, MTF, recurrence plots** — validated encodings but weaker/contradicted ranking. *(Refuted: "GAF best / line-plot worst", and "per-axis RGB recurrence-plot stacking wins".)*

## Frozen vision FM ranking
**Large OpenCLIP ViT** (TiViT-proven) → **DINOv2** (strong self-supervised linear-probe) → AST (spectrogram/audio-native) → CLIP/SigLIP. All public on HuggingFace.

## Preprocessing (mandatory vs harmful)
- 9 channels → image: **per-channel grayscale montage** or **RGB-group Acc/Gyr/Mag** (3 images). Open question which wins — ablate.
- Resize to the encoder's input (e.g., 224×224); **3-channel** expectation met by RGB-stacking or channel-averaging.
- **Do NOT blindly apply ImageNet normalization to spectrograms** — match the encoder/domain (AST used mean0/std0.5); verify empirically.
- Budget a **trainable MLP/adapter head** (not just linear) to partially close the natural-image→spectrogram domain gap.

## Why it's a good *novelty* bet
The 2026 sensor-HAR survey (arXiv:2604.02711) **deliberately excludes** vision-based pipelines, and 2025 SHL only used Flamingo → **frozen OpenCLIP/DINOv2 on IMU spectrograms is comparatively novel** for the challenge.

## Risks (be honest)
- **No direct evidence** of frozen-vision-FM-on-spectrogram on *SHL transport, 9-ch, user-independent, shuffled*. Pipeline is plausible-by-composition, not proven on SHL.
- **Domain gap**: frozen natural-image features can underperform on spectrograms; a trainable head only partially fixes it.
- Reported margins are often **1–3 pts**; fusion gains are **averages**, not per-dataset guarantees.
- **Vision-language models** (LLaVA/Qwen-VL/Flamingo on spectrograms, frozen): **zero confirming evidence** → speculative; not a 2-week primary bet (small exploratory at most, for novelty).
- Complementary hedge: TSFMs can **fail under spectral shift** (arXiv:2511.05619) — a vision branch diversifies that risk.

## Recommended visual branch (if/when built)
**Spectrogram (per-channel + RGB-grouped) → frozen large OpenCLIP ViT (+ DINOv2 as alt) → trainable MLP head → fuse (concat) with TS-FM embeddings + handcrafted features.** Priority: **after** Track A + Track B (see `STRATEGY.md`).
</content>
