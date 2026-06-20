# Feature Design — Re-validation Against Newest Literature (2024–2026)

Re-checked the feature choices in `FEATURES_TABLE.md` / `DESIGN_DECISIONS.md`
against 2024–2026 sources. **No contradictions found.** Verdicts below:
✅ Confirmed · ➕ Refinement opportunity · ⚠️ Judgment call now leans differently.

## Most relevant new sources
- **[SHL2024sum]** *Summary of SHL Challenge 2024* (ACM UbiComp/ISWC 2024, doi:10.1145/3675094.3678456) — most recent SHL edition; same 9-ch, 5 s, shuffled, position-independent setup as 2026.
- **[SignalSleuths]** *Magnitude and Rotation Invariant Detection of Transportation Modes with Missing Data Modalities* — arXiv:2407.11048 (**SHL 2024 winning approach**).
- **[TimesFMfusion]** *Multiresolution Sensor Fusion with TimesFM* (UbiComp/ISWC 2025, doi:10.1145/3714394.3756212) — fuses **handcrafted features + FM embeddings**, uses **MOMENT-1-large**.
- **[NatSciData2026]** *A comprehensive IMU dataset for evaluating sensor layouts…* Nature Scientific Data 2026 (s41597-026-06710-9) — systematic HAR feature engineering.
- **[GenHAR2026]** *GenHAR: Generalizing Cross-domain HAR* (KDD 2026); **[TimeToEmbed]** arXiv:2505.14543 (multivariate FM embeddings via channel descriptions).

## Verdict by decision
| Design choice | Newest-literature verdict | Evidence |
|---|---|---|
| Handcrafted features + tree ML (LightGBM/XGBoost) as the classical track | ✅ **Confirmed** | SHL 2024 winner used handcrafted + traditional ML; TDU_BSA hit F1 82.5% with XGBoost/LightGBM [SHL2024sum][SignalSleuths] |
| Per-sensor **magnitude / rotation-invariant** basis | ✅ **Confirmed (strongly)** | SHL 2024 winner: "rotation-invariant aggregation demonstrated substantial improvement over rotation-aware features" [SignalSleuths] |
| Time-domain set (mean, std, energy, kurt, skew, MCR, autocorr) | ✅ Confirmed | 2025–26 reviews list exactly these as standard [NatSciData2026] |
| Frequency set (dominant freq, FFT mean/std, peak ratio, subband energy) | ✅ Confirmed | standard freq features in current reviews [NatSciData2026] |
| Quantiles / IQR | ✅ Confirmed | IQR/quantiles still standard time-domain features |
| Magnetometer for rail vs road | ✅ Confirmed | remains a key cue in SHL 2024 (Mag one of 3 core modalities) |
| **Handcrafted ⊕ FM-embedding fusion** (our whole Track A+B plan) | ✅ **Confirmed (directly)** | "adding handcrafted features further improved performance, highlighting their complementarity" with FM embeddings (MOMENT) [TimesFMfusion] |
| Spectrogram → 2D model, **log-frequency** (Track C) | ✅ Confirmed | log-freq 2D spectrogram beats 1D temporal for transport-mode [data-fusion case study] |
| **Spectral centroid / spectral entropy** | ➕ **Add** | repeatedly cited as standard freq features (incl. minimal sets); we don't have them yet [NatSciData2026] |
| **SMA, RMS, MAD, median/mean frequency** | ➕ Optional add | common in current time/freq sets [NatSciData2026] |
| **z-normalization before spectral** (we default to mean-removal) | ⚠️ **Reconsider** | SHL 2024 winner found **"z-normalization proved crucial for robust spectral features"** [SignalSleuths] — newest SHL-specific evidence favors z-norm |
| Rectangular window (no taper) | ✅ Acceptable | no 2024–26 source flags this for HAR energy features |

## Actions arising
1. ⚠️ **z-normalization** — the most recent SHL-specific evidence (the 2024 winner, on a near-identical task) calls it *crucial*. **Promote from "ablate later" to "ablate first," and consider defaulting to z-norm.** Already a one-flag switch (`--zscore`); the bake-off should test both, with z-norm as the favored hypothesis now.
2. ➕ **Add spectral centroid + spectral entropy** per stream (and optionally SMA/RMS/MAD/median-freq). Cheap, literature-standard, would take the bank from 65 → ~67–70 features/stream. Implement behind the existing structure + re-run `validate_features.py` (add tsfel `spectral_centroid` / `spectral_entropy` as references — already importable).
3. ✅ **Whole-pipeline strategy validated**: handcrafted features fused with frozen-FM (MOMENT) embeddings is exactly what [TimesFMfusion] 2025 shows improves HAR — our Track A+B is current best practice, not legacy.

## Bottom line
Nothing in the 2024–2026 literature contradicts the feature design; the SHL **2024 winner used the same magnitude/rotation-invariant handcrafted + tree-ML recipe**, and 2025 work confirms **handcrafted ⊕ FM-embedding fusion** as state of practice. Two cheap refinements (spectral centroid/entropy) and one re-prioritized judgment call (z-normalization) are the only changes indicated — all additive, none invalidating.
</content>
