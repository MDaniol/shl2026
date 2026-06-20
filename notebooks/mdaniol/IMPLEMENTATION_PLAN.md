# SHL 2026 — Implementation Plan (mdaniol)

**Owner**: M. Daniol
**Date**: 2026-06-15 · **Predictions due 30.06.2026** (freeze by **28.06** for buffer) · **HASCA paper 04.07.2026**
**Scope**: frozen foundation-model embeddings + lightweight head, with a classical handcrafted-feature safety net.
**Companion docs**: `docs/plans/hasca_2026_challenge_plan_v2_doable.md` (strategy), `hasca_2026_challenge_plan.md` (Mistral draft — literature scan only).

---

## 0. Ground truth (verified against `dataset_parquet/`)

| Item | Value |
|---|---|
| Channels | 9: `Acc_{x,y,z}`, `Gyr_{x,y,z}`, `Mag_{x,y,z}` (no GPS) |
| Window | 500 samples @ 100 Hz (5 s); Nyquist 50 Hz; FFT res 0.2 Hz |
| Train | user **1 only**, 4 locations (Bag/Hips/Torso/Hand), 196 072 windows/loc |
| Validation | users **2 & 3**, 4 locations, 28 789 windows/loc → **the user-independent oracle** |
| Test | 92 726 windows, 9 ch, **no labels, merged locations, frames shuffled** |
| Labels | per-frame 1–8, **constant within window** (0.24% mixed) → 1 label/window, replicate ×500 |
| Imbalance | class 3 (Run) ≈ 4.3%; others 12–16% |
| Metric | **macro-F1** (select/early-stop/tune on this, NOT accuracy) |
| Submission | `teamName_predictions.txt`, **92 726 × 500** int matrix |
| Rule | FMs **frozen**; only lightweight heads trainable |

**Class index → mode**: 1 Still, 2 Walk, 3 Run, 4 Bike, 5 Car, 6 Bus, 7 Train, 8 Subway.

**Hard constraints driving every decision**
1. **Single training user** → user-DANN impossible. The exploitable domain axis is **phone location** (4 train → 3 test, no Hand).
2. **Shuffled test** → window-internal features only; **no cross-window/HMM/voting smoothing** (it was the biggest gain in past SHL editions — illegal here).
3. **Orientation/location varies** → features on **per-sensor magnitude / rotation-invariant** quantities, never raw per-axis.

---

## 1. Strategy

- **Track A — Safety net (must-have):** handcrafted features → LightGBM. Guarantees a valid, ~0.6+ macro-F1 submission early.
- **Track B — Primary FM (the contribution):** frozen FM embeddings over 9 channels → cached → MLP head. **Model chosen by the Day-2 bake-off**, not pre-committed (see §FM selection below).
  - *Not Chronos/TimesFM*: forecasters, no clean classification embedding.
- **Track C — Stretch:** spectrogram → frozen vision FM (AST / DINOv2) embedding, late-fused with B. The honest novelty: frozen-FM embeddings + physics fusion, single-user→multi-user, location-robust.

Deferred to "future work": SensorLLM (needs task-aware tuning), IMU2CLIP (trained IMU encoder), IMG2IMU (custom image pretraining), X-Fi, generative/graph DA.

### FM selection — Day-2 bake-off (evidence from `LITERATURE_REVIEW.md`)
Funnel: rubric shortlist → **frozen linear-probe bake-off** on val (users 2&3) → carry winners. Score on **macro-F1** + per-location breakdown (Hand held-out as test-shift proxy). Candidates:

| Pri | Model | Year | arXiv | Why | Weights |
|---|---|---|---|---|---|
| 1 | **UniMTS** | 2024 | 2410.19818 | IMU-native; *built for* cross-location + cross-orientation shift (graph nets + rotation-invariant aug); true zero-shot frozen. | HF `xiyuanz/UniMTS` |
| 2 | **Mantis** | 2025 | 2502.15637 | Strongest frozen-encoder TSFM on UCR/UEA (0.7816); 8M ViT; channel adapter for 9-ch; len-512. | HF + `vfeofanov/mantis` |
| 3 | **LIMU-BERT** | 2021 | SenSys'21 | MIT-licensed frozen IMU encoder; 6/9-ch Acc/Gyr/Mag. | `dapowan/LIMU-BERT-Public` |
| 4 | **MOMENT** | 2024 | 2402.03885 | Frozen embeddings/linear probe. ⚠️ re-verify weights/license (HF claim refuted). | verify |

⚠️ Caveats: FM numbers are **accuracy on UCR/UEA, not SHL macro-F1** — bake-off decides. **Magnetometer** support not guaranteed in any single frozen encoder (UniMTS/LIMU-BERT are accel/gyro-centric) — confirm 9-ch ingestion or add a Mag-handling input adapter (input-side only, keeps backbone frozen).

---

## 2. Feature design (Track A) — evidence-anchored

All features on the **magnitude** of derived 1-D streams; **z-normalize each window before spectral features** (arXiv:2407.11048).

**Derived streams (~8):** Acc-magnitude, Acc-body (gravity-removed, high-pass), gravity (low-pass) magnitude, Acc-jerk, Gyr-magnitude, Gyr-jerk, Mag-magnitude, Mag-rate (dMag/dt).

**Feature bank per stream (65), from the SHL authors' MI/MRMR analysis (Wang et al., IEEE Access 2019):**
- **Subband energy + energy-ratio (42):** centers {1,2,3,4,5,10,15} Hz × bandwidths {2,5,10} Hz × {energy, ratio}. (Clamp lower edge to 0 → ~2–3 collapse; ~40 effective.) *Dominant family.*
- **Time/freq scalars (14):** peak-FFT value, peak-FFT freq (cadence), FFT mean/std, 1st/2nd-peak ratio, DC; mean-crossing rate, top-autocorr value + lag; time mean/std/energy/kurtosis/skew.
- **Quantiles (9):** {0,5,10,25,50,75,90,95,100}.

**Counts** (see companion doc for table): magnitude-only first pass **195**; with derived streams **~520**; + catch22/stream **~700**. Start at **~520**; let LightGBM importance prune. Don't run heavy MRMR.

**Why all 3 sensors (don't drop any):**
- **Mag** is decisive for Still↔rail and road↔rail (Train/Subway metal casing) — lifts Subway recall 32→54%.
- **Gyr** rescues Bike (rotational) — 76→85%.
- **Acc** owns low-freq Still/pedestrian/vehicle separation.

**Expected hard cases:** Car↔Bus, Train↔Subway (low MI for all sensors); Run (rare + confused with Walk/Bike). Macro-F1 is won/lost here.

---

## 3. Milestones (≈ 1 week)

- **M1 (Day 1) — Pipeline + safety-net submission.** Parquet loader; `features.py` (§2); global scaler fit on train; LightGBM, **validate on users 2&3**, class-weighted, macro-F1; `submit.py` → 92 726×500 writer + shape/format asserts. **Produce a real submission today.**
- **M2 (Day 2) — Cache frozen FM embeddings.** MOMENT, bf16, batched, over train/val/test → shared cache. Log to MLflow.
- **M3 (Day 3) — FM head.** MLP (LayerNorm+dropout 0.3) → 8, AdamW, class-weighted CE, early-stop on **val macro-F1**. Compare vs A.
- **M4 (Day 4) — Fusion + robustness.** Concat physics features to FM embedding; per-**location** val breakdown (esp. Hand vs the 3 test locations); confusion matrix; A+B ensemble decision.
- **M5 (Day 5) — Stretch or harden.** Track C if A+B solid; else calibration, seed averaging, freeze config.
- **M6 (Day 6) — Freeze & submit.** Regenerate test preds from frozen best; sanity-check class prior; write matrix; **submit (before 30.06).**
- **M7 (Day 7+) — Paper.** 3–6 pp ACM sigconf from MLflow runs: method, single-user/location-shift framing, ablations (physics on/off, FM vs classical, fusion), error analysis.

---

## 4. Notebook / code layout (under `notebooks/mdaniol/`)

```
notebooks/mdaniol/
├── IMPLEMENTATION_PLAN.md      # this file
├── 01_explore.ipynb            # class/location dist, window-label constancy, sanity
├── 02_features_baseline.ipynb  # features.py → LightGBM → first submission (M1)
├── 03_fm_embeddings.ipynb      # MOMENT caching (M2)
├── 04_fm_head_fusion.ipynb     # head + physics fusion + robustness (M3–M4)
└── 05_submit.ipynb             # freeze, predict, write matrix (M6)
```
Shared code lands in `src/` (`data/features.py`, `models/`, `training/`) per the v2 plan's structure; notebooks stay thin drivers. Route every run through the existing **Parquet → MLflow → embedding-cache** stack (FAIR/Zenodo target).

---

## 5. Pitfalls checklist
- [ ] Global scaler fit on **train only**; applied to val/test. No per-user/window leakage.
- [ ] **macro-F1** everywhere; class weights / focal for Run.
- [ ] **No** cross-window smoothing (shuffled test).
- [ ] Use **all 9 channels** incl. Mag.
- [ ] Heads stay lightweight; FM backbone frozen (rule compliance).
- [ ] Validate on **users 2&3** from M1.
- [ ] Drop ">92%" optimism — report what val gives (classical ceiling ~0.76–0.83 user-independent; DT baseline ~0.55–0.63).

---

## 6. Key references
- Wang et al., *Enabling Reproducible Research… SHL* (IEEE Access 2019) — feature MI/MRMR, magnetometer finding, UI baselines.
- JSI, *What Actually Works… 2019/2020 SHL Challenges* (Sensors 2022, PMC9145859) — derived streams, 1124-feature bank, domain-shift tactics (HMM N/A here).
- *Magnitude & Rotation-Invariant Detection, missing modalities* (arXiv:2407.11048) — z-norm, rotation-invariant aggregation.
- MOMENT (ICML 2024); SensorLLM (EMNLP 2025, deferred); X-Fi (arXiv:2410.10167, deferred).
- catch22 / tsfresh — optional feature expansion.
</content>
