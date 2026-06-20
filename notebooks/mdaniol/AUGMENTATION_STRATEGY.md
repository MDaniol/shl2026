# SHL 2026 — Data Augmentation Strategy (IMU / Transport-Mode)

**Date**: 2026-06-20 · Author: HAR-FM scientist · Status: decision-oriented, literature-grounded.
Companion to `STRATEGY.md`, `feature_extraction/FEATURES_TABLE.md`.

## 0. Our setting (the constraints that decide everything)
- **Input**: 9-ch smartphone IMU (Acc/Gyr/Mag x/y/z), 5 s @ 100 Hz = 500-sample windows, 8 transport modes, user-independent, **macro-F1**.
- **Splits**: Train = User 1 (4 locations: Bag/Hips/Torso/Hand); val/test = Users 2&3; **test has only 3 locations (no Hand)** and is **per-window shuffled**. Dominant shift = phone **ORIENTATION + POSITION + USER**.
- **Frozen-FM rule**: FMs are frozen; only lightweight heads train. So augmentation can only act by: **augment raw window → (a) recompute handcrafted features, and/or (b) push through the FROZEN FM → embeddings → train head on augmented+original**.
- **Branch asymmetry (the crux)**: Branch A handcrafted features are **per-sensor magnitude** → already rotation-invariant. Rotation aug is therefore **near-useless for A** but **load-bearing for the raw-axis FM branch B** and for the fusion head. Amplitude/noise/time aug help **both** branches.

This single asymmetry drives the whole recipe: **apply rotation aug only to the raw-axis FM branch; apply amplitude/time/noise aug to both.**

---

## 1. Per-augmentation evidence table (ranked for locomotion/transport)

Legend — **Effect**: expected sign on macro-F1 in *our* setting. **Branch**: A=handcrafted-magnitude, B=raw-axis FM embeddings, both. **Verdict**: KEEP / CONDITIONAL / SKIP.

| Rank | Augmentation | Evidence (verified) | Expected effect (macro-F1) | Suggested params | Branch | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Random 3D rotation (SO(3)) of raw axes** | Um 2017 (ICMI'17): rotation+permutation 77.52%→**86.88%** acc; rotation = best single cross-domain aug. UniMTS (NeurIPS 2024, 2410.19818): per-joint per-timestep random rotation makes model **orientation-agnostic**, driving large cross-position zero/few-shot gains. | **Strong + for B** (test orientation unknown). **~0 for A** (magnitude invariant). | full random SO(3) per window (uniform on sphere). For *plausible* variant: yaw-dominant + small tilt (see §3). | **B only** | **KEEP (B)** |
| 2 | **Jitter / Gaussian noise** | Standard HAR aug; in Um 2017 jitter alone *hurt* PD (confused with tremor) — task-specific. For transport, mild noise = robustness to sensor/user variation. Biobehavioral eval (Yang/Yu/Sano 2022, 2210.06701): jitter among top-2 on some datasets, neutral on others. | **Small + / neutral**; cheap regularizer. Risk: too-high σ erases vibration cue (our key feature). | **σ = 0.01–0.05 × per-channel std**; keep small. | both | **KEEP (small σ)** |
| 3 | **Amplitude / magnitude scaling** | Yang 2022: scaling top-performer on several wearable/ECG sets. Simulates user mass / device-coupling differences. Preserves spectral *shape* (our ratio features). | **+** (cross-user robustness). | scale ~ N(1, 0.1), clip [0.8, 1.2] per window (optionally per-channel). | both | **KEEP** |
| 4 | **Time-warp (smooth)** | Um 2017: part of the rotation+perm+time-warp combo that generalized best. Simulates cadence/speed variation across users/vehicles. | **+ / small +**; helps cadence-sensitive classes (Walk/Run/Bike). | cubic-spline warp, 4 knots, warp σ=0.1–0.2. | both | **KEEP** |
| 5 | **Magnitude-warp (smooth amplitude envelope)** | Um 2017 family; multiplies signal by a smooth random curve → models slow gain drift. | **small +**; complements scaling. | cubic-spline curve, 4 knots, σ=0.1–0.2. | both | **CONDITIONAL** (add if 2-4 plateau) |
| 6 | **Physics-plausible aug (PPDA, WIMUSim)** | Mucha-style PPDA (2508.13284): simulate placement offset **U[−25°,25°]**, movement amplitude/speed, hardware bias U(−1,1)+noise via a body-motion model. **+3.7 pp avg macro-F1, up to +13 pp** (REALDISP/REALWORLD/MM-Fit); biggest gains in **low-subject regime** (≈ our 1-train-user setting!). | **Potentially largest +** for our 1-user-train problem, *but* requires skeleton/motion model → high build cost; SHL is phone-in-pocket (no joint model). | placement offset U[−25°,25°]; speed/amp ±10–20%. | B (raw) | **CONDITIONAL** (high value, high cost — only if time) |
| 7 | **Test-Time Augmentation (TTA)** | Established to improve robustness by averaging softmax over augmented copies (vision-origin; applied to HAR/TS). No SHL-specific macro-F1 number verified. | **small +**, low risk; directly attacks unknown test orientation. | average over **K=4–8** rotated(+mild-noise) copies of each test window; mean softmax. | B + head | **KEEP (cheap, test-time)** |
| 8 | **Mixup (input-space) / Manifold-mixup** | Mixup adapted to TS/HAR improves accuracy & macro-F1 over basic aug (IEEE 2025 transformer-HAR; Yang 2022). Manifold-mixup applicable **on embeddings** (works with frozen FM!). | **small +**; label-smoothing-like, helps imbalance (Run=4.3%). | α=0.2 Beta; for frozen FM, **mix embeddings** at head input. | head (on A+B embeddings) | **CONDITIONAL** (nice for imbalance) |
| 9 | **Permutation (segment shuffle)** | Um 2017: useful *combined* with rotation, but destroys temporal order → risky for cadence/periodicity features (our `time_ac_*`, `freq_peak_freq`). | **neutral/−** for our feature set; can hurt periodicity cues. | n_segments=4–5 if used at all. | — | **SKIP** (conflicts with periodicity features) |
| 10 | **Axis permutation / swap** | Crude orientation proxy; subsumed by proper SO(3) rotation and *less* physical. | redundant with #1. | — | B | **SKIP** (use rotation instead) |
| 11 | **Channel dropout (zero a channel)** | Models missing-modality (SHL'24 had random missing modality). 2026 test has all 3 modalities present but **fewer locations**, not missing channels. | **neutral** for 2026 (no modality dropout in test). | — | — | **SKIP** (not the 2026 failure mode) |
| 12 | **Sensor/window-warp, cropping** | Generic TS aug; window length fixed at 500 by challenge → cropping changes semantics. | neutral/−. | — | — | **SKIP** |
| 13 | **CutMix (time-series)** | Splices two windows → fabricates non-physical transitions; poor fit for window-level transport labels. | **−** risk. | — | — | **SKIP** |

**Headline ranking (keep set)**: rotation (B) > scaling ≈ jitter > time-warp > TTA > {magnitude-warp, embedding-mixup as conditionals}.

---

## 2. Rotation — the central question for SHL

**Why rotation matters here and for which branch.** The test distribution shift is dominated by unknown phone **orientation** (and position). A model that sees raw axes (Branch B / most FMs) will overfit to User-1 orientations. Rotation aug is the canonical fix (Um 2017; UniMTS 2410.19818). But our handcrafted Branch A is magnitude-based → **already rotation-invariant**, so rotation aug adds ~nothing there (it would only perturb via numerical edge effects). **Conclusion: rotation aug is a Branch-B (raw-axis FM) tool, not a Branch-A tool.**

**Gravity handling — rotate everything together, do NOT separate.** The accelerometer measures *gravity + body acceleration in the device frame*. Re-orienting the phone is mathematically a single rotation `R ∈ SO(3)` applied to the **whole 3-vector** at every timestep: `acc' = R·acc`, `gyr' = R·gyr`, `mag' = R·mag`. This is the physically correct simulation of a differently-mounted phone and it **automatically rotates the gravity component with the body component** — you must NOT high-pass out gravity, rotate, and re-add; just rotate the raw signal (UniMTS and virtual-IMU methods do exactly this single-rotation transform). **Apply the same R to Acc, Gyr, and Mag** (they share the device frame); rotating channels independently is unphysical.

**Full SO(3) vs gravity-aware ("plausible") rotation — the real trade-off.**
- *Full random SO(3)* (uniform on the sphere) maximally decorrelates orientation but generates poses a pocket/bag phone rarely holds, and **destroys the gravity-direction cue** — which is genuine signal for some modes (e.g., sustained tilt/Still vs vehicle). Verified physical fact: yaw rotation does **not** change the measured gravity vector (gravity has only 2 DoF), while roll/pitch do.
- *Gravity-aware / plausible rotation*: free **yaw about the gravity axis** + **bounded tilt** (e.g., roll/pitch within ±30–45°). This spans realistic phone poses without erasing the vertical reference.

**Recommendation**: because the test genuinely contains *arbitrary* orientations across 3 locations and a frozen FM has no orientation prior, use **full random SO(3) for Branch B** as the default (matches UniMTS's orientation-agnostic objective), and **A/B-test a gravity-aware (yaw-free + tilt ≤45°) variant** as the milder alternative. Decide by held-out macro-F1 on Users 2&3 (see §8). Do **not** assume one wins — Yang 2022 shows rotation's effect is dataset- and backbone-dependent.

**How prior SHL/teams handled it.** SHL'24 winner ("Signal Sleuths", 2407.11048): rather than augment, they used **rotation-invariant *aggregation/features*** and reported it **substantially outperformed rotation-aware features** (raw-CV macro-F1 90.01%, +tricks to 91.87%) — and that **magnitude-vector-only performed poorly** (invariance must be richer than plain magnitude). UniMTS instead bakes orientation-agnosticism in via **random per-joint per-timestep rotation during pretraining**. These are two routes to the same goal: *invariant features* (our Branch A) vs *rotation augmentation* (our Branch B). Both are legitimate; we exploit each on the branch where it fits.

---

## 3. Physics-plausible augmentation — what it adds over generic jitter/scale/rotate

PPDA (Physically Plausible DA, **arXiv:2508.13284**, evaluated on REALDISP/REALWORLD/MM-Fit) operates on an underlying **body-motion model (WIMUSim)** rather than on the signal directly, so it can vary **(1) movement amplitude, (2) movement speed → realistic peak-acceleration changes, (3) sensor placement (per-axis offset U[−25°,25°] + position), (4) hardware bias/noise** while preserving *natural motion dynamics and the gravity component*. Reported: **+3.7 pp avg macro-F1, up to +13 pp**, with the **largest gains in the low-training-subject regime** — directly analogous to our **single training user**.

**What it adds over generic aug**: generic rotate/scale/jitter produce signals that can be physically impossible (e.g., body dynamics inconsistent with the rotation). PPDA stays on the manifold of real motions, which is why it beats STDA most where data is scarcest. **Caveat for us**: PPDA needs a skeleton/joint motion model; SHL is a phone in a bag/pocket with no body-joint ground truth, so a faithful WIMUSim build is **out of scope for the 13-day timeline**. The *cheap, transferable lesson* we adopt without the full pipeline: bound placement perturbation to **±25° tilt** (their value) for the plausible-rotation variant, and combine speed-warp + amplitude-scale to mimic their movement-speed/amplitude axes.

---

## 4. Augmentation with FROZEN encoders — is augmenting FM inputs worth it?

**Mechanism.** With a frozen FM you cannot fine-tune the encoder, but you *can* run each augmented raw window through the frozen FM and train the head on the resulting (augmented) embeddings. This is **input-space augmentation for a fixed feature extractor** — equivalent to expanding the labeled set in embedding space. It is valid under the challenge rules (only the head sees gradients).

**Is it beneficial?** Reasoned position (no SHL-specific frozen-FM aug number is published — flagged as an open question):
- **Yes, if the FM is not already invariant to the augmentation.** A frozen raw-axis FM (MOMENT, MantisV2, NormWear) has *no* orientation invariance baked in; feeding it rotated copies is the only way to teach the **head** to be orientation-robust → expected **+**. This is the strongest case for aug under freezing.
- **Diminishing returns if the FM is already invariant.** UniMTS was pretrained with rotation aug → its embeddings are largely orientation-agnostic already; rotating its inputs adds little. (Test empirically per-FM.)
- **For Branch A (magnitude features)**, input rotation is wasted (invariant); only amplitude/time/noise aug change the features.

**Two ways to use aug under freezing, and when:**
1. **Embedding-set expansion** (push N augmented copies through FM, train head on all): best when the FM lacks the relevant invariance. Cost = N× FM forward passes (cache them once).
2. **Embedding-space aug** (mixup/jitter directly on cached embeddings — no extra FM passes): cheap, helps imbalance/regularization, but **cannot** create orientation invariance (rotation is non-linear in embedding space). Use as a *complement*, not a substitute, for #1.

**Practical guidance**: cache **original + a fixed set of augmented embeddings** per FM; treat "augment-or-not" and "N copies" as head-level hyperparameters chosen by val macro-F1. Augment-then-cache once (don't re-augment every epoch — too many FM passes for the timeline).

---

## 5. Test-Time Augmentation (TTA)

**What**: at inference, classify each test window K times under different augmentations (rotations ± mild noise), average the **softmax** outputs, then argmax. Directly targets the unknown test orientation without touching training.

**Evidence**: TTA is a standard robustness booster (averaging over augmented copies); broadly reported to give small consistent gains in vision and time-series classification. **No SHL/transport macro-F1 TTA number is verified** — treat the magnitude as unknown and *measure it* on Users 2&3.

**Recommendation**: **Adopt TTA for the raw-axis FM branch** with **K=4–8 random rotations (+ optional σ=0.02 jitter)**, mean-softmax fusion. It is cheap (inference-only), rule-compliant, and orientation is exactly our shift. **Pitfalls**: (i) only augment with transforms the test truly varies over (rotation: yes; time-warp: marginal; permutation: no). (ii) Use **mean of softmax/logits**, not majority vote (vote loses calibration → can hurt rare classes / macro-F1). (iii) Keep K modest — diminishing returns and K× inference cost on 92,726 windows. (iv) TTA on **magnitude Branch A is pointless** (rotation-invariant) — apply TTA only where the input transform changes the features.

---

## 6. Invariant features vs rotation augmentation — when each is redundant

| Branch | Nature | Rotation aug needed? | Why |
|---|---|---|---|
| A — handcrafted **magnitude** features | rotation-invariant by construction | **No (redundant)** | `|acc|`, PSD of magnitude etc. are unchanged by `R`. Augmenting can only add numerical noise. |
| B — raw-axis FM embeddings | **not** invariant | **Yes (necessary)** | FM sees x/y/z; only aug (or an already-rotation-pretrained FM) gives the head orientation robustness. |
| Fusion head | sees A⊕B | inherits B's need | Train head on aug-expanded B embeddings + invariant A features. |

**Caveat (from Inv2024 / 2407.11048)**: plain **magnitude-vector-only is too lossy** ("performed poorly") — richer rotation-invariant *aggregations* beat both naive magnitude and rotation-aware features. Our Branch A is already richer than plain magnitude (8 derived streams incl. body/gravity split, jerks, mag-rate), which is good; but it confirms invariant-feature design is a *real alternative* to augmentation, not just a fallback. **Strategy**: keep A invariant-by-design (no rotation aug), make B robust via rotation aug + TTA, and let fusion combine the two complementary routes.

---

## 7. RECOMMENDED RECIPE (concrete, implementable)

**Core principle: branch-specific augmentation.**

### Branch A (handcrafted magnitude → LightGBM / into fusion)
- **Rotation: NONE** (invariant — would be wasted).
- **Amplitude scaling**: per-window scale ~ N(1, 0.1), clip [0.8, 1.2]. (cross-user gain coupling)
- **Jitter**: σ = 0.01–0.03 × per-channel std (small — preserve vibration energy, our key cue).
- **Time-warp**: cubic-spline, 4 knots, σ=0.15. (cadence variation)
- **Copies**: 1–2 augmented copies per training window (start with 1; LightGBM doesn't need many).
- Recompute the 520 features on each augmented raw window.

### Branch B (raw-axis frozen FM → embeddings → head)
- **Rotation: full random SO(3)** applied identically to Acc/Gyr/Mag (default); **A/B vs gravity-aware** (free yaw + tilt ≤45°). This is the load-bearing aug.
- **Jitter**: σ = 0.02 × std.
- **Scaling**: N(1,0.1) clip [0.8,1.2].
- **Time-warp**: spline 4 knots σ=0.15 (optional).
- **Copies (embedding-set expansion)**: **N = 3–5** augmented + 1 original per window, pushed through the frozen FM, all cached. (Augment-then-cache **once**; do not re-augment per epoch.)
- Skip for FMs already rotation-pretrained (e.g., UniMTS) — test whether aug still helps; likely small.

### Fusion head
- Train on [A invariant features ⊕ B aug-expanded embeddings].
- Optional **embedding mixup** (Beta α=0.2) for class imbalance (Run=4.3%) — cheap, on cached embeddings.

### Test-time (TTA)
- Branch B + fusion: **K=6** random rotations (+ σ=0.02 jitter), **mean softmax**, then argmax.
- Branch A: no TTA (invariant).

### Parameter quick-reference
| Param | Value | Source/rationale |
|---|---|---|
| Rotation (B) | full SO(3); or yaw + tilt≤45° | UniMTS; gravity-2-DoF physics |
| Placement-offset (if PPDA-lite) | U[−25°,25°] | PPDA 2508.13284 |
| Jitter σ | 0.01–0.05 × ch-std (small) | Um 2017 (avoid masking vibration) |
| Scaling | N(1,0.1), clip[0.8,1.2] | Yang 2022 |
| Time-warp | spline, 4 knots, σ=0.15 | Um 2017 |
| Aug copies (A / B) | 1–2 / 3–5 | budget vs gain |
| TTA K | 6 (sweep 4–8) | diminishing returns |
| Mixup α | 0.2 (embedding-space) | imbalance |

### Things to explicitly SKIP
Permutation/segment-shuffle (breaks periodicity features), axis-swap (use rotation), channel/sensor dropout (test isn't missing modalities in 2026), CutMix (non-physical splices), window cropping (fixed window length), large-σ jitter (erases vibration cue).

---

## 8. Robustness-diagnostic protocol (must-run, traceable)

**Goal**: quantify each branch's macro-F1 *vs augmentation severity*, on a held-out slice that mimics the test shift — so augmentation decisions are evidence-based, not assumed.

1. **Held-out slice**: use Users 2&3 validation (the real shift). For orientation-sensitivity, additionally build a **synthetic-rotation stress set**: take val windows and apply rotations at increasing severity (e.g., tilt = 0°, 15°, 30°, 45°, 90°, full-SO(3)).
2. **Severity sweeps** (one curve per branch, macro-F1 on y-axis):
   - Rotation severity (tilt angle / full SO(3)).
   - Noise σ (0 → 0.1×std).
   - Time-warp σ (0 → 0.3).
3. **Compare**: model **trained without aug** vs **trained with the recipe** — the gap at high severity = the robustness the aug bought. Expect: Branch A curves **flat vs rotation** (confirms invariance, a sanity check — if A *drops* under rotation, there's a bug in the magnitude pipeline); Branch B should be flat **after** aug, steep **before**.
4. **Decision rules**:
   - Keep an aug only if it improves **val macro-F1** (Users 2&3) — not training acc.
   - Choose full-SO(3) vs gravity-aware rotation by which gives higher val macro-F1 on B.
   - Choose TTA K by the macro-F1 vs K curve (stop at plateau).
   - Watch **per-class F1**, especially Run (4.3%) and any vehicle-confusion pairs — aug must not trade rare-class recall for majority accuracy.
5. **Leakage guard**: compute all normalization stats on **train only**; augment **after** the train/val split; never let augmented copies of a train window leak into val. (Train=User1, val=Users2&3 already enforces subject separation — keep it.)
6. **Logging (traceability)**: log aug config (type, params, seed, n_copies), the exact commit, FM version, and per-class F1 to MLflow; tag the run so the augmentation ablation is reproducible.

---

## 9. Open questions / unverified (resolve empirically)
- **No published SHL-transport macro-F1 for FM-input augmentation** under the frozen regime → our §4 claims are reasoned, not measured. Bake-off decides.
- **No verified TTA gain number for HAR/transport** → measure on Users 2&3.
- **Full-SO(3) vs gravity-aware rotation** winner is dataset/FM-dependent (Yang 2022) → A/B-test, do not assume.
- The arXiv PDFs for 2210.06701 (biobehavioral eval) and 2508.13284 (PPDA) were only partially machine-readable in this pass; the **numbers cited (Um 86.88%; PPDA +3.7/+13 pp; placement U[−25°,25°]; SHL'24 90.01–91.87%) came through search/abstract extraction and should be re-verified against the PDFs before they go in the paper.**

---

## 10. References (verified IDs / venues)
- **Um et al. 2017** — *Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using CNNs.* ICMI '17. (Rotation+permutation 77.52%→86.88%; jitter hurt PD; rotation best cross-domain.)
- **UniMTS** — Zhang et al., *UniMTS: Unified Pre-training for Motion Time Series.* **arXiv:2410.19818**, NeurIPS 2024. (Random per-joint per-timestep rotation → orientation-agnostic.)
- **PPDA** — *Physically Plausible Data Augmentations for Wearable IMU-based HAR Using Physics Simulation.* **arXiv:2508.13284**. (WIMUSim; +3.7 pp avg, up to +13 pp; placement U[−25°,25°]; low-subject regime gains.) Related: WIMUSim, Frontiers in Computer Science 2025.
- **Yang, Yu, Sano 2022** — *Empirical Evaluation of Data Augmentations for Biobehavioral Time Series Data with Deep Learning.* **arXiv:2210.06701**. (8 aug × 7 datasets × 3 backbones; **no single best aug**; top-3 differ per dataset/backbone; rotation removes gravity/orientation/magnetic-direction info.)
- **Inv2024 / "Signal Sleuths"** — *Magnitude and Rotation Invariant Detection of Transportation Modes with Missing Data Modalities.* **arXiv:2407.11048** (SHL'24). (Rotation-invariant *aggregation* ≫ rotation-aware features; magnitude-vector-only poor; macro-F1 90.01%→91.87%.)
- **Wang et al. 2019** — *Enabling Reproducible Research … Sussex-Huawei Dataset.* IEEE Access 7:10870-10891. (SHL feature/MI study; magnitude features.)
- **SHL Challenge 2025 summaries** — Wang et al., UbiComp '25 Companion (Task 1); Okita et al., UbiComp '25 (Task 2, 10.1145/3714394.3756203). (Foundation-model edition; full-text not machine-extractable this pass — augmentation specifics to verify from PDF.)
- Mixup-for-TS-HAR (IEEE 2025, transformer HAR) and embedding/manifold-mixup — supporting evidence for §1 row 8 (effect modest; verify before citing in paper).
