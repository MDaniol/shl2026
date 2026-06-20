# SHL 2026 Feature Extraction — Design Decisions & Literature Grounding

Every parameter and definition in `shl_features.py`, with its rationale, source,
and validation status. Goal: each choice is traceable to the scientific
literature (ideally the SHL dataset authors' own work) or to a DSP standard —
not chosen by feel.

**Grounding categories**
- **[FIXED]** — imposed by the dataset/challenge, not a choice.
- **[LIT]** — definition/value taken directly from peer-reviewed literature.
- **[DSP]** — standard signal-processing best practice.
- **[JUDGMENT]** — our defensible decision; flagged, with how to validate it.

**Key references**
- **[Wang2019]** Wang, Gjoreski, Ciliberto, Mekki, Valentin, Roggen. *Enabling
  Reproducible Research in Sensor-Based Transportation Mode Recognition With the
  Sussex-Huawei Dataset.* IEEE Access, 2019. ← the SHL authors' own feature study
  (MI/MRMR analysis on these exact 8 classes).
- **[Anguita2013]** Anguita et al. *A Public Domain Dataset for HAR Using
  Smartphones.* ESANN 2013. ← gravity/body split, jerk.
- **[JSI2022]** Janko et al. *What Actually Works for Activity Recognition…
  Lessons from the 2019/2020 SHL Challenges.* Sensors, 2022.
- **[Inv2024]** *Magnitude and Rotation Invariant Detection of Transportation
  Modes with Missing Data Modalities.* arXiv:2407.11048 (SHL'24).
- **[Harris1978]** Harris. *On the Use of Windows for Harmonic Analysis with the
  DFT.* Proc. IEEE, 1978. — **[BoxJenkins]** Box & Jenkins, *Time Series
  Analysis.* — **[Winter]** Winter, *Biomechanics and Motor Control of Human
  Movement.* — **[Welch1967]** Welch, PSD estimation.

---

## A. Inputs (not choices)
| # | Item | Value | Cat | Source |
|---|---|---|---|---|
| 1 | Window length | 500 samples / 5 s | [FIXED] | SHL archive; also typical HAR ([Wang2019] uses 5.12 s) |
| 2 | Sampling rate | 100 Hz (Nyquist 50, FFT res 0.2 Hz) | [FIXED] | SHL archive |
| 3 | Channels | 9: Acc/Gyr/Mag ×xyz | [FIXED] | SHL archive |
| 4 | Output granularity | one label/window (majority of 500 frames) | [JUDGMENT] | windows 99.8% constant (measured); standard SHL practice |

## B. Orientation/location invariance
| # | Choice | Value | Cat | Grounding |
|---|---|---|---|---|
| 5 | Features on per-sensor **magnitude**, never raw axes | √(x²+y²+z²) | [LIT] | [Wang2019] computes features on the magnitude of each sensor; [Inv2024] shows rotation-invariant aggregation needed for SHL cross-position/orientation |
| 6 | No cross-window features (window-internal only) | — | [FIXED] | test frames are **shuffled** → temporal smoothing impossible; [JSI2022] notes HMM smoothing (a past top trick) is unavailable here |

## C. Derived streams (8)
| # | Stream | Definition | Cat | Grounding |
|---|---|---|---|---|
| 7 | acc_mag | \|acc\| (with gravity) | [LIT] | [Wang2019] |
| 8 | gravity_mag / acc_body_mag | low-pass(acc) and acc − gravity, magnitudes | [LIT] | [Anguita2013] gravity/body separation |
| 9 | acc_jerk_mag | \|d acc/dt\| | [LIT] | [Anguita2013] jerk signals |
| 10 | gyr_mag, gyr_jerk_mag | \|gyro\|, \|d gyro/dt\| | [LIT] | [Anguita2013] (gyro jerk), [Wang2019] (gyro magnitude) |
| 11 | mag_mag | \|magnetic field\| | [LIT] | [Wang2019] (magnetometer magnitude) |
| 12 | mag_rate_mag | \|d mag/dt\| | [JUDGMENT] | motivated by rail/electric-traction magnetic transients ([Wang2019] finds Mag discriminates rail vs road; rate emphasizes transitions) — validate by ablation/importance |
| 13 | "virtual streams × feature bank" pattern | apply the same bank to each stream | [LIT] | [JSI2022] (derives extra streams, computes features on each) |

## D. Gravity filter
| # | Choice | Value | Cat | Grounding |
|---|---|---|---|---|
| 14 | Low-pass cutoff | **0.3 Hz** | [LIT] | [Anguita2013] found 0.3 Hz optimal for the gravity component |
| 15 | Filter type/order | Butterworth, order 4 | [LIT/DSP] | [Anguita2013] Butterworth (3rd order); 3–4 standard |
| 16 | Zero-phase application | `sosfiltfilt` (forward-backward, SOS form) | [DSP] | [Winter] zero-phase avoids time shift; SciPy recommends SOS for low-cutoff stability (verified via `sosfreqz`) |

## E. Spectral estimation
| # | Choice | Value | Cat | Grounding |
|---|---|---|---|---|
| 17 | Detrend before FFT | remove window mean (DC) | [DSP] | standard; prevents the 0-Hz/gravity offset dominating low subbands |
| 18 | Amplitude normalization | **mean-removal only**, not full z-score (z-score = `--zscore` ablation) | [JUDGMENT] | [Inv2024] used z-norm (missing-modality setting); we keep amplitude because vibration **energy** is class-discriminative ([Wang2019] uses absolute subband energy). Validate by `--zscore` ablation |
| 19 | Spectrum | single-window periodogram, **one-sided PSD** (Parseval-exact) | [DSP] | full-window needed for 0.2 Hz resolution at low subbands (Welch sub-segmenting would destroy it, [Welch1967]); one-sided/Parseval standard |
| 20 | Window taper | **rectangular** (no Hann) | [JUDGMENT] | trade-off: rectangular preserves energy (Parseval) & resolution; Hann reduces leakage [Harris1978] but distorts energy. [Wang2019] uses FFT directly. Bandwidths ≥2 Hz (≥10 bins) contain most leakage. Validate by Hann ablation if needed |

## F. Frequency-domain features
| # | Feature | Cat | Grounding ([Wang2019] Table 3 unless noted) |
|---|---|---|---|
| 21 | Subband **energy + energy-ratio** | [LIT] | [Wang2019] uses both per subband |
| 22 | Subband grid: centers {1,2,3,4,5,10,15} Hz × bw {2,5,10} Hz | [LIT/JUDGMENT] | [Wang2019] sweeps bw {1,2,3,4,5,10,15,20,25}; MI peaks at **0–15 Hz** → we use that high-MI subset. Exact grid is our reduction (ablatable) |
| 23 | Peak (dominant) frequency + peak power | [LIT] | [Wang2019] "frequency with highest FFT value" |
| 24 | 1st/2nd FFT peak ratio | [LIT] | [Wang2019] "ratio between 1st and 2nd highest FFT peaks" |
| 25 | FFT mean, FFT std | [LIT] | [Wang2019] "mean and std of FFT coefficients" |
| 26 | DC component | [LIT] | [Wang2019] "DC of FFT" |

## G. Time-domain features
| # | Feature | Cat | Grounding |
|---|---|---|---|
| 27 | mean, std, energy | [LIT] | [Wang2019] Table 3 |
| 28 | kurtosis, skewness (**unbiased**, Fisher) | [LIT/DSP] | [Wang2019]; bias-corrected G1/G2 estimators standard |
| 29 | mean-crossing rate | [LIT] | [Wang2019] "mean crossing rate" |
| 30 | top autocorrelation value + lag | [LIT] | [Wang2019] "highest autocorrelation value and offset" |
| 31 | autocorr estimator: **biased**, normalized; **positive** local-max selection | [DSP/JUDGMENT] | biased estimator is standard ([BoxJenkins], statsmodels default); positive-peak requirement is our refinement (periodicity ⇒ positive correlation) — validated vs statsmodels |
| 32 | Quantiles {0,5,10,25,50,75,90,95,100} | [LIT] | **exact set in [Wang2019] Table 9** |

## H. Engineering
| # | Choice | Value | Cat | Grounding |
|---|---|---|---|---|
| 33 | Derivative method | central difference (`np.gradient`), ×fs | [DSP] | 2nd-order accurate; standard for jerk |
| 34 | Degenerate/NaN → 0 | finite guard | [JUDGMENT] | FM inputs must be finite; degenerate windows (zero-variance) have undefined moments → 0 is meaningful (no tailedness / no periodicity) |
| 35 | float32 storage | — | [JUDGMENT] | sufficient for ML; ~1e-7 vs float64 (documented) |

---

## Can these be validated against the literature?

**Yes for the definitions.** Items marked [LIT] (the large majority: streams,
gravity split, jerk, the full time/frequency feature set, quantile set, subband
energy+ratio, peak/cadence features) come **directly from [Wang2019]** — the SHL
dataset creators' own feature paper analyzing these exact 8 classes — plus
[Anguita2013] for the gravity/jerk pipeline. Items marked [DSP] are textbook
standards. This is about as well-grounded as a feature set for this dataset can be.

**The [JUDGMENT] items cannot be settled by literature alone** — they are
parameter/variant choices the literature leaves open. The honest way to validate
*those* is an **ablation in the modeling phase** (each is a one-line switch):
1. mean-removal vs full z-norm (#18) → `--zscore`
2. subband grid (#22) → vary centers/bandwidths
3. rectangular vs Hann taper (#20)
4. mag_rate stream (#12) → leave-one-stream-out importance
5. gravity cutoff/order (#14–15) → sweep 0.2–0.5 Hz

**Bottom line:** the feature *definitions* are literature-grounded and
implementation-verified (`validate_features.py`); the handful of free *parameters*
are flagged here and should be confirmed empirically by ablation, not asserted.
</content>
