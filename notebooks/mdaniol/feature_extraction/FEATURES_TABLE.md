# SHL 2026 — Features, Decisions, Justifications & Literature

Single consolidated table. 8 derived streams × 65 features = **520 features/window**.
Reference keys defined at the bottom.

## Derived streams (the 8 signals features are computed on)
| Stream | Definition | Justification | Reference |
|---|---|---|---|
| `acc_mag` | √(aₓ²+aᵧ²+a_z²) | orientation-invariant accel intensity | Wang2019; Inv2024 |
| `acc_body_mag` | \|acc − gravity\| (high-pass) | separates motion from posture; vibration cue | Anguita2013 |
| `gravity_mag` | \|low-pass(acc)\| | posture/tilt; sustained linear accel | Anguita2013 |
| `acc_jerk_mag` | \|d·acc/dt\| | harsh dynamics (vehicle stop-go) | Anguita2013 |
| `gyr_mag` | \|gyro\| | rotation intensity (Bike turns) | Wang2019 |
| `gyr_jerk_mag` | \|d·gyro/dt\| | rotational transients | Anguita2013 |
| `mag_mag` | \|magnetic field\| | rail vs road discriminator | Wang2019 |
| `mag_rate_mag` | \|d·mag/dt\| | electric-traction magnetic transients (Train/Subway) | Wang2019 (Mag↔rail); JUDGMENT |

## Preprocessing decisions
| Decision | Value | Justification | Reference |
|---|---|---|---|
| Feature domain | per-sensor **magnitude** (not raw axes) | unknown phone orientation/location at test | Wang2019; Inv2024 |
| Gravity/body split | Butterworth low-pass, **0.3 Hz**, order 4 | 0.3 Hz found optimal for gravity | Anguita2013 |
| Filter application | zero-phase `sosfiltfilt` (SOS form) | no time shift; SOS stable at low cutoff | Winter; SciPy |
| Derivatives | central difference (`np.gradient`)×fs | 2nd-order accurate | DSP standard |
| Detrend before FFT | remove window mean (DC) | stops gravity offset dominating low bands | DSP standard |
| Amplitude scaling | **mean-removal only** (z-score = opt-in `--zscore`) | preserve discriminative vibration energy | Wang2019; cf. Inv2024 (JUDGMENT) |
| Spectrum | single-window **one-sided PSD** (Parseval-exact), rectangular window | full 0.2 Hz resolution for low subbands; energy-preserving | Welch1967; Harris1978 (JUDGMENT: taper) |
| Per-window label | majority vote of 500 frames | windows 99.8% constant (measured) | JUDGMENT |
| No cross-window features | window-internal only | test frames are shuffled | SHL rules; JSI2022 |

## Time-domain features (per stream — 8)
| # | Feature | Definition | Justification | Reference |
|---|---|---|---|---|
| 1 | `time_mean` | mean | central tendency / level | Wang2019 |
| 2 | `time_std` | std | dispersion (motion intensity) | Wang2019 |
| 3 | `time_energy` | mean(x²) | signal power (still vs active) | Wang2019 |
| 4 | `time_kurtosis` | Fisher excess, unbiased | impulsiveness/peakedness | Wang2019 |
| 5 | `time_skew` | unbiased G1 | asymmetry | Wang2019 |
| 6 | `time_mcr` | mean-crossing rate | oscillation frequency proxy | Wang2019 |
| 7 | `time_ac_val` | top positive autocorr value | periodicity strength (cadence) | Wang2019; BoxJenkins |
| 8 | `time_ac_lag` | lag of that peak | period of motion (gait) | Wang2019 |

## Quantile features (per stream — 9)
| # | Feature | Definition | Justification | Reference |
|---|---|---|---|---|
| 9–17 | `q{0,5,10,25,50,75,90,95,100}` | percentiles | robust amplitude distribution | **Wang2019 Table 9 (exact set)** |

## Frequency-domain scalar features (per stream — 6)
| # | Feature | Definition | Justification | Reference |
|---|---|---|---|---|
| 18 | `freq_dc` | \|mean\| (DC component) | offset/level | Wang2019 |
| 19 | `freq_peak_val` | max one-sided power (excl. DC) | dominant component strength | Wang2019 |
| 20 | `freq_peak_freq` | frequency of peak | **cadence** (Walk≈1.5–2.5, Run≈2.5–3.5, Bike≈1–2 Hz) | Wang2019 |
| 21 | `freq_fft_mean` | mean of magnitude spectrum | spectral level | Wang2019 |
| 22 | `freq_fft_std` | std of magnitude spectrum | spectral spread | Wang2019 |
| 23 | `freq_peak_ratio` | 1st/2nd peak power ratio | tonal vs broadband | Wang2019 |

## Subband features (per stream — 42 = 7 centers × 3 bandwidths × {energy, ratio})
| # | Feature | Definition | Justification | Reference |
|---|---|---|---|---|
| 24–44 | `sb_e_fc{1,2,3,4,5,10,15}_bw{2,5,10}` | band power | absolute vibration energy per band (amplitude cue) | Wang2019 |
| 45–65 | `sb_r_fc…_bw…` | band power / total | amplitude-invariant spectral shape | Wang2019 |
| — | grid choice (centers 1–15 Hz, bw 2/5/10) | subset of Wang's sweep | MI peaks at 0–15 Hz | Wang2019 (JUDGMENT: exact grid) |

## Task-level constraints (context for all choices)
| Constraint | Implication for features | Reference |
|---|---|---|
| User-independent (train = 1 user) | no user-DANN; rely on invariant features | SHL three-year review |
| Test missing Hand / shuffled | location-robust, window-internal only | SHL rules; JSI2022 |
| Ranking = macro-F1 (Run = 4.3%) | features must serve rare-class separation | Wang2019 |

---

## References
- **Wang2019** — Wang, Gjoreski, Ciliberto, Mekki, Valentin, Roggen. *Enabling Reproducible Research in Sensor-Based Transportation Mode Recognition With the Sussex-Huawei Dataset.* IEEE Access 7 (2019) 10870–10891. (SHL authors' feature MI/MRMR study on these 8 classes.)
- **Anguita2013** — Anguita, Ghio, Oneto, Parra, Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013. (0.3 Hz Butterworth gravity/body split; jerk.)
- **JSI2022** — Janko et al. *What Actually Works for Activity Recognition in Scenarios with Significant Domain Shift: Lessons from the 2019/2020 SHL Challenges.* Sensors 22(10):3613, 2022. (derived streams; HMM smoothing unavailable when shuffled.)
- **Inv2024** — *Magnitude and Rotation Invariant Detection of Transportation Modes with Missing Data Modalities.* arXiv:2407.11048 (SHL'24). (magnitude/rotation invariance; z-norm.)
- **Harris1978** — Harris. *On the Use of Windows for Harmonic Analysis with the DFT.* Proc. IEEE 66(1), 1978. (spectral leakage / window trade-off.)
- **Welch1967** — Welch. *The Use of FFT for Estimation of Power Spectra.* IEEE Trans. Audio Electroacoust. 15(2), 1967.
- **BoxJenkins** — Box, Jenkins, Reinsel. *Time Series Analysis: Forecasting and Control.* (biased autocorrelation estimator.)
- **Winter** — Winter. *Biomechanics and Motor Control of Human Movement.* Wiley. (zero-phase filtering of movement signals.)
- **SHL three-year review** — Wang et al. *Three-Year Review of the 2018–2020 SHL Challenge.* Frontiers in Computer Science, 2021.

**Tags:** *JUDGMENT* = our choice, literature-open → validate by ablation (see `DESIGN_DECISIONS.md` §"Can these be validated"). All non-tagged rows are literature- or DSP-standard and implementation-verified by `validate_features.py`.
</content>
