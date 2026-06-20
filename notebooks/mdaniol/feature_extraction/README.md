# SHL 2026 — Feature Extraction

Computes the handcrafted feature bank from `IMPLEMENTATION_PLAN.md §2`:
**8 derived 1-D streams × 65 features = 520 features per 5 s window.**
Input: `dataset_parquet/<split>/<location>.parquet` → output:
`dataset_parquet_features/<split>/<location>.parquet` (520 float32 columns +
`label` int8 for train/validation; test has no label).

## Files
| File | Purpose |
|---|---|
| `shl_features.py` | Core vectorized feature library (+ `--self-test` DSP checks) |
| `validate_features.py` | **Differential validation** — every feature vs an independent library |
| `extract_features.py` | Per-file runner; streams chunks to a feature Parquet |
| `extract_features.sbatch` | SLURM **array** job (one task per file, 9 tasks) |
| `run_all_local.sh` | Sequential local run over all 9 files |
| `setup_env.sh` | Create the `SHL-local` uv venv + install + self-test |
| `requirements.txt` | Pinned-ish deps |

## Quick start
```bash
# 1. environment (creates ../../../SHL-local)
bash setup_env.sh
source ../../../SHL-local/bin/activate

# 2. verify the math
python extract_features.py --self-test     # synthetic DSP checks
python validate_features.py --n 600        # vs scipy/numpy/statsmodels/tsfel on real data

# 3a. local: all 9 files (~10 min)
bash run_all_local.sh

# 3b. cluster: SLURM array (edit #SBATCH --account/--partition first)
sbatch extract_features.sbatch
```

## The 520 features (per window)
**8 streams** (all magnitudes → orientation/location invariant):
`acc_mag, acc_body_mag` (gravity-removed), `gravity_mag` (low-pass),
`acc_jerk_mag, gyr_mag, gyr_jerk_mag, mag_mag, mag_rate_mag`.

**65 features/stream** = 8 time + 6 spectral scalars + 9 quantiles + 42 subband:
- time: mean, std, energy, kurtosis, skew, mean-crossing-rate, top-autocorr value, autocorr lag
- spectral (on mean-removed signal): DC, peak power, **peak frequency** (cadence), FFT mean, FFT std, 1st/2nd-peak ratio
- quantiles: 0,5,10,25,50,75,90,95,100
- subband: energy + energy-ratio over centers {1,2,3,4,5,10,15} Hz × bandwidths {2,5,10} Hz

Name format: `"<stream>__<feature>"`, e.g. `acc_mag__freq_peak_freq`.

## Correctness notes (important)
- **Magnitudes only** — never raw axes (test orientation/location differ).
- **Mean removed before spectral** features (not full z-norm): keeps amplitude
  meaningful (vibration energy separates Still/vehicle) while killing the gravity
  DC offset that would swamp low subbands. `--zscore` enables full z-norm (ablation).
- **Energy + ratio** subbands kept (absolute = amplitude cue; ratio = scale-invariant).
- **Autocorrelation** uses biased-normalized estimator + local-max selection
  (robust period/cadence; avoids lag-1 and large-lag artifacts).
- **No cross-window features** — test windows are shuffled (no temporal smoothing).
- **Per-window label** = majority vote over the 500 frames (windows 99.8% constant).
- Validated by `--self-test`: synthetic peak-freq, gravity split, autocorr lag,
  ratio bounds, finiteness, 520-count/uniqueness.
- Gravity low-pass uses **SOS form** (`sosfiltfilt`), not `(b,a)` — required for
  numerical stability at this low normalized cutoff (0.006); zero-phase.
- Spectral block uses a **proper one-sided PSD** (positive bins doubled) so
  `sum(P) == N·var` exactly (**Parseval**-verified).
- **Differentially validated** (`validate_features.py --all`): on **all 9 files**
  (train/validation/test) every feature — and every derived stream — is recomputed
  with an *independent* reference and asserted to agree to ~6e-8 (float32 limit) or
  exactly. References: scipy.fft + scipy.signal.periodogram (FFT, 3rd impl),
  statsmodels `acf`, tsfel `zero_cross`, textbook G1/G2 moments, `np.linalg.norm`
  (magnitudes), central-difference (jerk), Parseval (energy), `sosfreqz` (filter
  response). Edge cases (zeros/const/spike/NaN-input) asserted finite. This
  process caught & fixed a real autocorrelation bug on near-constant streams.

## Performance
~3,300 windows/s single-core → ~60 s per 196k-window file; ~10 min for all 9
locally; trivially parallel as a SLURM array. Output ≈ 0.5 GB/file. Memory bounded
by `--chunk-size` (default 20k windows ≈ 1.5 GB peak).

## Optional: catch22 expansion
`pycatch22` is installed for an ablation feature block; not wired into the default
520 (kept lean per the plan). Add behind a flag if the LightGBM baseline shows headroom.
```
