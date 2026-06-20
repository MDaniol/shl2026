#!/usr/bin/env python3
"""SHL 2026 handcrafted feature bank — vectorized, DSP-correct.

Implements the feature spec in ``notebooks/mdaniol/IMPLEMENTATION_PLAN.md §2``:
**8 derived 1-D streams × a 65-feature bank = 520 features per window.**

Design decisions (read before changing — these affect correctness):

1. **Orientation/location invariance.** Every feature is computed on the
   *magnitude* of a derived stream, never on raw per-axis values. The test set
   has unknown phone orientation and a different location mix (no Hand), so
   axis-dependent features would not transfer. (Wang et al., IEEE Access 2019;
   arXiv:2407.11048.)

2. **Mean removal before spectral analysis (NOT full z-norm by default).** The
   raw magnitude carries a large DC offset (e.g. acc-magnitude ≈ 9.8 from
   gravity). If left in, the 0-Hz bin dominates every low-frequency subband and
   the *fluctuation* energy that actually distinguishes Still/Walk/vehicle is
   swamped. So spectral features use the mean-removed stream ``s - mean(s)``.
   This **preserves amplitude** (variance/energy stay meaningful — vibration
   intensity separates Still from vehicles) while removing the offset.
   ``--zscore`` additionally divides by std: then absolute subband energy
   becomes shape-only and collinear with the energy *ratio*, so it is offered
   only as an ablation, not the default. (arXiv:2407.11048 found z-norm helps in
   the *missing-modality* setting; here all 9 channels are present and amplitude
   is informative, so mean-removal is the right default.)

3. **Energy AND energy-ratio subbands.** Absolute subband energy is
   amplitude-bearing (Still≈flat, vehicle≈vibration); the ratio
   (band / total fluctuation energy) is amplitude-invariant. Keeping both is
   what the SHL authors did and gives trees complementary cues.

4. **No cross-window features.** The test windows are shuffled, so every feature
   is strictly window-internal. Nothing here looks outside the 500 samples.

Sampling: 500 samples @ 100 Hz → 5 s window, Nyquist 50 Hz, FFT resolution
0.2 Hz.

All public functions operate on **batches**: arrays of shape ``(n_windows, N)``.
The math is identical to a per-window computation but runs ~1000× faster.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Constants (match the SHL archive + the implementation plan §2)
# ---------------------------------------------------------------------------
FS = 100.0                 # Hz
N = 500                    # samples per window
GRAVITY_CUTOFF_HZ = 0.3    # low-pass cutoff separating gravity from body accel
GRAVITY_FILTER_ORDER = 4

# Subband grid: centers × bandwidths × {energy, ratio} = 7 × 3 × 2 = 42 features.
CENTERS_HZ = (1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0)
BANDWIDTHS_HZ = (2.0, 5.0, 10.0)
# Quantiles (percentiles) = 9 features.
QUANTILES = (0, 5, 10, 25, 50, 75, 90, 95, 100)
# Autocorrelation search range (lag 0 excluded; up to half the window).
AC_LAG_MIN = 1
AC_LAG_MAX = N // 2

_EPS = 1e-12

# Order of the 8 derived streams (also the feature-name prefix order).
STREAM_NAMES = (
    "acc_mag",        # |acc| including gravity
    "acc_body_mag",   # |acc - gravity|  (high-pass / linear acceleration)
    "gravity_mag",    # |low-pass(acc)|  (~9.8; varies with sustained accel)
    "acc_jerk_mag",   # |d acc / dt|
    "gyr_mag",        # |gyro|
    "gyr_jerk_mag",   # |d gyro / dt|
    "mag_mag",        # |magnetic field|
    "mag_rate_mag",   # |d magnetic field / dt|
)


# ---------------------------------------------------------------------------
# Derived streams
# ---------------------------------------------------------------------------
def _lowpass_gravity(axis_xyz: np.ndarray) -> np.ndarray:
    """Zero-phase low-pass of each axis to estimate the gravity component.

    ``axis_xyz`` : (n, 3, N).  Returns same shape.

    Uses **second-order-sections (SOS)** form, not transfer-function (b, a):
    the normalized cutoff here is very low (0.3/50 = 0.006) and a 4th-order
    (b, a) filter is numerically ill-conditioned at such cutoffs — scipy
    recommends SOS for stability. ``sosfiltfilt`` is zero-phase (no time shift),
    which is essential so the gravity estimate stays time-aligned with the signal.
    """
    sos = sp_signal.butter(
        GRAVITY_FILTER_ORDER, GRAVITY_CUTOFF_HZ / (FS / 2.0),
        btype="low", output="sos",
    )
    # sosfiltfilt operates along the last axis; N=500 >> padlen so it is safe.
    return sp_signal.sosfiltfilt(sos, axis_xyz, axis=-1)


def _mag(axis_xyz: np.ndarray) -> np.ndarray:
    """Euclidean magnitude across the 3 axes. (n, 3, N) -> (n, N)."""
    return np.sqrt(np.sum(axis_xyz * axis_xyz, axis=1))


def _deriv(axis_xyz: np.ndarray) -> np.ndarray:
    """Time derivative per axis in physical units (per second). (n,3,N)->(n,3,N)."""
    return np.gradient(axis_xyz, 1.0 / FS, axis=-1)


def derive_streams(
    acc: np.ndarray, gyr: np.ndarray, mag: np.ndarray
) -> dict[str, np.ndarray]:
    """Build the 8 derived 1-D streams from the 3 tri-axial sensors.

    acc, gyr, mag : (n, 3, N) arrays (axes order x, y, z).
    Returns dict ``{stream_name: (n, N)}`` in ``STREAM_NAMES`` order.
    """
    gravity = _lowpass_gravity(acc)         # (n,3,N)
    body = acc - gravity                    # linear acceleration

    streams = {
        "acc_mag": _mag(acc),
        "acc_body_mag": _mag(body),
        "gravity_mag": _mag(gravity),
        "acc_jerk_mag": _mag(_deriv(acc)),
        "gyr_mag": _mag(gyr),
        "gyr_jerk_mag": _mag(_deriv(gyr)),
        "mag_mag": _mag(mag),
        "mag_rate_mag": _mag(_deriv(mag)),
    }
    return streams


# ---------------------------------------------------------------------------
# Feature blocks (operate on a single stream X: (n, N))
# ---------------------------------------------------------------------------
def _autocorr_top(Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top normalized autocorrelation value and its lag (Wiener–Khinchin, FFT).

    Xc : mean-removed stream (n, N). Returns (top_value (n,), lag (n,)).
    Periodic motion (Walk/Run cadence) gives a high autocorr peak at the stride lag.

    Uses the **biased** normalized estimator (numerically stable) and selects the
    strongest **local maximum** in the lag range, not the global max. This (a)
    excludes the monotonically-decaying main lobe so lag 1 cannot win spuriously,
    and (b) avoids the unbiased estimator's variance blow-up at large lags. The
    biased envelope (1-k/N) still leaves the fundamental period as the dominant
    local peak. A genuine periodicity has a **positive** autocorrelation at its
    period, so only positive local maxima count; streams with none (flat /
    aperiodic, e.g. gravity_mag) return value 0, lag 0.
    """
    n, m = Xc.shape
    nfft = 1 << int(np.ceil(np.log2(2 * m)))      # zero-pad to avoid circular wrap
    F = np.fft.rfft(Xc, n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, :m].real
    ac0 = ac[:, 0:1]
    acn = ac / (ac0 + _EPS)                        # biased, normalized so lag-0 == 1

    # interior, POSITIVE local maxima (valid for lags 1 .. m-2)
    is_peak = (
        (acn[:, 1:-1] > acn[:, :-2])
        & (acn[:, 1:-1] > acn[:, 2:])
        & (acn[:, 1:-1] > 0.0)
    )
    cand = np.full_like(acn, -np.inf)
    cand[:, 1:-1] = np.where(is_peak, acn[:, 1:-1], -np.inf)

    window = cand[:, AC_LAG_MIN:AC_LAG_MAX + 1]
    top_val = window.max(axis=1)
    lag = window.argmax(axis=1) + AC_LAG_MIN
    no_peak = ~np.isfinite(top_val)               # no positive local max
    # also treat near-constant streams (zero variance) as non-periodic
    no_peak |= Xc.std(axis=1) <= _EPS
    top_val = np.where(no_peak, 0.0, top_val)
    lag = np.where(no_peak, 0, lag).astype(np.float32)
    return top_val, lag


def _time_features(X: np.ndarray) -> dict[str, np.ndarray]:
    """8 time-domain features on the raw stream (amplitude is meaningful)."""
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    energy = np.mean(X * X, axis=1)               # mean power
    # bias=False -> sample (unbiased) estimators. Near-constant windows trigger a
    # harmless precision warning and yield nan; suppress, then set those to 0.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        kurt = sp_stats.kurtosis(X, axis=1, fisher=True, bias=False)
        skew = sp_stats.skew(X, axis=1, bias=False)
    nonconst = std > _EPS
    kurt = np.where(nonconst, kurt, 0.0)
    skew = np.where(nonconst, skew, 0.0)
    centered = X - mean[:, None]
    signs = centered >= 0.0
    mcr = np.count_nonzero(np.diff(signs, axis=1), axis=1) / float(X.shape[1])
    ac_val, ac_lag = _autocorr_top(centered)
    return {
        "time_mean": mean,
        "time_std": std,
        "time_energy": energy,
        "time_kurtosis": kurt,
        "time_skew": skew,
        "time_mcr": mcr,
        "time_ac_val": ac_val,
        "time_ac_lag": ac_lag,
    }


def _quantile_features(X: np.ndarray) -> dict[str, np.ndarray]:
    """9 percentile features on the raw stream."""
    qs = np.percentile(X, QUANTILES, axis=1)      # (9, n)
    return {f"q{p}": qs[i] for i, p in enumerate(QUANTILES)}


def _spectral_features(
    X: np.ndarray, freqs: np.ndarray, zscore: bool
) -> dict[str, np.ndarray]:
    """6 spectral scalars + 42 subband features on the mean-removed stream.

    freqs : one-sided rfft frequency grid (len N//2+1), precomputed once.
    """
    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean                                 # remove DC offset (gravity, etc.)
    if zscore:                                     # optional ablation
        Xc = Xc / (X.std(axis=1, keepdims=True) + _EPS)

    spec = np.fft.rfft(Xc, axis=1)
    P = (spec.real ** 2 + spec.imag ** 2)
    # Proper one-sided power spectrum: double every bin except DC and (for even
    # N) Nyquist, so that sum(P) == N * sum(Xc**2) exactly (Parseval). This makes
    # absolute subband energies physically meaningful; ratios are unaffected.
    onesided = np.full(P.shape[1], 2.0)
    onesided[0] = 1.0
    if N % 2 == 0:
        onesided[-1] = 1.0
    P = P * onesided
    magspec = np.sqrt(P)

    pos = freqs > 0.0                              # exclude the (now ~0) DC bin
    Ppos = P[:, pos]
    magpos = magspec[:, pos]
    fpos = freqs[pos]

    total = Ppos.sum(axis=1) + _EPS               # total fluctuation energy

    # --- 6 scalars ---
    dc = np.abs(mean[:, 0])                        # the removed DC == |mean|
    peak_idx = Ppos.argmax(axis=1)
    peak_val = Ppos[np.arange(Ppos.shape[0]), peak_idx]
    peak_freq = fpos[peak_idx]
    fft_mean = magpos.mean(axis=1)
    fft_std = magpos.std(axis=1)

    # 1st/2nd peak ratio: blank a ±2-bin neighborhood of the global peak.
    P2 = Ppos.copy()
    rows = np.arange(P2.shape[0])
    for d in range(-2, 3):
        cols = np.clip(peak_idx + d, 0, P2.shape[1] - 1)
        P2[rows, cols] = -np.inf
    peak2 = P2.max(axis=1)
    peak2 = np.where(np.isfinite(peak2), peak2, 0.0)
    peak_ratio = peak_val / (np.maximum(peak2, 0.0) + _EPS)

    out = {
        "freq_dc": dc,
        "freq_peak_val": peak_val,
        "freq_peak_freq": peak_freq,
        "freq_fft_mean": fft_mean,
        "freq_fft_std": fft_std,
        "freq_peak_ratio": peak_ratio,
    }

    # --- 42 subband features (energy + ratio) ---
    for fc in CENTERS_HZ:
        for bw in BANDWIDTHS_HZ:
            lo = max(0.0, fc - bw / 2.0)
            hi = fc + bw / 2.0
            band = (fpos >= lo) & (fpos <= hi)
            e = Ppos[:, band].sum(axis=1)
            out[f"sb_e_fc{fc:g}_bw{bw:g}"] = e
            out[f"sb_r_fc{fc:g}_bw{bw:g}"] = e / total
    return out


# ---------------------------------------------------------------------------
# Per-stream and full-window assembly
# ---------------------------------------------------------------------------
def stream_features(
    X: np.ndarray, freqs: np.ndarray, zscore: bool = False
) -> dict[str, np.ndarray]:
    """65 features for one stream X: (n, N). Keys are the bare feature names."""
    feats: dict[str, np.ndarray] = {}
    feats.update(_time_features(X))
    feats.update(_quantile_features(X))
    feats.update(_spectral_features(X, freqs, zscore))
    return feats


def feature_names(zscore: bool = False) -> list[str]:
    """Deterministic ordered list of all 520 feature names."""
    probe = np.zeros((1, N), dtype=np.float64)
    probe[0, 0] = 1.0  # non-degenerate so nothing is undefined
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)
    names: list[str] = []
    for stream in STREAM_NAMES:
        for fname in stream_features(probe, freqs, zscore):
            names.append(f"{stream}__{fname}")
    return names


def extract_features(
    acc: np.ndarray,
    gyr: np.ndarray,
    mag: np.ndarray,
    zscore: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Compute the full (n, 520) feature matrix for a batch of windows.

    acc, gyr, mag : (n, 3, N) float arrays.
    Returns (features (n, 520) float32, ordered names).
    NaN/inf (e.g. zero-variance windows) are replaced with 0.0.
    """
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)
    streams = derive_streams(acc, gyr, mag)
    cols: list[np.ndarray] = []
    names: list[str] = []
    for stream in STREAM_NAMES:
        feats = stream_features(streams[stream], freqs, zscore)
        for fname, vals in feats.items():
            names.append(f"{stream}__{fname}")
            cols.append(vals.astype(np.float32))
    mat = np.stack(cols, axis=1)
    np.nan_to_num(mat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return mat, names


# ---------------------------------------------------------------------------
# Self-test: synthetic signals with known properties
# ---------------------------------------------------------------------------
def self_test() -> None:
    """Validate the DSP against signals whose answers we know analytically."""
    rng = np.random.default_rng(0)
    t = np.arange(N) / FS

    # (1) Peak-frequency detection: a pure 3 Hz tone in acc_x must yield a
    #     dominant frequency at 3 Hz on the acc_mag stream.
    f0 = 3.0
    acc = np.zeros((1, 3, N))
    acc[0, 0] = 9.81 + 2.0 * np.sin(2 * np.pi * f0 * t)  # gravity + 3 Hz tone
    gyr = np.zeros((1, 3, N))
    mag = np.full((1, 3, N), 20.0)
    mat, names = extract_features(acc, gyr, mag)
    idx = {n: i for i, n in enumerate(names)}
    pk = mat[0, idx["acc_mag__freq_peak_freq"]]
    assert abs(pk - f0) <= 0.2, f"peak freq {pk} != {f0}"

    # (2) Gravity separation: gravity_mag ≈ 9.81, body magnitude small-mean.
    gmag = mat[0, idx["gravity_mag__time_mean"]]
    assert abs(gmag - 9.81) < 0.5, f"gravity mean {gmag} != ~9.81"

    # (3) Mean-removal: subband ratios sum (over the full spectrum) to ~1.
    #     Check a broadband white-noise stream: total ratio in [0,1] per band.
    acc2 = rng.standard_normal((4, 3, N))
    gyr2 = rng.standard_normal((4, 3, N))
    mag2 = rng.standard_normal((4, 3, N))
    mat2, _ = extract_features(acc2, gyr2, mag2)
    rcols = [i for i, n in enumerate(names) if "__sb_r_" in n]
    assert np.all(mat2[:, rcols] >= -1e-6) and np.all(mat2[:, rcols] <= 1 + 1e-6)

    # (4) Autocorrelation lag: a 2 Hz tone (period 50 samples) → ac lag ≈ 50.
    #     Use a large DC offset so |acc| does NOT rectify the tone (which would
    #     halve its period); the magnitude then tracks (offset + tone).
    acc3 = np.zeros((1, 3, N))
    acc3[0, 0] = 10.0 + np.sin(2 * np.pi * 2.0 * t)
    mat3, _ = extract_features(acc3, np.zeros((1, 3, N)), np.zeros((1, 3, N)))
    lag = mat3[0, idx["acc_mag__time_ac_lag"]]
    assert abs(lag - 50) <= 2, f"ac lag {lag} != ~50"

    # (5) Shape + finiteness.
    assert mat.shape[1] == 520, f"expected 520 features, got {mat.shape[1]}"
    assert len(names) == len(set(names)) == 520, "duplicate/!=520 feature names"
    assert np.all(np.isfinite(mat2)), "non-finite features leaked"

    # (6) Parseval: one-sided power must conserve energy (sum P == N*sum xc^2).
    x = rng.standard_normal(N)
    xc = x - x.mean()
    spec = np.fft.rfft(xc)
    P = spec.real ** 2 + spec.imag ** 2
    onesided = np.full(P.size, 2.0); onesided[0] = 1.0
    if N % 2 == 0:
        onesided[-1] = 1.0
    assert abs((P * onesided).sum() - N * np.sum(xc ** 2)) < 1e-6 * N, "Parseval failed"

    # (7) Gravity filter frequency response (rigorous, window-independent):
    #     gain ~1 in the passband (<0.3 Hz) and ~0 in the stopband (>>0.3 Hz).
    sos = sp_signal.butter(GRAVITY_FILTER_ORDER, GRAVITY_CUTOFF_HZ / (FS / 2.0),
                           btype="low", output="sos")
    _, h = sp_signal.sosfreqz(sos, worN=[0.05, GRAVITY_CUTOFF_HZ, 2.0, 5.0], fs=FS)
    gain = np.abs(h)
    # filtfilt applies the response twice (zero-phase), so effective gain = gain**2
    eff = gain ** 2
    assert eff[0] > 0.99, f"passband gain too low at 0.05 Hz: {eff[0]:.3f}"
    assert eff[1] > 0.45, f"gain at cutoff unexpectedly low: {eff[1]:.3f}"  # ~0.5
    assert eff[3] < 0.01, f"stopband gain too high at 5 Hz: {eff[3]:.4f}"

    # (8) Degenerate inputs must yield finite features (no NaN/inf leakage).
    zeros = np.zeros((1, 3, N))
    const = np.full((1, 3, N), 7.0)
    spike = np.zeros((1, 3, N)); spike[0, 0, 250] = 100.0
    withnan = np.zeros((1, 3, N)); withnan[0, 0, 0] = np.nan
    for name_, bad in [("zeros", zeros), ("const", const),
                       ("spike", spike), ("nan-input", withnan)]:
        m, _ = extract_features(bad, np.zeros((1, 3, N)), np.zeros((1, 3, N)))
        assert np.all(np.isfinite(m)), f"non-finite features for {name_} input"

    print(f"self_test OK — {mat.shape[1]} features, all DSP/edge/Parseval checks passed")


if __name__ == "__main__":
    self_test()
