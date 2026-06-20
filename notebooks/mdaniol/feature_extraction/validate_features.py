#!/usr/bin/env python3
"""Differential validation of the SHL feature bank against independent libraries.

We do NOT trust the hand-written math in ``shl_features.py``. Instead, for every
feature we recompute the same quantity a second time with an *independent*
reference — a different code path or an established library — and assert the two
agree to floating-point tolerance on **real SHL windows**.

References used:
  * numpy / scipy.fft  — independent FFT recomputation (peak freq, FFT stats,
    subband energy/ratio, DC, peak ratio)
  * statsmodels.tsa.stattools.acf — reference autocorrelation (biased-normalized)
  * tsfel.feature_extraction.features.zero_cross — reference mean/zero crossings
  * textbook G1/G2 formulas — independent unbiased skew / excess-kurtosis
  * numpy direct — mean, std, energy, quantiles (these ARE numpy in the library,
    so they are correct by construction; we still diff them to prove ~0)

Run:
    python validate_features.py                # 400 real windows, all streams
    python validate_features.py --n 1000
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.fft as sp_fft               # independent FFT (not numpy.fft)
import scipy.signal as sp_signal         # periodogram cross-check
from statsmodels.tsa.stattools import acf as sm_acf

warnings.filterwarnings("ignore")
import tsfel.feature_extraction.features as TF  # noqa: E402

import shl_features as shl  # noqa: E402

FS, N = shl.FS, shl.N
FREQS = np.fft.rfftfreq(N, d=1.0 / FS)
POS = FREQS > 0.0
FPOS = FREQS[POS]
_ONESIDED = np.full(FREQS.size, 2.0); _ONESIDED[0] = 1.0
if N % 2 == 0:
    _ONESIDED[-1] = 1.0


# ---- independent reference implementations -------------------------------
def ref_unbiased_kurtosis(x: np.ndarray) -> float:
    """Fisher (excess) unbiased kurtosis G2, textbook formula."""
    n = x.size
    m = x.mean()
    m2 = np.mean((x - m) ** 2)
    m4 = np.mean((x - m) ** 4)
    if m2 < 1e-12:
        return 0.0
    g2 = m4 / m2 ** 2 - 3.0
    return ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)


def ref_unbiased_skew(x: np.ndarray) -> float:
    n = x.size
    m = x.mean()
    m2 = np.mean((x - m) ** 2)
    m3 = np.mean((x - m) ** 3)
    if m2 < 1e-12:
        return 0.0
    b1 = m3 / m2 ** 1.5
    return np.sqrt(n * (n - 1)) / (n - 2) * b1


def ref_spectral(x: np.ndarray):
    """Independent FFT (scipy.fft) on the mean-removed window -> scalars + subbands.

    Uses scipy.fft (a different implementation than the library's numpy.fft) and
    the same one-sided PSD scaling, so this is an independent recomputation of the
    same definition.
    """
    xc = x - x.mean()
    P = (np.abs(sp_fft.rfft(xc)) ** 2) * _ONESIDED   # scipy.fft + one-sided scale
    Ppos = P[POS]
    mag = np.sqrt(Ppos)
    total = Ppos.sum() + 1e-12
    k = Ppos.argmax()
    out = {
        "freq_dc": abs(x.mean()),
        "freq_peak_val": Ppos[k],
        "freq_peak_freq": FPOS[k],
        "freq_fft_mean": mag.mean(),
        "freq_fft_std": mag.std(),
    }
    # peak ratio: blank +/-2 bins around the peak, take next max
    P2 = Ppos.copy()
    for d in range(-2, 3):
        P2[min(max(k + d, 0), P2.size - 1)] = -np.inf
    p2 = P2.max()
    p2 = p2 if np.isfinite(p2) else 0.0
    out["freq_peak_ratio"] = Ppos[k] / (max(p2, 0.0) + 1e-12)
    for fc in shl.CENTERS_HZ:
        for bw in shl.BANDWIDTHS_HZ:
            lo, hi = max(0.0, fc - bw / 2), fc + bw / 2
            band = (FPOS >= lo) & (FPOS <= hi)
            e = Ppos[band].sum()
            out[f"sb_e_fc{fc:g}_bw{bw:g}"] = e
            out[f"sb_r_fc{fc:g}_bw{bw:g}"] = e / total
    return out


def ref_autocorr_top(x: np.ndarray):
    """Reference ACF via statsmodels, then independent local-max selection."""
    xc = x - x.mean()
    if np.std(xc) < 1e-12:
        return 0.0, 0.0
    # request one extra lag so the local-max test at AC_LAG_MAX has its right
    # neighbour available (matches shl, which uses the full ACF for neighbours).
    r = sm_acf(xc, adjusted=False, nlags=shl.AC_LAG_MAX + 1, fft=True)
    best_v, best_l = 0.0, 0
    for k in range(shl.AC_LAG_MIN, shl.AC_LAG_MAX + 1):
        if k <= 0 or k >= r.size - 1:
            continue
        if r[k] > r[k - 1] and r[k] > r[k + 1] and r[k] > best_v:
            best_v, best_l = r[k], k
    return best_v, float(best_l)


def maxreldiff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    denom = np.maximum(np.abs(b), 1.0)          # relative for large, absolute-ish for small
    return float(np.max(np.abs(a - b) / denom))


def validate_streams(acc, gyr, mag, streams) -> dict[str, float]:
    """Independently re-derive the 8 streams and diff against shl.derive_streams."""
    out: dict[str, float] = {}
    # magnitude vs np.linalg.norm (independent of the library's einsum-style sum)
    out["stream magnitude (np.linalg.norm)"] = max(
        maxreldiff(streams["acc_mag"], np.linalg.norm(acc, axis=1)),
        maxreldiff(streams["gyr_mag"], np.linalg.norm(gyr, axis=1)),
        maxreldiff(streams["mag_mag"], np.linalg.norm(mag, axis=1)),
    )
    # gravity wiring: derive_streams must use the documented low-pass + body=acc-gravity
    g = shl._lowpass_gravity(acc)
    out["stream body_mag (norm of acc-gravity)"] = maxreldiff(
        streams["acc_body_mag"], np.linalg.norm(acc - g, axis=1))
    out["stream gravity_mag (norm of low-pass)"] = maxreldiff(
        streams["gravity_mag"], np.linalg.norm(g, axis=1))
    # gravity property: the low-pass must REDUCE the high-frequency energy fraction
    # vs the raw acc magnitude (a correct low-pass concentrates energy at low freq).
    def hi_frac(sig):  # fraction of post-mean-removal energy above 2 Hz
        sc = sig - sig.mean(axis=1, keepdims=True)
        P = (np.abs(sp_fft.rfft(sc, axis=1)) ** 2)
        return P[:, FREQS > 2.0].sum(1) / (P.sum(1) + 1e-12)
    raw_hi = hi_frac(np.linalg.norm(acc, axis=1)).mean()
    grav_hi = hi_frac(np.linalg.norm(g, axis=1)).mean()
    # 0 when the filter clearly suppresses highs (>5x reduction), else the ratio.
    out["stream gravity suppresses highs (>5x)"] = (
        0.0 if grav_hi < raw_hi / 5 else grav_hi / max(raw_hi, 1e-12))
    # jerk: np.gradient interior == central difference (x[i+1]-x[i-1])/(2 dt)
    cd = (acc[:, :, 2:] - acc[:, :, :-2]) * (FS / 2.0)            # (n,3,N-2)
    jerk_cd_mag = np.linalg.norm(cd, axis=1)                       # interior magnitude
    # compare interior region of the library jerk stream
    lib_jerk_interior = streams["acc_jerk_mag"][:, 1:-1]
    out["stream jerk interior (central diff)"] = maxreldiff(
        lib_jerk_interior, jerk_cd_mag)
    return out


# ---- driver --------------------------------------------------------------
ALL_FILES = ("train/Bag", "train/Hips", "train/Torso", "train/Hand",
             "validation/Bag", "validation/Hips", "validation/Torso",
             "validation/Hand", "test/all")


def validate_file(src: Path, n: int, rtol: float) -> tuple[bool, dict[str, float]]:
    pf = pq.ParquetFile(src)
    batch = next(pf.iter_batches(batch_size=n))
    def axes(p):
        cols = [batch.column(batch.schema.get_field_index(f"{p}_{a}"))
                for a in "xyz"]
        return np.stack([c.combine_chunks().flatten().to_numpy(False).reshape(-1, N)
                         if hasattr(c, "combine_chunks")
                         else c.flatten().to_numpy(False).reshape(-1, N)
                         for c in cols], axis=1).astype(np.float64)
    acc, gyr, mag = axes("Acc"), axes("Gyr"), axes("Mag")
    nw = acc.shape[0]

    mat, names = shl.extract_features(acc, gyr, mag)
    col = {nm: i for i, nm in enumerate(names)}
    streams = shl.derive_streams(acc, gyr, mag)

    checks: dict[str, float] = {}
    def record(key, vals_mine, vals_ref):
        d = maxreldiff(vals_mine, vals_ref)
        checks[key] = max(checks.get(key, 0.0), d)

    # --- (A) derived streams, independently re-derived ---
    checks.update(validate_streams(acc, gyr, mag, streams))

    for sname, X in streams.items():
        c = lambda f: mat[:, col[f"{sname}__{f}"]]  # noqa: E731
        record("time_mean (numpy)", c("time_mean"), X.mean(1))
        record("time_std (numpy)", c("time_std"), X.std(1))
        record("time_energy (numpy)", c("time_energy"), np.mean(X * X, 1))
        for p in shl.QUANTILES:
            record("quantiles (numpy.percentile)", c(f"q{p}"),
                   np.percentile(X, p, axis=1))
        kref = np.array([ref_unbiased_kurtosis(r) for r in X])
        sref = np.array([ref_unbiased_skew(r) for r in X])
        record("kurtosis (G2 formula)", c("time_kurtosis"), kref)
        record("skew (G1 formula)", c("time_skew"), sref)
        zc = np.array([TF.zero_cross(r - r.mean()) for r in X]) / float(N)
        record("mean_crossing (tsfel.zero_cross)", c("time_mcr"), zc)
        av = np.array([ref_autocorr_top(r) for r in X])
        record("autocorr value (statsmodels.acf)", c("time_ac_val"), av[:, 0])
        record("autocorr lag (statsmodels.acf)", c("time_ac_lag"), av[:, 1])
        # spectral vs independent scipy.fft (one-sided) recomputation
        sp = [ref_spectral(r) for r in X]
        for fkey in sp[0]:
            ref = np.array([s[fkey] for s in sp])
            tag = ("subband energy/ratio (scipy.fft)" if fkey.startswith("sb_")
                   else f"{fkey} (scipy.fft)")
            record(tag, c(fkey), ref)
        # (B) peak frequency cross-check vs scipy.signal.periodogram (3rd impl)
        pf_peaks = []
        for r in X:
            f, Pxx = sp_signal.periodogram(r, fs=FS, window="boxcar",
                                           detrend="constant", scaling="spectrum")
            pf_peaks.append(f[1:][Pxx[1:].argmax()])  # exclude DC
        record("peak_freq (scipy.periodogram)", c("freq_peak_freq"),
               np.array(pf_peaks))
        # (C) Parseval: one-sided spectral energy conserves time-domain energy
        xc = X - X.mean(1, keepdims=True)
        Pall = (np.abs(sp_fft.rfft(xc, axis=1)) ** 2) * _ONESIDED
        record("Parseval (sum P == N*var)", Pall.sum(1) / N, (xc * xc).sum(1))

    ok = True
    for k in sorted(checks):
        d = checks[k]
        tol = 0.0 if ("lag" in k or "peak_freq" in k) else rtol
        passed = d <= max(tol, rtol)
        ok &= passed
    return ok, checks


def _print_report(title: str, checks: dict[str, float], rtol: float) -> bool:
    print(f"\n=== {title} ===")
    print(f"{'check':46s} {'max rel/abs diff':>18s}   verdict")
    print("-" * 80)
    ok = True
    for k in sorted(checks):
        d = checks[k]
        tol = 0.0 if ("lag" in k or "peak_freq" in k) else rtol
        passed = d <= max(tol, rtol)
        ok &= passed
        print(f"{k:46s} {d:18.2e}   {'PASS' if passed else 'FAIL'}")
    print("-" * 80)
    print("ALL PASS" if ok else "*** FAILURES ABOVE ***")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="windows per file")
    ap.add_argument("--src", type=Path,
                    default=Path(__file__).resolve().parents[3]
                    / "dataset_parquet/train/Bag.parquet")
    ap.add_argument("--all", action="store_true",
                    help="validate all 9 files (train+validation+test)")
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3] / "dataset_parquet"
    targets = ([root / f"{rel}.parquet" for rel in ALL_FILES] if args.all
               else [args.src])

    all_ok = True
    for src in targets:
        if not src.exists():
            print(f"SKIP (missing): {src}")
            continue
        ok, checks = validate_file(src, args.n, args.rtol)
        all_ok &= _print_report(f"{src.parent.name}/{src.name}  (n={args.n})",
                                checks, args.rtol)
    print("\n" + ("=" * 80))
    print("OVERALL: ALL CHECKS PASSED across all files"
          if all_ok else "OVERALL: SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
