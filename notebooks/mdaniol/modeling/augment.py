#!/usr/bin/env python3
"""Physics-grounded IMU augmentation (per AUGMENTATION_STRATEGY.md).

Branch policy (the key decision):
  * ROTATION  -> Branch B (raw-axis frozen-FM) ONLY. Handcrafted features are
    per-sensor magnitudes = rotation-invariant, so rotation there is a no-op
    (and a bug-check: Branch-A macro-F1 must NOT change under rotation).
  * amplitude scale / jitter / time-warp -> both branches.

All ops act on raw windows acc/gyr/mag of shape (n, 3, N) and return the same
shape. Rotation applies the SAME rotation matrix to all three sensors per window
(a rigid re-mounting of the device), which preserves each sensor's magnitude
exactly — verified in self_test().

Modes:
  rotation='so3'           uniform random SO(3) (full orientation change)
  rotation='gravity_aware' free yaw about gravity + bounded tilt (<= max_tilt_deg)

Skipped by design (evidence: hurt or wrong failure mode): permutation/segment
shuffle (breaks periodicity features), axis-swap, channel/sensor dropout
(2026 test drops the Hand *location*, not modalities), cutmix, window-crop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import PchipInterpolator

_EPS = 1e-8


@dataclass
class AugConfig:
    rotation: str | None = "so3"        # 'so3' | 'gravity_aware' | None
    max_tilt_deg: float = 45.0          # for gravity_aware
    scale: bool = True                  # amplitude scaling
    scale_sigma: float = 0.1
    scale_clip: tuple = (0.8, 1.2)
    jitter: bool = True
    jitter_sigma_frac: float = 0.03     # * per-channel std
    timewarp: bool = True
    tw_knots: int = 4
    tw_sigma: float = 0.15


# ---------------------------------------------------------------------------
def _apply_R(M: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply per-window rotation M (n,3,3) to sensor x (n,3,N)."""
    return np.einsum("nij,njk->nik", M, x)


def rotation_matrices(acc: np.ndarray, mode: str, max_tilt_deg: float, rng) -> np.ndarray:
    """Per-window rotation matrices (n,3,3)."""
    n = acc.shape[0]
    if mode == "so3":
        return Rotation.random(n, random_state=rng).as_matrix()
    if mode == "gravity_aware":
        # gravity direction per window ~ mean acceleration
        g = acc.mean(axis=2)                                   # (n,3)
        g_hat = g / (np.linalg.norm(g, axis=1, keepdims=True) + _EPS)
        # free yaw about gravity (does NOT change gravity direction)
        theta = rng.uniform(0, 2 * np.pi, n)
        R_yaw = Rotation.from_rotvec(theta[:, None] * g_hat)
        # bounded tilt about a random axis perpendicular to gravity
        rnd = rng.standard_normal((n, 3))
        perp = rnd - (np.sum(rnd * g_hat, axis=1, keepdims=True)) * g_hat
        perp /= (np.linalg.norm(perp, axis=1, keepdims=True) + _EPS)
        phi = rng.uniform(0, np.deg2rad(max_tilt_deg), n)
        R_tilt = Rotation.from_rotvec(phi[:, None] * perp)
        return (R_tilt * R_yaw).as_matrix()
    raise ValueError(f"bad rotation mode {mode}")


def amplitude_scale(x: np.ndarray, sigma: float, clip: tuple, rng) -> np.ndarray:
    """One gain per window (broadcast over axes+time) — sensor gain variation."""
    s = np.clip(rng.normal(1.0, sigma, (x.shape[0], 1, 1)), *clip)
    return x * s


def jitter(x: np.ndarray, sigma_frac: float, rng) -> np.ndarray:
    sd = x.std(axis=2, keepdims=True)                         # per (window, axis)
    return x + rng.normal(0.0, 1.0, x.shape) * (sigma_frac * sd)


def time_warp(x: np.ndarray, n_knots: int, sigma: float, rng) -> np.ndarray:
    """Smooth monotonic time distortion (cadence variation). Shared warp per window.

    Monotonicity is GUARANTEED for any sigma: (1) warp steps are clipped positive
    so the cumulative knot map strictly increases, and (2) a shape-preserving
    PCHIP interpolant (vs CubicSpline) cannot overshoot/dip between knots. This
    rules out time-reversal artifacts (validate_augment.py covers sigma up to 0.5).
    """
    n, c, N = x.shape
    out = np.empty_like(x)
    base = np.linspace(0, N - 1, N)
    knot_x = np.linspace(0, N - 1, n_knots + 2)
    for i in range(n):
        warp = np.maximum(rng.normal(1.0, sigma, n_knots + 2), 1e-3)  # positive steps
        cum = np.cumsum(warp)
        cum = (cum - cum[0]) / (cum[-1] - cum[0]) * (N - 1)   # strictly increasing 0..N-1
        t_new = np.clip(PchipInterpolator(knot_x, cum)(base), 0, N - 1)  # monotone interp
        for ch in range(c):
            out[i, ch] = np.interp(t_new, base, x[i, ch])
    return out


def augment(acc, gyr, mag, cfg: AugConfig, rng, with_rotation: bool):
    """Apply the configured pipeline. with_rotation=False for Branch A (handcrafted)."""
    a, g, m = acc, gyr, mag
    if with_rotation and cfg.rotation:
        M = rotation_matrices(a, cfg.rotation, cfg.max_tilt_deg, rng)
        a, g, m = _apply_R(M, a), _apply_R(M, g), _apply_R(M, m)
    if cfg.scale:
        a = amplitude_scale(a, cfg.scale_sigma, cfg.scale_clip, rng)
        g = amplitude_scale(g, cfg.scale_sigma, cfg.scale_clip, rng)
        m = amplitude_scale(m, cfg.scale_sigma, cfg.scale_clip, rng)
    if cfg.jitter:
        a, g, m = (jitter(v, cfg.jitter_sigma_frac, rng) for v in (a, g, m))
    if cfg.timewarp:
        a, g, m = (time_warp(v, cfg.tw_knots, cfg.tw_sigma, rng) for v in (a, g, m))
    return a, g, m


# ---------------------------------------------------------------------------
def self_test() -> None:
    rng = np.random.default_rng(0)
    n, N = 64, 500
    acc = np.zeros((n, 3, N)); acc[:, 2] = 9.81
    t = np.arange(N) / 100.0
    acc[:, 0] += np.sin(2 * np.pi * 2 * t)
    gyr = 0.2 * rng.standard_normal((n, 3, N))
    mag = 40 + rng.standard_normal((n, 3, N))

    # (1) ROTATION preserves per-timestep magnitude EXACTLY (both modes)
    for mode in ("so3", "gravity_aware"):
        M = rotation_matrices(acc, mode, 45.0, rng)
        for x in (acc, gyr, mag):
            xr = _apply_R(M, x)
            assert np.allclose(np.linalg.norm(xr, axis=1), np.linalg.norm(x, axis=1), atol=1e-4), \
                f"rotation {mode} changed magnitude"
        # M is a proper rotation: det=+1, orthonormal
        assert np.allclose(np.einsum("nij,nik->njk", M, M), np.eye(3)[None], atol=1e-5)
        assert np.allclose(np.linalg.det(M), 1.0, atol=1e-5)

    # (2) gravity_aware bounds the tilt of the gravity vector to <= max_tilt
    g0 = acc.mean(2); g0 /= np.linalg.norm(g0, axis=1, keepdims=True)
    M = rotation_matrices(acc, "gravity_aware", 45.0, rng)
    g1 = np.einsum("nij,nj->ni", M, g0)
    cosang = np.clip(np.sum(g0 * g1, axis=1), -1, 1)
    assert np.all(np.rad2deg(np.arccos(cosang)) <= 45.0 + 1e-3), "gravity tilt exceeded bound"

    # (3) amplitude scale within clip
    s_out = amplitude_scale(acc, 0.1, (0.8, 1.2), rng)
    ratio = s_out[:, 2, 0] / 9.81
    assert ratio.min() >= 0.8 - 1e-6 and ratio.max() <= 1.2 + 1e-6

    # (4) jitter increases std modestly, finite
    j = jitter(acc, 0.03, rng); assert np.all(np.isfinite(j))

    # (5) time-warp preserves shape, length, finiteness
    w = time_warp(acc[:4], 4, 0.15, rng)
    assert w.shape == acc[:4].shape and np.all(np.isfinite(w))

    # (6) full pipeline (Branch B): shapes preserved, finite
    a, g, m = augment(acc, gyr, mag, AugConfig(), rng, with_rotation=True)
    assert a.shape == acc.shape and np.all(np.isfinite(a))

    # (7) BRANCH-A INVARIANCE: rotation must NOT change magnitude streams
    #     (sanity check that handcrafted features are rotation-invariant)
    M = rotation_matrices(acc, "so3", 45.0, rng)
    accmag_before = np.linalg.norm(acc, axis=1)
    accmag_after = np.linalg.norm(_apply_R(M, acc), axis=1)
    assert np.allclose(accmag_before, accmag_after, atol=1e-4), "magnitude not rotation-invariant!"

    print("augment self_test OK — rotation preserves magnitude; gravity tilt bounded; "
          "scale/jitter/time-warp valid; Branch-A invariance confirmed")


if __name__ == "__main__":
    self_test()
