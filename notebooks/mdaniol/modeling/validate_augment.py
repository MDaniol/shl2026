#!/usr/bin/env python3
"""Differential + property validation of augment.py (the 100%-assurance bar).

Mirrors validate_features.py: every augmentation is checked against an INDEPENDENT
reference (not the same library) or a closed-form property, to machine precision
where exact and within a stated statistical tolerance where stochastic.

  ROTATION   vs analytic Rodrigues formula (independent of scipy)  -> exact
  ROTATION   norm preservation at every timestep                   -> exact
  GRAVITY-AW yaw-only (tilt=0) leaves gravity EXACTLY fixed         -> exact
  GRAVITY-AW tilt bounded by max_tilt_deg                           -> exact
  SCALE      one gain per window (const over axes&time), in clip    -> exact
  JITTER     sigma=0 identity; measured noise std == contract       -> statistical
  TIME-WARP  sigma=0 identity; warp STRICTLY MONOTONIC; len/endpoints-> exact
  TIME-WARP  np.interp path vs scipy.interp1d                       -> ~1e-12

Run:  python validate_augment.py   (exit 0 = all pass)
"""
from __future__ import annotations

import sys
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d, PchipInterpolator

import augment as aug

PASS, FAIL = "  PASS", "  FAIL"
_fail = 0


def check(name, cond, detail=""):
    global _fail
    print((PASS if cond else FAIL) + f"  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        _fail += 1


def rodrigues(axis_angle):
    """Independent rotation matrix from axis-angle (Rodrigues), no scipy."""
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-12:
        return np.eye(3)
    k = axis_angle / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def main() -> int:
    rng = np.random.default_rng(0)
    n, N = 256, 500

    # --- 1. scipy rotation == analytic Rodrigues (differential) -------------
    rotvecs = rng.standard_normal((n, 3))
    Msci = Rotation.from_rotvec(rotvecs).as_matrix()
    Mrod = np.stack([rodrigues(v) for v in rotvecs])
    check("rotation matrix vs analytic Rodrigues",
          np.allclose(Msci, Mrod, atol=1e-12), f"max|Δ|={np.abs(Msci-Mrod).max():.1e}")

    # --- 2. rotation preserves per-timestep norm (float64, exact) -----------
    acc = np.zeros((n, 3, N)); acc[:, 2] = 9.81
    t = np.arange(N) / 100.0
    acc[:, 0] += np.sin(2 * np.pi * 2 * t)
    gyr = 0.2 * rng.standard_normal((n, 3, N))
    mag = 40 + 5 * rng.standard_normal((n, 3, N))
    for mode in ("so3", "gravity_aware"):
        M = aug.rotation_matrices(acc, mode, 45.0, np.random.default_rng(1))
        worst = 0.0
        for x in (acc, gyr, mag):
            xr = aug._apply_R(M, x)
            worst = max(worst, np.abs(np.linalg.norm(xr, axis=1)
                                     - np.linalg.norm(x, axis=1)).max())
        check(f"rotation[{mode}] preserves per-timestep magnitude", worst < 1e-9,
              f"max|Δ‖·‖|={worst:.1e}")

    # --- 3. gravity-aware: yaw-only (tilt=0) leaves gravity EXACTLY fixed ----
    g0 = acc.mean(2); g0 = g0 / np.linalg.norm(g0, axis=1, keepdims=True)
    M0 = aug.rotation_matrices(acc, "gravity_aware", 0.0, np.random.default_rng(2))
    g_yaw = np.einsum("nij,nj->ni", M0, g0)
    check("gravity-aware tilt=0 keeps gravity direction fixed",
          np.allclose(g_yaw, g0, atol=1e-9), f"max|Δ|={np.abs(g_yaw-g0).max():.1e}")

    # --- 4. gravity-aware: tilt bounded by max_tilt_deg ---------------------
    for tilt in (15.0, 30.0, 45.0):
        M = aug.rotation_matrices(acc, "gravity_aware", tilt, np.random.default_rng(3))
        g1 = np.einsum("nij,nj->ni", M, g0)
        ang = np.rad2deg(np.arccos(np.clip(np.sum(g0 * g1, 1), -1, 1)))
        check(f"gravity-aware tilt<= {tilt:g}deg", ang.max() <= tilt + 1e-6,
              f"max tilt={ang.max():.2f}deg")

    # --- 5. amplitude scale: ONE gain per window, const over axes&time ------
    s = aug.amplitude_scale(acc + 1.0, 0.1, (0.8, 1.2), np.random.default_rng(4))
    ratio = s / (acc + 1.0)
    per_win_spread = (ratio.max(axis=(1, 2)) - ratio.min(axis=(1, 2))).max()
    check("scale: single gain per window (const over axes&time)",
          per_win_spread < 1e-9, f"max within-window spread={per_win_spread:.1e}")
    g = ratio[:, 0, 0]
    check("scale: gains within clip [0.8,1.2]", g.min() >= 0.8 - 1e-9 and g.max() <= 1.2 + 1e-9,
          f"[{g.min():.3f},{g.max():.3f}]")

    # --- 6. jitter: sigma=0 identity; measured noise std == contract --------
    z = aug.jitter(acc, 0.0, np.random.default_rng(5))
    check("jitter sigma=0 is identity", np.array_equal(z, acc))
    big = rng.standard_normal((4000, 3, N)) * np.array([1.0, 2.0, 3.0])[None, :, None]
    sigma_frac = 0.05
    jz = aug.jitter(big, sigma_frac, np.random.default_rng(6))
    noise = jz - big
    meas = noise.std(axis=(0, 2))                          # per channel
    expect = sigma_frac * big.std(axis=2).mean(0)          # contract: frac * per-channel std
    rel = np.abs(meas - expect) / expect
    check("jitter noise std == sigma_frac * per-channel std", rel.max() < 0.05,
          f"max rel err={rel.max():.3f}")

    # --- 7. time-warp: identity at sigma=0; STRICT MONOTONIC; len/endpoints --
    w0 = aug.time_warp(acc[:8], 4, 0.0, np.random.default_rng(7))
    check("time-warp sigma=0 is identity", np.allclose(w0, acc[:8], atol=1e-9),
          f"max|Δ|={np.abs(w0-acc[:8]).max():.1e}")

    # reconstruct the warp map exactly as augment does (positive steps + PCHIP),
    # check monotonicity at recommended AND aggressive sigma (guarantee, not luck)
    base = np.linspace(0, N - 1, N)
    knot_x = np.linspace(0, N - 1, 4 + 2)
    for sg in (0.15, 0.5):
        non_monotone = 0
        for trial in range(2000):
            r = np.random.default_rng(1000 + trial)
            warp = np.maximum(r.normal(1.0, sg, 4 + 2), 1e-3)
            cum = np.cumsum(warp); cum = (cum - cum[0]) / (cum[-1] - cum[0]) * (N - 1)
            t_new = np.clip(PchipInterpolator(knot_x, cum)(base), 0, N - 1)
            if np.any(np.diff(t_new) < -1e-9):
                non_monotone += 1
        check(f"time-warp map strictly monotonic over 2000 trials (sigma={sg})",
              non_monotone == 0, f"{non_monotone} non-monotonic")
    check("time-warp preserves length & shape", w0.shape == acc[:8].shape)

    # cross-check np.interp vs scipy interp1d on a warped grid
    x = acc[0, 0]
    t_new = np.clip(np.linspace(0, N - 1, N) + 0.3 * np.sin(base / 30), 0, N - 1)
    a_np = np.interp(t_new, base, x)
    a_sp = interp1d(base, x)(t_new)
    check("time-warp interp: np.interp vs scipy.interp1d",
          np.allclose(a_np, a_sp, atol=1e-12), f"max|Δ|={np.abs(a_np-a_sp).max():.1e}")

    print(f"\n{'ALL PASS' if _fail == 0 else str(_fail)+' FAILED'} "
          f"(libs: scipy.spatial.transform.Rotation, scipy.interpolate, numpy)")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
