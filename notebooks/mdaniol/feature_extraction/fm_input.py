#!/usr/bin/env python3
"""Preprocess raw SHL windows into FROZEN-FM input tensors (per-model contracts).

This is the **FM-input branch** (distinct from `shl_features.py`, which produces
the 520 scalar features for the fusion head). Here we keep everything a **time
series** and package it to exactly what each frozen foundation model expects.

Two layers, per `PREPROCESSING_PLAN.md`:

  1. `build_channel_library()` — derive a library of candidate 1-D channels from
     the 3 tri-axial sensors, reusing the **already-validated** primitives in
     `shl_features` (`_lowpass_gravity` = SOS 0.3 Hz, `_mag`, `_deriv`). Keeps
     axes AND magnitudes (the feature module keeps only magnitudes).

  2. `pack_<model>(lib, variant)` — select channels for an input VARIANT and apply
     the model's MANDATORY contract (length/rate/scaling). Normalization policy is
     model-specific and literature-grounded:
       - MOMENT  : no external norm (applies RevIN internally); T=512, left-pad. [arXiv:2402.03885]
       - Mantis  : no external norm (internal per-instance z-score); 512 via interp; <=10 ch. [arXiv:2502.15637]
       - UniMTS  : 6 ch (acc+gyr, m/s^2), NO magnetometer; 10 s window. [arXiv:2410.19818 + repo]
       - LIMU-BERT: bespoke scaling (acc/9.8, mag->L2-unit*2, gyr raw); 20 Hz/120; z-score HURTS. [SenSys'21 + repo]

INPUT VARIANTS (ablation axis — chosen later by val macro-F1):
  V0 raw       : 9 raw axes (baseline; let internal norm handle it)
  V1 body      : gravity-removed acc + gyr + mag (9)
  V2 invariant : magnitudes + inter-vector angle (5; orientation/position-invariant)
  V3 augmented : V1 + magnitudes + jerk/mag-rate (16; max cues, channel-independent FMs only)

SHL units (verified from data): accelerometer in m/s^2 (|acc| mean ~9.8), so no
unit conversion is needed for UniMTS (m/s^2) or LIMU-BERT (acc/9.8).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

import shl_features as shl

FS, N = shl.FS, shl.N           # 100 Hz, 500 samples
_EPS = 1e-12

# ---------------------------------------------------------------------------
# Layer 1: candidate channel library
# ---------------------------------------------------------------------------
def _vector_angle(v: np.ndarray) -> np.ndarray:
    """Angle (rad) between consecutive 3-vectors — orientation-invariant. (n,3,N)->(n,N)."""
    a, b = v[:, :, 1:], v[:, :, :-1]
    dot = np.sum(a * b, axis=1)
    cos = dot / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + _EPS)
    ang = np.arccos(np.clip(cos, -1.0, 1.0))           # (n, N-1)
    return np.pad(ang, ((0, 0), (1, 0)), mode="edge")   # back to N


def build_channel_library(acc: np.ndarray, gyr: np.ndarray,
                          mag: np.ndarray) -> dict[str, np.ndarray]:
    """Derive all candidate 1-D channels. acc/gyr/mag: (n,3,N). Returns {name:(n,N)}."""
    grav = shl._lowpass_gravity(acc)        # SOS 0.3 Hz, validated
    body = acc - grav
    lib: dict[str, np.ndarray] = {}
    for i, ax in enumerate("xyz"):
        lib[f"Acc_{ax}"] = acc[:, i]
        lib[f"Gyr_{ax}"] = gyr[:, i]
        lib[f"Mag_{ax}"] = mag[:, i]
        lib[f"BodyAcc_{ax}"] = body[:, i]
        lib[f"Grav_{ax}"] = grav[:, i]
    lib["AccMag"] = shl._mag(acc)
    lib["GyrMag"] = shl._mag(gyr)
    lib["MagMag"] = shl._mag(mag)
    lib["BodyAccMag"] = shl._mag(body)
    lib["AccJerkMag"] = shl._mag(shl._deriv(acc))
    lib["GyrJerkMag"] = shl._mag(shl._deriv(gyr))
    lib["MagRateMag"] = shl._mag(shl._deriv(mag))
    lib["AccAngle"] = _vector_angle(acc)
    return lib


# Input variants (channel selections). Order matters for downstream concat.
VARIANTS: dict[str, list[str]] = {
    "V0": ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z",
           "Mag_x", "Mag_y", "Mag_z"],
    "V1": ["BodyAcc_x", "BodyAcc_y", "BodyAcc_z", "Gyr_x", "Gyr_y", "Gyr_z",
           "Mag_x", "Mag_y", "Mag_z"],
    "V2": ["AccMag", "GyrMag", "MagMag", "BodyAccMag", "AccAngle"],
    "V3": ["BodyAcc_x", "BodyAcc_y", "BodyAcc_z", "Gyr_x", "Gyr_y", "Gyr_z",
           "Mag_x", "Mag_y", "Mag_z", "AccMag", "GyrMag", "MagMag",
           "BodyAccMag", "AccJerkMag", "GyrJerkMag", "MagRateMag"],
}
# UniMTS can only take acc+gyr (6 ch, no magnetometer).
VARIANTS_UNIMTS = {
    "V0": ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z"],
    "V1": ["BodyAcc_x", "BodyAcc_y", "BodyAcc_z", "Gyr_x", "Gyr_y", "Gyr_z"],
}


def _stack(lib: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """(n,N) channels -> (n, C, N)."""
    return np.stack([lib[n] for n in names], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Length / rate adapters
# ---------------------------------------------------------------------------
def _pad_left(x: np.ndarray, L: int) -> np.ndarray:
    """Left zero-pad (or right-crop) the last axis to length L. MOMENT contract."""
    T = x.shape[-1]
    if T == L:
        return x
    if T > L:
        return x[..., -L:]
    out = np.zeros((*x.shape[:-1], L), dtype=x.dtype)
    out[..., L - T:] = x
    return out


def _pad_right(x: np.ndarray, L: int) -> np.ndarray:
    """Right zero-pad (or crop) the last axis to length L. UniMTS 10 s window."""
    T = x.shape[-1]
    if T == L:
        return x
    if T > L:
        return x[..., :L]
    out = np.zeros((*x.shape[:-1], L), dtype=x.dtype)
    out[..., :T] = x
    return out


def _interp_to(x: np.ndarray, L: int) -> np.ndarray:
    """Linear time-interpolate the last axis to length L. Mantis contract."""
    T = x.shape[-1]
    if T == L:
        return x
    f = interp1d(np.linspace(0.0, 1.0, T), x, axis=-1, kind="linear")
    return f(np.linspace(0.0, 1.0, L)).astype(x.dtype)


def _resample_rate(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Anti-aliased polyphase resample along the last axis (e.g. 100->20 Hz)."""
    g = np.gcd(int(fs_in), int(fs_out))
    return sp_signal.resample_poly(x, fs_out // g, fs_in // g, axis=-1).astype(x.dtype)


# ---------------------------------------------------------------------------
# Layer 2: per-model packers  (return (tensor, channel_names))
# ---------------------------------------------------------------------------
def pack_moment(lib, variant: str = "V0"):
    """MOMENT: channels as independent univariate, T=512 left-padded, NO external norm."""
    names = VARIANTS[variant]
    return _pad_left(_stack(lib, names), 512), names


def pack_mantis(lib, variant: str = "V0"):
    """Mantis: independent univariate, length 512 via interpolation, NO external norm.
    Channel-independent No-Adapter path requires <=10 channels."""
    names = VARIANTS[variant]
    if len(names) > 10:
        raise ValueError(f"Mantis No-Adapter path needs <=10 channels; "
                         f"variant {variant} has {len(names)} (use an adapter).")
    return _interp_to(_stack(lib, names), 512), names


def pack_unimts(lib, variant: str = "V0", window_s: float = 10.0):
    """UniMTS: 6 ch (acc+gyr, m/s^2), no magnetometer; pad to a `window_s` window @100 Hz."""
    if variant not in VARIANTS_UNIMTS:
        raise ValueError(f"UniMTS supports variants {list(VARIANTS_UNIMTS)} (acc+gyr only)")
    names = VARIANTS_UNIMTS[variant]
    L = int(round(window_s * FS))                 # 10 s @ 100 Hz = 1000
    return _pad_right(_stack(lib, names), L), names   # acc already m/s^2


def pack_limu(lib, variant: str = "V0", fs_out: int = 20, length: int = 120):
    """LIMU-BERT: 9 ch raw order [acc,gyr,mag]; resample 100->20 Hz; pad to `length`;
    apply LIMU scaling (acc/9.8, mag->L2-unit*alpha, gyr raw). NO z-score."""
    if variant not in ("V0", "V1"):
        raise ValueError("LIMU-BERT uses V0 (raw) or V1 (body-acc); other variants N/A")
    names = VARIANTS[variant]                      # [acc(3), gyr(3), mag(3)]
    x = _stack(lib, names)                          # (n,9,500)
    x = _resample_rate(x, int(FS), fs_out)         # (n,9,~100)
    x = _pad_right(x, length)                       # (n,9,120)  [caveat: 100->120 pad]
    return _limu_scale(x), names


def _limu_scale(x: np.ndarray, acc_g: float = 9.8, alpha: float = 2.0) -> np.ndarray:
    """LIMU-BERT normalization (Eq.1): acc/=9.8; mag-> alpha * unit-vector; gyr untouched.
    Channel order assumed [acc_x,y,z, gyr_x,y,z, mag_x,y,z]."""
    x = x.copy()
    x[:, 0:3] /= acc_g
    mag = x[:, 6:9]
    norm = np.linalg.norm(mag, axis=1, keepdims=True) + _EPS
    x[:, 6:9] = alpha * mag / norm
    return x


def pack_ast(lib, variant: str = "V2"):
    """AST / audio-spectrogram FM: return the RAW channel signals (n, C, 500) @100 Hz. NO length
    adapter here — the embedder computes a per-channel log-spectrogram over the IMU band (0-50 Hz,
    not the 16 kHz audio mel front-end) and applies the frozen AST ViT downstream. Default V2
    (orientation-invariant magnitudes) to bound the per-channel spectrogram+ViT cost."""
    names = VARIANTS[variant]
    return _stack(lib, names), names


def pack_imagebind(lib, variant: str = "V0"):
    """ImageBind native IMU encoder (arXiv:2305.05665): 6-ch acc+gyr, interp 500->2000 (its 5 s /
    2000-sample contract), per-channel **MEAN-SUBTRACTION** — ImageBind's training-time IMU
    preprocessing is zero-mean per axis, NOT z-norm (amplitude is signal for transport). No
    magnetometer (ImageBind IMU = acc+gyr only). Returns (n, 6, 2000)."""
    names = VARIANTS_UNIMTS["V0"]                      # Acc_xyz + Gyr_xyz, no mag
    x = _interp_to(_stack(lib, names), 2000)           # (n, 6, 2000)
    x = x - x.mean(axis=-1, keepdims=True)             # per-channel mean-subtraction (per window)
    return x.astype(np.float32), names


PACKERS = {"moment": pack_moment, "mantis": pack_mantis, "unimts": pack_unimts,
           "limu": pack_limu, "ast": pack_ast, "imagebind": pack_imagebind}


# ---------------------------------------------------------------------------
# Self-test: contracts (shape / channels / scaling) on synthetic + structural checks
# ---------------------------------------------------------------------------
def self_test() -> None:
    rng = np.random.default_rng(0)
    n = 8
    t = np.arange(N) / FS
    acc = np.zeros((n, 3, N)); acc[:, 2] = 9.8           # gravity on z
    acc[:, 0] += np.sin(2 * np.pi * 2.0 * t)             # 2 Hz body motion on x
    gyr = 0.1 * rng.standard_normal((n, 3, N))
    mag = 40.0 + rng.standard_normal((n, 3, N))
    lib = build_channel_library(acc, gyr, mag)

    # library completeness & shapes
    assert all(v.shape == (n, N) for v in lib.values())
    assert "AccMag" in lib and "BodyAcc_x" in lib and "AccAngle" in lib

    # gravity split sanity: Grav_z ~ 9.8 (gravity on z), body z ~ 0 (gravity removed)
    assert abs(lib["Grav_z"].mean() - 9.8) < 0.6, "gravity z not ~9.8"
    assert abs(np.mean(lib["BodyAcc_z"])) < 0.5, "gravity not removed from body z"

    # MOMENT: (n,9,512), left-padded -> first 12 cols zero
    Xm, nm = pack_moment(lib, "V0")
    assert Xm.shape == (n, 9, 512) and np.allclose(Xm[..., :12], 0.0)

    # Mantis: (n,9,512) interpolated (not zero-padded -> first col nonzero generally)
    Xa, _ = pack_mantis(lib, "V0")
    assert Xa.shape == (n, 9, 512)
    try:
        pack_mantis(lib, "V3"); raise AssertionError("V3>10ch should be rejected")
    except ValueError:
        pass

    # UniMTS: (n,6,1000), no magnetometer, right-padded
    Xu, nu = pack_unimts(lib, "V0")
    assert Xu.shape == (n, 6, 1000) and not any("Mag" in c for c in nu)
    assert np.allclose(Xu[..., 500:], 0.0), "UniMTS should right-pad 500->1000"

    # LIMU: (n,9,120) @20 Hz; acc scaled by 1/9.8 (gravity z -> ~1); mag -> L2 norm 2
    Xl, _ = pack_limu(lib, "V0")
    assert Xl.shape == (n, 9, 120)
    valid = Xl[:, :, :100]                                # pre-pad region
    assert abs(np.mean(Xl[:, 2, :100]) - 1.0) < 0.2, "acc/9.8 scaling wrong (z~1)"
    magnorm = np.linalg.norm(Xl[:, 6:9, :100], axis=1)   # per-timestep mag L2
    assert abs(magnorm.mean() - 2.0) < 0.05, f"mag L2 should be ~2, got {magnorm.mean():.3f}"

    # finiteness across all packers / variants
    for name, fn in PACKERS.items():
        vs = VARIANTS_UNIMTS if name == "unimts" else (
            ["V0", "V1"] if name == "limu" else ["V0", "V1", "V2"])
        for v in vs:
            X, _ = fn(lib, v)
            assert np.all(np.isfinite(X)), f"{name}/{v} non-finite"

    print("fm_input self_test OK — contracts (shape/channels/scaling) verified")


if __name__ == "__main__":
    self_test()
