#!/usr/bin/env python3
"""Tier-1 macro-F1 lever: label-shift prior adaptation + logit adjustment (pure NumPy).

The metric is macro-F1 on a SHUFFLED test with an UNKNOWN class prior; our train is
User-1-heavy. Two complementary, per-window, order-invariant post-hocs that the SHL field
under-uses (everyone leaned on the now-dead temporal smoothing):

  * estimate_prior_mlls : Saerens-Latinne-Decaestecker EM — estimate the EVAL prior from the
    classifier's posteriors on the **unlabeled** eval set (uses test INPUTS only, never labels),
    then adapt() reweights posteriors to it. Directly fixes our worst class (Run, weak from
    scarcity, not confusability). [Saerens 2002; MLLS, Alexandari 2020 / arXiv:1901.06852]
  * logit_adjust : post-hoc balanced-error correction using the TRAIN prior only (no test
    needed) — argmax of P * pi_train^(-tau). tau=1 = uniform target = what macro-F1 rewards.
    [Menon 2020, arXiv:2007.07314]

These compose with the existing per-class macro-F1 calibration (probe_fusion.calibrate). All
operate on class-aligned (1..8) row-normalized probabilities and return the same shape.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def class_prior(y, n_classes: int = 8) -> np.ndarray:
    """Normalized class frequencies over labels 1..n_classes."""
    c = np.bincount(np.asarray(y).astype(int), minlength=n_classes + 1)[1:n_classes + 1]
    return c / (c.sum() + _EPS)


def adapt(P: np.ndarray, pi_src: np.ndarray, pi_tgt: np.ndarray) -> np.ndarray:
    """Label-shift reweight posteriors from source prior to target prior; row-normalized.
    Identity when pi_tgt == pi_src."""
    Pa = P * (pi_tgt / (pi_src + _EPS))[None, :]
    return Pa / (Pa.sum(1, keepdims=True) + _EPS)


def estimate_prior_mlls(P: np.ndarray, pi_src: np.ndarray,
                        n_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """EM estimate of the eval-set class prior from posteriors P (trained under pi_src).
    Uses only P (model outputs on the unlabeled eval inputs) — no eval labels."""
    pi = pi_src.copy()
    for _ in range(n_iter):
        Pa = P * (pi / (pi_src + _EPS))[None, :]
        Pa /= (Pa.sum(1, keepdims=True) + _EPS)
        pi_new = Pa.mean(0)
        pi_new /= pi_new.sum() + _EPS
        if np.abs(pi_new - pi).max() < tol:
            pi = pi_new
            break
        pi = pi_new
    return pi


def logit_adjust(P: np.ndarray, pi_train: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Post-hoc logit adjustment: argmax of P * pi_train^(-tau), row-normalized.
    tau=0 -> identity; tau=1 -> balanced-error (uniform-target) optimal = macro-F1 friendly."""
    Pa = P * (pi_train + _EPS) ** (-tau)
    return Pa / (Pa.sum(1, keepdims=True) + _EPS)


def self_test() -> None:
    """Differential checks: MLLS recovers a known induced label shift; adapt/logit identities."""
    rng = np.random.default_rng(0)
    C, n = 5, 40000
    mu = np.arange(C) * 1.6                              # separated 1-D Gaussian classes
    pi_src = np.array([0.40, 0.25, 0.20, 0.10, 0.05])
    pi_tgt = np.array([0.05, 0.10, 0.20, 0.25, 0.40])   # strong shift (reversed)
    y = rng.choice(C, size=n, p=pi_tgt)                  # eval drawn from TARGET prior
    x = mu[y] + rng.standard_normal(n)
    # source posteriors p_src(c|x) ∝ pi_src * N(x; mu_c)
    like = np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2)
    P = pi_src[None, :] * like
    P /= P.sum(1, keepdims=True)
    pi_hat = estimate_prior_mlls(P, pi_src)
    err = np.abs(pi_hat - pi_tgt).max()
    assert err < 0.03, f"MLLS failed to recover induced shift: max|Δ|={err:.4f}"
    # no-shift identity
    assert np.allclose(adapt(P, pi_src, pi_src), P, atol=1e-9), "adapt not identity at no shift"
    assert np.allclose(logit_adjust(P, pi_src, 0.0), P, atol=1e-9), "logit_adjust tau=0 not identity"
    # rows stay normalized
    assert np.allclose(adapt(P, pi_src, pi_tgt).sum(1), 1.0, atol=1e-6)
    # class_prior sums to 1
    assert abs(class_prior([1, 1, 2, 3, 8]).sum() - 1.0) < 1e-9
    print(f"prior_adapt.self_test OK ✅  MLLS recovered shift to max|Δ|={err:.4f}")


if __name__ == "__main__":
    self_test()
