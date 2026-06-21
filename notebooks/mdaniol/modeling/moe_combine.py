#!/usr/bin/env python3
"""Pure-NumPy combiners for the location mixture-of-experts (no training here).

Consumes class-aligned, row-normalized probabilities (see probe_fusion.aligned_proba):
  P_global : (n, 8)         global model
  P_by_loc : {loc: (n, 8)}  per-location experts, evaluated on the SAME frames
  q        : (n, L)         router posteriors over EXPERT_LOCS (rows sum to 1);
                            for the ORACLE, q is one-hot at the true location.

All mixes return (n, 8) row-normalized probabilities; `predict` -> labels 1..8.
Everything is deterministic and side-effect-free so the β / threshold sweeps are
cheap re-combinations of cached probabilities.
"""
from __future__ import annotations

import numpy as np

CLASSES = list(range(1, 9))
EXPERT_LOCS = ("Bag", "Hips", "Torso")   # test locations (no Hand)


def stack_experts(P_by_loc: dict, locs=EXPERT_LOCS) -> np.ndarray:
    """{loc: (n,8)} -> (L, n, 8) in `locs` order."""
    return np.stack([P_by_loc[l] for l in locs], axis=0)


def uniform_mix(P: np.ndarray) -> np.ndarray:
    """Equal-weight average of experts. P: (L,n,8) -> (n,8)."""
    return P.mean(axis=0)


def soft_mix(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Soft MoE: P_moe = sum_l q_l * P_l. P:(L,n,8), q:(n,L) -> (n,8)."""
    return np.einsum("lnc,nl->nc", P, q)


def hard_mix(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hard router: pick the argmax-q expert per frame. P:(L,n,8), q:(n,L) -> (n,8)."""
    idx = q.argmax(axis=1)
    return P[idx, np.arange(P.shape[1])]


def global_fallback(P_global: np.ndarray, P_moe: np.ndarray, beta) -> np.ndarray:
    """P_final = beta*P_global + (1-beta)*P_moe. beta: scalar or (n,1)."""
    beta = np.asarray(beta, dtype=np.float64)
    return beta * P_global + (1.0 - beta) * P_moe


def adaptive_beta(q: np.ndarray, thresh: float = 0.65,
                  beta_unsure: float = 0.70, beta_sure: float = 0.35) -> np.ndarray:
    """Confidence-adaptive beta (n,1): if max(q) < thresh trust the global model
    more (beta_unsure), else trust the MoE more (beta_sure)."""
    return np.where(q.max(axis=1, keepdims=True) < thresh, beta_unsure, beta_sure)


def predict(P: np.ndarray) -> np.ndarray:
    """(n,8) probabilities -> integer labels 1..8."""
    return np.asarray(CLASSES)[P.argmax(axis=1)]


def self_test() -> None:
    rng = np.random.default_rng(0)
    n, L, C = 200, 3, 8
    def norm(a): return a / a.sum(-1, keepdims=True)
    P_by = {loc: norm(rng.random((n, C))) for loc in EXPERT_LOCS}
    P = stack_experts(P_by)
    P_global = norm(rng.random((n, C)))
    true_loc = rng.integers(0, L, n)
    q_oracle = np.eye(L)[true_loc]                       # one-hot true location

    # shapes + normalization
    for M in (uniform_mix(P), soft_mix(P, q_oracle), hard_mix(P, q_oracle),
              global_fallback(P_global, soft_mix(P, q_oracle), 0.4)):
        assert M.shape == (n, C)
        assert np.allclose(M.sum(1), 1.0, atol=1e-9), "mix not normalized"

    # oracle soft == hard == the true-location expert (one-hot q)
    expert_pick = np.stack([P_by[EXPERT_LOCS[true_loc[i]]][i] for i in range(n)])
    assert np.allclose(soft_mix(P, q_oracle), expert_pick, atol=1e-9)
    assert np.allclose(hard_mix(P, q_oracle), expert_pick, atol=1e-9)

    # adaptive beta picks the two regimes correctly
    q_soft = norm(rng.random((n, L)))
    b = adaptive_beta(q_soft)
    assert np.all((b == 0.70) == (q_soft.max(1, keepdims=True) < 0.65))

    # uniform == soft with uniform q
    q_unif = np.full((n, L), 1.0 / L)
    assert np.allclose(uniform_mix(P), soft_mix(P, q_unif), atol=1e-9)
    print("moe_combine self_test OK — mixes normalized; oracle==true-expert; "
          "adaptive-beta regimes correct; uniform==soft(1/L)")


if __name__ == "__main__":
    self_test()
