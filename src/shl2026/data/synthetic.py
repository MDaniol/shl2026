"""Synthetic embedding cache so the inner loop runs *today*.

Before real FM embeddings exist in the shared cache, students need to exercise the
exact same code path. :func:`make_synthetic_embedding_cache` writes
class-separable fake embeddings (with a small per-location shift, mimicking
domain shift) into the cache layout that :func:`shl2026.data.embeddings`
reads. Swap ``$SHL_EMB_CACHE`` to the real shared cache and nothing changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .embeddings import (
    N_CLASSES,
    TEST_LOCATIONS,
    TRAIN_LOCATIONS,
    save_embedding_set,
)


def make_synthetic_embedding_cache(
    root: str | os.PathLike[str],
    *,
    fm: str = "synthetic",
    revision: str = "v1",
    dim: int = 32,
    n_per_class: int = 64,
    n_test: int = 256,
    separation: float = 2.5,
    seed: int = 0,
) -> Path:
    """Populate ``root`` with a small, learnable synthetic embedding cache.

    Each class gets a random centroid; samples are centroid + Gaussian noise.
    A per-location offset emulates the train/test domain shift. The test split
    is written unlabelled, exactly like the real challenge data.

    Returns the cache root as a :class:`~pathlib.Path`.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((N_CLASSES, dim)) * separation
    loc_shift = {
        loc: rng.standard_normal(dim) * 0.3 for loc in TRAIN_LOCATIONS
    }

    def _sample(n: int, classes: np.ndarray, shift: np.ndarray) -> np.ndarray:
        noise = rng.standard_normal((n, dim))
        return (centroids[classes - 1] + shift + noise).astype(np.float32)

    for split, n_each in (("train", n_per_class), ("validation", max(8, n_per_class // 4))):
        for loc in TRAIN_LOCATIONS:
            labels = np.repeat(np.arange(1, N_CLASSES + 1), n_each)
            rng.shuffle(labels)
            X = _sample(labels.size, labels, loc_shift[loc])
            save_embedding_set(
                X, labels, fm=fm, revision=revision, split=split, location=loc, root=root
            )

    # Test: single shuffled, unlabelled set mixing the test-time locations.
    test_classes = rng.integers(1, N_CLASSES + 1, size=n_test)
    test_shift = np.mean([loc_shift[loc] for loc in TEST_LOCATIONS], axis=0)
    X_test = _sample(n_test, test_classes, test_shift)
    save_embedding_set(
        X_test, None, fm=fm, revision=revision, split="test", location=None, root=root
    )
    return root
