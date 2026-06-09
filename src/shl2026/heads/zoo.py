"""The head zoo — lightweight trainable classifiers on frozen embeddings.

These are the *only* trainable component allowed by the SHL 2026 frozen-FM
rule. Each head is a scikit-learn estimator wrapped in a ``StandardScaler``
pipeline so students get sane behaviour without thinking about feature scaling.
All run comfortably on CPU, which keeps the inner loop fast in JupyterHub.

Add your own idea by registering a builder in :data:`_BUILDERS`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


@runtime_checkable
class Head(Protocol):
    """Minimal fit/predict interface every head satisfies."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Head: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def _linear(seed: int, **kw: object) -> Pipeline:
    return make_pipeline(StandardScaler(), RidgeClassifier(random_state=seed, **kw))


def _logreg(seed: int, **kw: object) -> Pipeline:
    params: dict[str, object] = {"max_iter": 1000, "random_state": seed}
    params.update(kw)
    return make_pipeline(StandardScaler(), LogisticRegression(**params))


def _mlp(seed: int, **kw: object) -> Pipeline:
    params: dict[str, object] = {
        "hidden_layer_sizes": (256,),
        "max_iter": 300,
        "early_stopping": True,
        "random_state": seed,
    }
    params.update(kw)
    return make_pipeline(StandardScaler(), MLPClassifier(**params))


_BUILDERS = {
    "linear": _linear,
    "logreg": _logreg,
    "mlp": _mlp,
}


def list_heads() -> list[str]:
    """Names accepted by :func:`make_head`."""
    return sorted(_BUILDERS)


def make_head(name: str = "logreg", *, seed: int = 0, **kwargs: object) -> Pipeline:
    """Build a head by name.

    Args:
        name: one of :func:`list_heads` (``linear``, ``logreg``, ``mlp``).
        seed: random state for reproducibility.
        kwargs: forwarded to the underlying scikit-learn estimator.
    """
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(f"unknown head {name!r}; choose from {list_heads()}") from None
    return builder(seed, **kwargs)
