"""Evaluation against the challenge metric.

The SHL challenge ranks submissions by **macro-averaged F1** over the eight
classes. :func:`evaluate` is the one call students use to score a trained head
on a labelled split; it also surfaces accuracy and a per-class F1 breakdown so
weak classes are visible at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..data.embeddings import N_CLASSES, EmbeddingSet
from ..heads.zoo import Head

DEFAULT_LABELS = tuple(range(1, N_CLASSES + 1))


@dataclass(frozen=True)
class EvalResult:
    """Outcome of scoring predictions against ground truth."""

    macro_f1: float
    accuracy: float
    per_class_f1: dict[int, float]
    labels: tuple[int, ...]
    confusion: np.ndarray
    n: int

    def summary(self) -> str:
        """One-line headline plus the weakest class, for quick prototyping."""
        worst = min(self.per_class_f1.items(), key=lambda kv: kv[1])
        return (
            f"macro-F1={self.macro_f1:.4f}  acc={self.accuracy:.4f}  "
            f"n={self.n}  weakest=class {worst[0]} (F1={worst[1]:.3f})"
        )

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[int, ...] = DEFAULT_LABELS,
) -> EvalResult:
    """Score predictions with the challenge's macro-F1 (and friends)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return EvalResult(
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class_f1={int(c): float(s) for c, s in zip(labels, per, strict=True)},
        labels=labels,
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
        n=int(y_true.shape[0]),
    )


def evaluate(
    head: Head,
    eval_set: EmbeddingSet,
    labels: tuple[int, ...] = DEFAULT_LABELS,
) -> EvalResult:
    """Predict with ``head`` on a labelled :class:`EmbeddingSet` and score it.

    Raises if ``eval_set`` has no labels (e.g. the test split) — there is
    nothing to score against there.
    """
    if eval_set.y is None:
        raise ValueError(
            f"{eval_set.split!r} split has no labels; evaluate on 'validation' instead."
        )
    y_pred = head.predict(eval_set.X)
    return evaluate_predictions(eval_set.y, y_pred, labels=labels)
