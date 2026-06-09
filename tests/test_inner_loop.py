"""The student inner loop must work end-to-end on synthetic embeddings.

This is the contract the whole skeleton exists to guarantee: load shared
embeddings -> train a head -> score on validation -> log with provenance ->
emit a submission. It runs in milliseconds, CPU-only, no real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shl2026 import embeddings, evaluate, make_head, write_submission
from shl2026.data.embeddings import list_available
from shl2026.data.synthetic import make_synthetic_embedding_cache
from shl2026.heads.zoo import list_heads
from shl2026.tracking.autolog import track


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return make_synthetic_embedding_cache(tmp_path / "emb", dim=16, n_per_class=48, seed=1)


def test_cache_is_discoverable(cache: Path) -> None:
    assert ("synthetic", "v1") in list_available(cache)


def test_train_split_loads_and_concatenates_locations(cache: Path) -> None:
    train = embeddings("synthetic", "train", root=cache)
    assert train.location is None
    assert train.has_labels
    assert train.dim == 16
    # 8 classes x 48 per class x 4 locations
    assert train.n == 8 * 48 * 4
    assert set(np.unique(train.y)) <= set(range(1, 9))


def test_test_split_is_unlabelled(cache: Path) -> None:
    test = embeddings("synthetic", "test", root=cache)
    assert not test.has_labels
    assert test.y is None


def test_inner_loop_learns(cache: Path) -> None:
    train = embeddings("synthetic", "train", root=cache)
    val = embeddings("synthetic", "validation", root=cache)
    head = make_head("logreg", seed=0).fit(train.X, train.y)
    result = evaluate(head, val)
    # Synthetic data is class-separable; a linear head should do well.
    assert result.macro_f1 > 0.8
    assert len(result.per_class_f1) == 8
    assert isinstance(result.summary(), str)


@pytest.mark.parametrize("name", list_heads())
def test_every_head_runs(cache: Path, name: str) -> None:
    train = embeddings("synthetic", "train", root=cache)
    val = embeddings("synthetic", "validation", root=cache)
    head = make_head(name, seed=0).fit(train.X, train.y)
    assert evaluate(head, val).macro_f1 > 0.5


def test_evaluate_refuses_unlabelled(cache: Path) -> None:
    train = embeddings("synthetic", "train", root=cache)
    test = embeddings("synthetic", "test", root=cache)
    head = make_head("linear").fit(train.X, train.y)
    with pytest.raises(ValueError, match="no labels"):
        evaluate(head, test)


def test_track_degrades_without_breaking(cache: Path, tmp_path: Path) -> None:
    train = embeddings("synthetic", "train", root=cache)
    val = embeddings("synthetic", "validation", root=cache)
    uri = f"file://{tmp_path / 'mlruns'}"
    with track("pytest", run_name="smoke", seed=0,
               params={"fm": "synthetic", "head": "linear"}, tracking_uri=uri) as run:
        head = make_head("linear", seed=0).fit(train.X, train.y)
        run.log_eval(evaluate(head, val))  # must not raise


def test_submission_writer_broadcasts_and_validates(tmp_path: Path) -> None:
    preds = np.ones(10, dtype=int)  # wrong row count on purpose
    report = write_submission(preds, tmp_path / "sub.txt")
    assert report.n_cols == 500
    assert report.n_rows == 10
    # Row count != 92726 -> validator flags it rather than silently passing.
    assert not report.ok
