"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shl2026.submission.validate import EXPECTED_COLS, EXPECTED_ROWS


@pytest.fixture
def tiny_frames() -> np.ndarray:
    """A tiny stand-in for a sensor-frame matrix: 16 frames x 500 samples."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((16, 500)).astype(np.float32)


@pytest.fixture
def valid_submission_file(tmp_path: Path) -> Path:
    """A correctly shaped submission file with a single class label everywhere."""
    arr = np.ones((EXPECTED_ROWS, EXPECTED_COLS), dtype=int)
    p = tmp_path / "valid.txt"
    np.savetxt(p, arr, fmt="%d")
    return p
