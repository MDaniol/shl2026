"""Submission validator: shape, label range, NaN handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shl2026.submission.validate import EXPECTED_COLS, EXPECTED_ROWS, validate


def test_valid_submission(valid_submission_file: Path) -> None:
    rep = validate(valid_submission_file)
    assert rep.ok, rep.messages
    assert rep.n_rows == EXPECTED_ROWS
    assert rep.n_cols == EXPECTED_COLS
    assert rep.n_nans == 0


def test_wrong_shape(tmp_path: Path) -> None:
    arr = np.ones((10, 10), dtype=int)
    p = tmp_path / "bad.txt"
    np.savetxt(p, arr, fmt="%d")
    rep = validate(p)
    assert not rep.ok
    assert any("rows" in m for m in rep.messages)
    assert any("cols" in m for m in rep.messages)


def test_out_of_range_label(tmp_path: Path) -> None:
    arr = np.full((EXPECTED_ROWS, EXPECTED_COLS), 99, dtype=int)
    p = tmp_path / "oor.txt"
    np.savetxt(p, arr, fmt="%d")
    rep = validate(p)
    assert not rep.ok
    assert any("outside allowed set" in m for m in rep.messages)
