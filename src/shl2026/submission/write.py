"""Write the challenge submission matrix from per-frame predictions.

A frozen-FM + head pipeline produces one class per *frame*. The challenge
expects a 92726 x 500 matrix — one label per sample (timepoint). The standard
baseline broadcasts each frame's predicted class across its 500 samples. The
written file is immediately re-validated by :mod:`shl2026.submission.validate`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .validate import EXPECTED_COLS, ValidationReport, validate


def write_submission(
    frame_predictions: np.ndarray,
    out: str | Path,
    *,
    n_cols: int = EXPECTED_COLS,
    validate_after: bool = True,
) -> ValidationReport:
    """Broadcast per-frame predictions to an (n_frames x n_cols) matrix and save.

    Args:
        frame_predictions: 1-D array of integer class labels, one per frame.
        out: destination path for the plain-text matrix.
        n_cols: samples per frame (500 for the SHL challenge).
        validate_after: run the strict validator on the written file.

    Returns:
        The :class:`ValidationReport` for the written file (``ok`` may be False
        if, e.g., the row count differs from the official 92726 — surfaced, not
        silently accepted).
    """
    preds = np.asarray(frame_predictions)
    if preds.ndim != 1:
        raise ValueError(f"expected 1-D per-frame predictions, got shape {preds.shape}")
    matrix = np.repeat(preds[:, None], n_cols, axis=1).astype(int)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, matrix, fmt="%d")
    if validate_after:
        return validate(out)
    return ValidationReport(
        ok=True,
        path=out,
        n_rows=int(matrix.shape[0]),
        n_cols=int(matrix.shape[1]),
        dtype=str(matrix.dtype),
        unique_labels=tuple(sorted({int(x) for x in np.unique(matrix)})),
        n_nans=0,
        messages=(),
    )
