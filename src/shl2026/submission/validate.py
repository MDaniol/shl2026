"""Strict validator for the SHL 2026 submission matrix.

Per the challenge specification (retrieved 2026-05-25): the submission file
must be a plain-text matrix of shape 92726 x 500 with integer class labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXPECTED_ROWS = 92_726
EXPECTED_COLS = 500
# The 8 SHL transportation/locomotion classes. Class numbering convention
# follows previous SHL editions; verify against challenge documentation before
# any submission.
ALLOWED_LABELS: frozenset[int] = frozenset(range(1, 9))


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    path: Path
    n_rows: int
    n_cols: int
    dtype: str
    unique_labels: tuple[int, ...]
    n_nans: int
    messages: tuple[str, ...]


def validate(path: str | Path) -> ValidationReport:
    """Validate ``path`` as an SHL 2026 submission matrix.

    Returns a :class:`ValidationReport`; ``ok`` is True only if shape, dtype,
    label-set, and NaN checks all pass. Does not raise.
    """
    p = Path(path)
    msgs: list[str] = []
    arr = np.loadtxt(p)
    if arr.ndim != 2:
        msgs.append(f"expected 2-D matrix, got {arr.ndim}-D")
    n_rows, n_cols = (arr.shape if arr.ndim == 2 else (-1, -1))
    if n_rows != EXPECTED_ROWS:
        msgs.append(f"expected {EXPECTED_ROWS} rows, got {n_rows}")
    if n_cols != EXPECTED_COLS:
        msgs.append(f"expected {EXPECTED_COLS} cols, got {n_cols}")
    n_nans = int(np.isnan(arr).sum())
    if n_nans:
        msgs.append(f"{n_nans} NaN entries")
    is_int = np.all(arr == arr.astype(int))
    if not is_int:
        msgs.append("non-integer entries present")
    uniq = tuple(sorted({int(x) for x in np.unique(arr.astype(int)) if not np.isnan(x)}))
    bad = set(uniq) - ALLOWED_LABELS
    if bad:
        msgs.append(f"labels outside allowed set: {sorted(bad)}")
    return ValidationReport(
        ok=not msgs,
        path=p,
        n_rows=n_rows,
        n_cols=n_cols,
        dtype=str(arr.dtype),
        unique_labels=uniq,
        n_nans=n_nans,
        messages=tuple(msgs),
    )
