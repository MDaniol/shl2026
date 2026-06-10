"""Code-snapshot artifacts: every run recreatable, even from a dirty tree.

:func:`shl2026.tracking.autolog.track` stamps ``git_sha`` on each run, which
recreates *committed* runs — but a number produced from uncommitted edits
would be unrecoverable once the student edits further. On a dirty tree, the
snapshot captures the delta against HEAD into the MLflow run itself
(``code_snapshot/`` artifacts), so any run can be restored with::

    git checkout <git_sha tag>
    git apply diff.patch
    tar -xzf untracked.tar.gz

This is a safety net, not a substitute for committing: ``submit.sh`` still
refuses dirty trees, so only committed runs can be certified.
"""

from __future__ import annotations

import json
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

#: Per-file / total caps for the untracked-files archive. Snapshots ride the
#: MLflow proxied-artifact path on every dirty run; they must stay small.
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024

#: Untracked directories that are never code (local caches, venvs, stray
#: MLflow file stores). Anything .gitignore'd is already excluded by git.
_EXCLUDE_PARTS = {"mlruns", ".venv", "__pycache__", ".ipynb_checkpoints", ".dvc"}


@dataclass(frozen=True)
class SnapshotReport:
    """What was captured; written verbatim into MANIFEST.json."""

    git_sha: str
    has_patch: bool
    untracked_included: list[str] = field(default_factory=list)
    untracked_skipped: list[str] = field(default_factory=list)


def _git(args: list[str], cwd: str | Path | None = None) -> str:
    return subprocess.check_output(["git", *args], text=True, cwd=cwd)


def create_code_snapshot(dest: str | Path) -> SnapshotReport | None:
    """Write restore artifacts for the working tree's delta vs HEAD into ``dest``.

    Produces ``diff.patch`` (tracked changes, incl. staged and binary),
    ``untracked.tar.gz`` (new files git doesn't know yet, size-capped), and
    ``MANIFEST.json`` (what's included/skipped + restore commands).

    Returns:
        A :class:`SnapshotReport`, or ``None`` when there is nothing to
        snapshot (clean tree) or the cwd is not a usable git checkout.
    """
    dest = Path(dest)
    try:
        root = Path(_git(["rev-parse", "--show-toplevel"]).strip())
        git_sha = _git(["rev-parse", "HEAD"]).strip()
        if not _git(["status", "--porcelain"], cwd=root).strip():
            return None
        patch = _git(["diff", "HEAD", "--binary"], cwd=root)
        untracked = [
            p
            for p in _git(
                ["ls-files", "--others", "--exclude-standard", "-z"], cwd=root
            ).split("\0")
            if p and not (_EXCLUDE_PARTS & set(Path(p).parts))
        ]
    except Exception:
        return None  # not a git checkout / no commits yet — caller warns

    has_patch = bool(patch.strip())
    if has_patch:
        (dest / "diff.patch").write_text(patch)

    included: list[str] = []
    skipped: list[str] = []
    total = 0
    if untracked:
        with tarfile.open(dest / "untracked.tar.gz", "w:gz") as tar:
            for rel in untracked:
                f = root / rel
                size = f.stat().st_size if f.is_file() else 0
                if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
                    skipped.append(rel)
                    continue
                tar.add(f, arcname=rel)
                included.append(rel)
                total += size
        if not included:
            (dest / "untracked.tar.gz").unlink()

    report = SnapshotReport(
        git_sha=git_sha,
        has_patch=has_patch,
        untracked_included=included,
        untracked_skipped=skipped,
    )
    manifest = {
        "git_sha": git_sha,
        "has_patch": has_patch,
        "untracked_included": included,
        "untracked_skipped_size_cap": skipped,
        "restore": [
            f"git checkout {git_sha}",
            *(["git apply diff.patch"] if has_patch else []),
            *(["tar -xzf untracked.tar.gz"] if included else []),
        ],
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return report


def dump_notebook_history(dest: str | Path) -> bool:
    """If running under IPython/Jupyter, write the executed-cell history.

    A notebook's *kernel* state can differ from the ``.ipynb`` on disk
    (unsaved cells), so neither git nor the file snapshot sees the code as
    actually executed. ``In[...]`` does.

    Returns:
        True if a history file was written.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except Exception:
        return False
    if ip is None:
        return False
    cells = [c for c in ip.user_ns.get("In", []) if c.strip()]
    if not cells:
        return False
    body = "\n".join(
        f"# %% In[{i}]\n{cell}\n" for i, cell in enumerate(cells, start=1)
    )
    Path(dest, "notebook_history.py").write_text(body)
    return True
