"""Capture the per-run context required by the reproducibility contract.

Every Slurm job calls :func:`bind_run_context` once, before doing any work, to
attach git SHA, container SHA-256, Slurm job ID, compute node, PLGrid grant,
and the resolved Hydra config to the active MLflow run.
"""

from __future__ import annotations

import hashlib
import os
import random
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    git_sha: str
    git_dirty: bool
    container_sha256: str | None
    slurm_job_id: str | None
    helios_node: str
    plgrid_grant: str | None
    params_sha256: str | None


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def collect(params_path: str | os.PathLike[str] | None = "params.yaml") -> RunContext:
    """Snapshot the current process's reproducibility-relevant context."""
    try:
        git_sha = _git(["rev-parse", "HEAD"])
        git_dirty = bool(_git(["status", "--porcelain"]))
    except Exception:
        git_sha, git_dirty = "unknown", True

    params_sha = None
    if params_path and Path(params_path).exists():
        params_sha = hashlib.sha256(Path(params_path).read_bytes()).hexdigest()

    return RunContext(
        git_sha=git_sha,
        git_dirty=git_dirty,
        container_sha256=os.environ.get("SHL_SIF_SHA256"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        helios_node=socket.gethostname(),
        plgrid_grant=os.environ.get("PLGRID_GRANT"),
        params_sha256=params_sha,
    )


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch deterministically."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
