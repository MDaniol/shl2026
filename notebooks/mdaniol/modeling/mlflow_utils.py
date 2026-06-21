#!/usr/bin/env python3
"""Thin MLflow logging helper for SHL-2026 experiments (reuses the team server).

The team runs a shared MLflow server (scripts/mlflow_server.sh); the group env.sh
exports MLFLOW_TRACKING_URI, which env_mdaniol.sh sources. This module wraps every
experiment run so params/metrics/artifacts land there with a consistent schema.

Design:
  * **No-op safe** — if MLFLOW_TRACKING_URI is unset (e.g. local dev), it degrades
    to a dummy logger so scripts run unchanged; mlflow is imported only when a URI
    is present (no hard dependency off-cluster).
  * **One run per config.** Tags carry git SHA + phase/branch so runs are filterable.
  * Namespaced experiment (default "shl2026-mdaniol", override via MLFLOW_EXPERIMENT)
    so it doesn't collide on the shared server.

Usage:
    from mlflow_utils import mlflow_run, log_class_report
    with mlflow_run("global+softmoe_b0.40", params={"rep": "handcrafted", "beta": 0.40},
                    tags={"phase": "moe", "branch": "fusion"}) as run:
        run.metrics({"macro_f1_test": 0.75, "macro_f1_tune": 0.78,
                     "selection_lock_gap": 0.03})
        log_class_report(run, rep_test, prefix="test")   # per-class F1, Train/Subway
        run.artifact("artifacts/moe_b0.40.json")
"""
from __future__ import annotations

import contextlib
import os
import subprocess
from pathlib import Path

CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class _NoOp:
    """Logger used when no MLFLOW_TRACKING_URI — every call is a silent no-op."""
    active = False
    def params(self, *_a, **_k): pass
    def metric(self, *_a, **_k): pass
    def metrics(self, *_a, **_k): pass
    def artifact(self, *_a, **_k): pass
    def tags(self, *_a, **_k): pass


class _Run:
    """Live logger backed by mlflow (one open run)."""
    active = True
    def __init__(self, mlflow):
        self._mlflow = mlflow
    def params(self, d: dict): self._mlflow.log_params(d)
    def metric(self, k: str, v, step: int | None = None):
        self._mlflow.log_metric(k, float(v), step=step)
    def metrics(self, d: dict, step: int | None = None):
        self._mlflow.log_metrics({k: float(v) for k, v in d.items()}, step=step)
    def artifact(self, path):
        p = Path(path)
        if p.exists(): self._mlflow.log_artifact(str(p))
    def tags(self, d: dict): self._mlflow.set_tags(d)


@contextlib.contextmanager
def mlflow_run(run_name: str, params: dict | None = None,
               experiment: str | None = None, tags: dict | None = None):
    """Context manager yielding a logger. No-ops if MLFLOW_TRACKING_URI is unset."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print(f"[mlflow] no MLFLOW_TRACKING_URI — logging disabled for '{run_name}'")
        yield _NoOp()
        return
    import mlflow
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or os.environ.get("MLFLOW_EXPERIMENT", "shl2026-mdaniol"))
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"git_sha": _git_sha(), **(tags or {})})
        if params:
            mlflow.log_params(params)
        print(f"[mlflow] logging run '{run_name}' -> {uri}")
        yield _Run(mlflow)


def log_class_report(run, rep: dict, prefix: str = "test") -> None:
    """Log macro + per-class F1 (+ Train/Subway explicitly) from metrics.class_report."""
    if not getattr(run, "active", False):
        return
    m = {f"{prefix}_macro_f1": rep["macro_f1"]}
    for name, d in rep.get("per_class", {}).items():
        m[f"{prefix}_f1_{name}"] = d["f1"]
    run.metrics(m)
