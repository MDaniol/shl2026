"""One-line provenance for every experiment — reproducibility for free.

In a no-PR sprint, traceability has to come from *recording*, not reviewing.
:func:`track` is a context manager that, on every run, stamps the git SHA,
container hash, Slurm job id, compute node, grant, params hash, and seed into an
MLflow run — so any logged result is linkable back to the exact code, config,
and environment that produced it, with zero student effort.

It degrades gracefully: if MLflow is unreachable (server down, offline laptop),
it yields a no-op logger and prints a warning, so prototyping never breaks.

Usage::

    with track("alice", run_name="moment+mlp", seed=0,
               params={"fm": "moment", "head": "mlp"}) as run:
        head = make_head("mlp").fit(Xtr, ytr)
        run.log_eval(evaluate(head, val))
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from . import run_context

log = logging.getLogger(__name__)


def _context_tags(params_path: str | os.PathLike[str] | None) -> dict[str, str]:
    ctx = run_context.collect(params_path)
    tags = {
        "git_sha": ctx.git_sha,
        "git_dirty": str(ctx.git_dirty).lower(),
        "container_sha256": ctx.container_sha256,
        "slurm_job_id": ctx.slurm_job_id,
        "helios_node": ctx.helios_node,
        "plgrid_grant": ctx.plgrid_grant,
        "params_sha256": ctx.params_sha256,
        "student": os.environ.get("USER") or os.environ.get("USERNAME"),
        # Which Python env produced the run: the team venv = reproducible in
        # the container; a personal venv = provisional (see STUDENTS.md,
        # "Need an extra library?").
        "python_env": sys.prefix,
    }
    return {k: v for k, v in tags.items() if v is not None}


class _RunLogger:
    """Thin wrapper over the active MLflow run."""

    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    def _safe(self, fn: Any, *args: Any, **kw: Any) -> None:
        try:
            fn(*args, **kw)
        except Exception as exc:  # pragma: no cover - network/server faults
            log.warning("MLflow logging call failed (continuing): %s", exc)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._safe(self._mlflow.set_tags, dict(tags))

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._safe(self._mlflow.log_params, dict(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._safe(self._mlflow.log_metrics, {k: float(v) for k, v in metrics.items()}, step=step)

    def log_eval(self, result: Any, prefix: str = "") -> None:
        """Log an :class:`~shl2026.eval.metrics.EvalResult`."""
        metrics = {f"{prefix}macro_f1": result.macro_f1, f"{prefix}accuracy": result.accuracy}
        metrics.update({f"{prefix}class_{c}_f1": v for c, v in result.per_class_f1.items()})
        self.log_metrics(metrics)

    def log_artifact(self, path: str | os.PathLike[str]) -> None:
        self._safe(self._mlflow.log_artifact, os.fspath(path))


class _NoOpLogger:
    """Same surface as :class:`_RunLogger`, but records nothing."""

    def set_tags(self, tags: Mapping[str, str]) -> None: ...
    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None: ...
    def log_eval(self, result: Any, prefix: str = "") -> None: ...
    def log_artifact(self, path: str | os.PathLike[str]) -> None: ...


@contextmanager
def track(
    experiment: str,
    run_name: str | None = None,
    *,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
    seed: int | None = None,
    params_path: str | os.PathLike[str] | None = "params.yaml",
    tracking_uri: str | None = None,
) -> Iterator[_RunLogger | _NoOpLogger]:
    """Open an auto-provenanced MLflow run; yield a logger.

    Args:
        experiment: MLflow experiment name (convention: the student's name).
        run_name: human-readable run label.
        params: hyperparameters to log (fm, head, lr, ...).
        tags: extra tags merged over the auto-captured run-context tags.
        seed: if given, seeds Python/NumPy/PyTorch before the run and logs it.
        params_path: file whose hash anchors the params side of the contract.
        tracking_uri: override ``$MLFLOW_TRACKING_URI`` (e.g. for tests).
    """
    if seed is not None:
        run_context.set_global_seeds(seed)
    auto_tags = _context_tags(params_path)

    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
    except Exception as exc:
        log.warning("MLflow unavailable (%s); continuing without tracking.", exc)
        yield _NoOpLogger()
        return

    logger = _RunLogger(mlflow)
    try:
        logger.set_tags({**auto_tags, **(dict(tags) if tags else {})})
        if seed is not None:
            logger.log_params({"seed": seed})
        if params:
            logger.log_params(params)
        yield logger
    finally:
        try:
            mlflow.end_run()
        except Exception as exc:  # pragma: no cover
            log.warning("MLflow end_run failed: %s", exc)
