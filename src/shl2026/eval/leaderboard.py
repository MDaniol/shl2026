"""The shared leaderboard — the team's coordination surface, not git.

With 12 students pushing freely and no PRs, the way you see who is ahead and
pick the single submission is by ranking MLflow runs on validation macro-F1.
:func:`leaderboard` flattens runs across all (or selected) student experiments
into a tidy, sorted DataFrame.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

_COLUMNS = [
    "experiment",
    "run_name",
    "macro_f1",
    "accuracy",
    "fm",
    "head",
    "student",
    "git_sha",
    "git_dirty",
    "start_time",
    "run_id",
]


def leaderboard(
    experiments: list[str] | None = None,
    *,
    metric: str = "macro_f1",
    tracking_uri: str | None = None,
    max_results: int = 1000,
) -> pd.DataFrame:
    """Rank MLflow runs by ``metric`` (default validation macro-F1), descending.

    Args:
        experiments: experiment names to include; ``None`` scans them all.
        metric: metric column to sort on.
        tracking_uri: override ``$MLFLOW_TRACKING_URI``.
        max_results: cap on runs fetched.

    Returns:
        A DataFrame with one row per run. Empty (with the expected columns) if
        MLflow is unavailable or no runs exist.
    """
    empty = pd.DataFrame(columns=_COLUMNS)
    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        all_exps = mlflow.search_experiments()
        id_to_name = {e.experiment_id: e.name for e in all_exps}
        if experiments is None:
            experiments = list(id_to_name.values())
        if not experiments:
            return empty
        runs = mlflow.search_runs(
            experiment_names=experiments,
            max_results=max_results,
            output_format="pandas",
        )
    except Exception as exc:
        log.warning("leaderboard: MLflow unavailable (%s)", exc)
        return empty

    if runs.empty:
        return empty

    def col(name: str, default: object = None) -> pd.Series:
        if name in runs.columns:
            return runs[name].reset_index(drop=True)
        return pd.Series([default] * len(runs))

    table = pd.DataFrame(
        {
            "experiment": col("experiment_id").map(lambda i: id_to_name.get(i, i)),
            "run_name": col("tags.mlflow.runName"),
            "macro_f1": col(f"metrics.{metric}"),
            "accuracy": col("metrics.accuracy"),
            "fm": col("params.fm"),
            "head": col("params.head"),
            "student": col("tags.student"),
            "git_sha": col("tags.git_sha"),
            "git_dirty": col("tags.git_dirty"),
            "start_time": col("start_time"),
            "run_id": col("run_id"),
        }
    )
    return table.sort_values("macro_f1", ascending=False, na_position="last").reset_index(
        drop=True
    )
