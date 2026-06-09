"""Command-line entry point.

The batch/outer-loop face of the SDK: the same functions students call
interactively, wired as ``shl2026 <verb>`` so the DVC pipeline and Slurm jobs
can certify a result reproducibly. The fast inner loop lives in notebooks; the
CLI is for jobs that need to be recorded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command(name="verify-data")
def verify_data(manifest: str = typer.Argument(..., help="Path to MANIFEST.sha256")) -> None:
    """Re-hash raw data and compare to the manifest. Stub."""
    raise NotImplementedError("implemented in scripts/verify_data.py for now")


@app.command(name="window")
def window(config_name: str = typer.Option("default", "--config-name")) -> None:
    """Window + normalise raw sensor files. Stub."""
    raise NotImplementedError


@app.command(name="embed")
def embed(config_name: str = typer.Option("default", "--config-name")) -> None:
    """Compute frozen foundation-model embeddings. Stub (needs an FM adapter)."""
    raise NotImplementedError


@app.command(name="train-head")
def train_head(
    fm: str = typer.Option(..., "--fm", help="Foundation-model id in the embedding cache"),
    head: str = typer.Option("logreg", "--head", help="Head name (linear/logreg/mlp)"),
    revision: str | None = typer.Option(None, "--revision"),
    seed: int = typer.Option(0, "--seed"),
    experiment: str = typer.Option("cli", "--experiment", help="MLflow experiment"),
    out: Path = typer.Option(Path("runs/head.joblib"), "--out", help="Where to save the head"),
) -> None:
    """Train a head on cached embeddings, score on validation, log to MLflow."""
    import joblib

    from .data.embeddings import embeddings
    from .eval.metrics import evaluate
    from .heads.zoo import make_head
    from .tracking.autolog import track

    train = embeddings(fm, "train", revision=revision)
    val = embeddings(fm, "validation", revision=revision)
    with track(experiment, run_name=f"{fm}+{head}", seed=seed,
               params={"fm": fm, "head": head, "revision": train.revision}) as run:
        model = make_head(head, seed=seed).fit(train.X, train.y)
        result = evaluate(model, val)
        run.log_eval(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"head": model, "fm": fm, "revision": train.revision}, out)
    typer.echo(result.summary())
    typer.echo(f"saved head -> {out}")


@app.command(name="predict")
def predict(
    head_path: Path = typer.Option(..., "--head-path", help="Head saved by train-head"),
    revision: str | None = typer.Option(None, "--revision"),
    out: Path = typer.Option(Path("runs/test_frame_predictions.npy"), "--out"),
) -> None:
    """Apply a trained head to the test embeddings; save per-frame predictions."""
    import joblib

    from .data.embeddings import embeddings

    bundle = joblib.load(head_path)
    test = embeddings(bundle["fm"], "test", revision=revision or bundle.get("revision"))
    preds = np.asarray(bundle["head"].predict(test.X)).astype(int)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, preds)
    typer.echo(f"predicted {preds.size} frames -> {out}")


@app.command(name="format-submission")
def format_submission(
    predictions: str = typer.Argument(..., help="Per-frame predictions (.npy or text)"),
    out: str = typer.Option("teamName_predictions.txt", "--out"),
) -> None:
    """Broadcast per-frame predictions to the 92726 x 500 matrix and validate."""
    from .submission.write import write_submission

    p = Path(predictions)
    preds = np.load(p) if p.suffix == ".npy" else np.loadtxt(p, dtype=int)
    report = write_submission(np.ravel(preds), out)
    typer.echo(f"wrote {out}: {report.n_rows}x{report.n_cols} ok={report.ok}")
    if not report.ok:
        for msg in report.messages:
            typer.echo(f"  - {msg}")
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
