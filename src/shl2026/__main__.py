"""Command-line entry point.

Subcommands are thin wrappers that resolve a Hydra config and delegate to the
relevant subpackage. Heavy lifting lives in `src/shl2026/<area>/`.
"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def verify_data(manifest: str = typer.Argument(..., help="Path to MANIFEST.sha256")) -> None:
    """Re-hash raw data and compare to the manifest. Stub."""
    raise NotImplementedError("implemented in scripts/verify_data.py for now")


@app.command()
def window(config_name: str = typer.Option("default", "--config-name")) -> None:
    """Window + normalise raw sensor files. Stub."""
    raise NotImplementedError


@app.command()
def embed(config_name: str = typer.Option("default", "--config-name")) -> None:
    """Compute frozen foundation-model embeddings. Stub."""
    raise NotImplementedError


@app.command()
def train_head(config_name: str = typer.Option("default", "--config-name")) -> None:
    """Train the lightweight head on cached embeddings. Stub."""
    raise NotImplementedError


@app.command()
def predict(config_name: str = typer.Option("default", "--config-name")) -> None:
    """Run inference on the test embeddings. Stub."""
    raise NotImplementedError


@app.command()
def format_submission(
    predictions: str = typer.Argument(..., help="Path to raw model outputs"),
    out: str = typer.Option("teamName_predictions.txt", "--out"),
) -> None:
    """Format and validate the 92726 x 500 submission matrix. Stub."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    app()
