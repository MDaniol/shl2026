"""End-to-end smoke: importable package + CLI present."""

from __future__ import annotations


def test_package_imports() -> None:
    import shl2026

    assert shl2026.__version__


def test_cli_registers_commands() -> None:
    from shl2026.__main__ import app

    cmds = {c.name for c in app.registered_commands}
    assert {"verify-data", "window", "embed", "train-head", "predict", "format-submission"} <= cmds
