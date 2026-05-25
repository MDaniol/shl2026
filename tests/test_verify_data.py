"""Manifest verification: detect any single-byte corruption."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _make_raw(root: Path) -> None:
    (root / "Bag").mkdir(parents=True)
    (root / "Bag" / "Acc_x.txt").write_text("1 2 3\n4 5 6\n")
    (root / "Bag" / "Label.txt").write_text("1 1 1\n2 2 2\n")


def _verify(script: Path, root: Path) -> Path:
    manifest = root / "MANIFEST.sha256"
    subprocess.run(
        [sys.executable, str(script), "generate", "--root", str(root), "--out", str(manifest)],
        check=True,
    )
    return manifest


def test_clean_data_verifies(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _make_raw(root)
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_data.py"
    manifest = _verify(script, root)
    out = subprocess.run(
        [sys.executable, str(script), "verify", "--manifest", str(manifest)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr


def test_corrupted_byte_detected(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _make_raw(root)
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_data.py"
    manifest = _verify(script, root)

    bad = root / "Bag" / "Acc_x.txt"
    bad.write_text(bad.read_text().replace("1", "9", 1))

    out = subprocess.run(
        [sys.executable, str(script), "verify", "--manifest", str(manifest)],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    assert "hash mismatch" in out.stderr
