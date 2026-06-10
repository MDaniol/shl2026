"""Dirty-tree code snapshots must make every tracked run recreatable.

The contract: from a run's git_sha tag + its code_snapshot/ artifacts, the
exact working tree that produced the run can be rebuilt (checkout + apply
patch + untar). Clean trees need no snapshot; broken git must never break a
run.
"""

from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from shl2026.tracking import snapshot
from shl2026.tracking.autolog import track


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny committed git repo, with cwd inside it."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.chdir(root)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (root / "model.py").write_text("alpha = 1\n")
    git("add", "-A")
    git("commit", "-qm", "init")
    return root


def test_clean_tree_needs_no_snapshot(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "snap"
    out.mkdir()
    assert snapshot.create_code_snapshot(out) is None
    assert not list(out.iterdir())


def test_snapshot_captures_diff_and_untracked(repo: Path, tmp_path: Path) -> None:
    (repo / "model.py").write_text("alpha = 2\n")  # tracked edit
    (repo / "exp.py").write_text("idea = 'fusion'\n")  # untracked new file
    out = tmp_path / "snap"
    out.mkdir()

    report = snapshot.create_code_snapshot(out)
    assert report is not None
    assert report.has_patch
    assert report.untracked_included == ["exp.py"]
    assert not report.untracked_skipped

    assert "alpha = 2" in (out / "diff.patch").read_text()
    with tarfile.open(out / "untracked.tar.gz") as tar:
        assert tar.getnames() == ["exp.py"]

    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["git_sha"] == report.git_sha
    assert manifest["restore"][0] == f"git checkout {report.git_sha}"


def test_snapshot_restores_exact_tree(repo: Path, tmp_path: Path) -> None:
    (repo / "model.py").write_text("alpha = 3\n")
    (repo / "exp.py").write_text("idea = 'fusion'\n")
    out = tmp_path / "snap"
    out.mkdir()
    report = snapshot.create_code_snapshot(out)
    assert report is not None

    # Recreate in a fresh clone, following MANIFEST.json's restore steps.
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(repo), str(clone)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "checkout", "-q", report.git_sha], cwd=clone, check=True
    )
    subprocess.run(
        ["git", "apply", str(out / "diff.patch")], cwd=clone, check=True
    )
    with tarfile.open(out / "untracked.tar.gz") as tar:
        tar.extractall(clone, filter="data")

    assert (clone / "model.py").read_text() == "alpha = 3\n"
    assert (clone / "exp.py").read_text() == "idea = 'fusion'\n"


def test_oversized_untracked_files_are_skipped_but_listed(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot, "MAX_FILE_BYTES", 10)
    (repo / "small.py").write_text("x = 1\n")
    (repo / "huge.bin").write_bytes(b"\0" * 100)
    out = tmp_path / "snap"
    out.mkdir()

    report = snapshot.create_code_snapshot(out)
    assert report is not None
    assert report.untracked_included == ["small.py"]
    assert report.untracked_skipped == ["huge.bin"]
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["untracked_skipped_size_cap"] == ["huge.bin"]


def test_local_caches_are_excluded(repo: Path, tmp_path: Path) -> None:
    (repo / "mlruns" / "0").mkdir(parents=True)
    (repo / "mlruns" / "0" / "meta.yaml").write_text("x")
    (repo / "exp.py").write_text("y = 1\n")
    out = tmp_path / "snap"
    out.mkdir()

    report = snapshot.create_code_snapshot(out)
    assert report is not None
    assert report.untracked_included == ["exp.py"]


def test_outside_git_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nogit = tmp_path / "plain"
    nogit.mkdir()
    monkeypatch.chdir(nogit)
    out = tmp_path / "snap"
    out.mkdir()
    assert snapshot.create_code_snapshot(out) is None


def test_notebook_history_noop_outside_ipython(tmp_path: Path) -> None:
    assert snapshot.dump_notebook_history(tmp_path) is False
    assert not list(tmp_path.iterdir())


def test_track_attaches_snapshot_artifacts(repo: Path, tmp_path: Path) -> None:
    (repo / "exp.py").write_text("tweak = True\n")  # dirty: untracked file
    uri = f"file://{tmp_path / 'mlruns'}"

    with track("pytest", run_name="dirty-run", tracking_uri=uri) as run:
        run.log_metrics({"macro_f1": 0.5})

    found = list((tmp_path / "mlruns").rglob("code_snapshot/MANIFEST.json"))
    assert len(found) == 1
    archived = list((tmp_path / "mlruns").rglob("code_snapshot/untracked.tar.gz"))
    assert len(archived) == 1
