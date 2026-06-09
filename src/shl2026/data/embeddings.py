"""The shared embedding cache — the stable student-facing API.

Because the SHL 2026 foundation models are *frozen*, the expensive step
(running a FM over ~1M windows) is computed once and shared. Everything a
student prototypes runs on these cached embeddings, in seconds, on CPU.

The cache lives on shared group storage (Athena) and is content-addressed by
``(foundation-model id, revision)`` so 12 students reuse a single copy:

    <cache_root>/<fm>@<revision>/<split>/<location>.npz   # train, validation
    <cache_root>/<fm>@<revision>/test/all.npz             # test (no labels)

Each ``.npz`` holds ``X`` (n, dim) float32, optional ``y`` (n,) int labels,
and self-describing metadata. The one entry point students need is
:func:`embeddings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The three challenge splits.
SPLITS = ("train", "validation", "test")
#: Phone locations present in train/validation.
TRAIN_LOCATIONS = ("Bag", "Hips", "Torso", "Hand")
#: Phone locations present at test time (Hand is dropped by the challenge).
TEST_LOCATIONS = ("Bag", "Hips", "Torso")
#: Number of transportation/locomotion classes.
N_CLASSES = 8

#: Environment variable pointing at the shared cache root on group storage.
CACHE_ENV = "SHL_EMB_CACHE"
_DEFAULT_CACHE = "data/processed/embeddings"


@dataclass(frozen=True)
class EmbeddingSet:
    """Frozen-FM embeddings for one (fm, revision, split[, location])."""

    X: np.ndarray
    y: np.ndarray | None
    fm: str
    revision: str
    split: str
    location: str | None
    pooling: str

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def has_labels(self) -> bool:
        return self.y is not None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        loc = self.location or "all"
        lab = "labelled" if self.has_labels else "unlabelled"
        return (
            f"EmbeddingSet({self.fm}@{self.revision} {self.split}/{loc} "
            f"n={self.n} dim={self.dim} pooling={self.pooling} {lab})"
        )


def cache_root(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the cache root: explicit arg > ``$SHL_EMB_CACHE`` > default."""
    if override is not None:
        return Path(override)
    return Path(os.environ.get(CACHE_ENV, _DEFAULT_CACHE))


def _safe(fm: str) -> str:
    """Make a foundation-model id safe to use as a directory name."""
    return fm.replace("/", "__")


def _model_dir(root: Path, fm: str, revision: str) -> Path:
    return root / f"{_safe(fm)}@{revision}"


def _resolve_revision(root: Path, fm: str, revision: str | None) -> str:
    """Pick a revision, auto-detecting when exactly one is cached for ``fm``."""
    if revision is not None:
        return revision
    prefix = f"{_safe(fm)}@"
    matches = sorted(p.name[len(prefix) :] for p in root.glob(f"{prefix}*") if p.is_dir())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"no cached embeddings for fm={fm!r} under {root}. "
            f"Run the embedding-extraction job first (see STUDENTS.md)."
        )
    raise ValueError(
        f"multiple revisions cached for fm={fm!r}: {matches}. "
        f"Pass revision=... to disambiguate."
    )


def embedding_path(
    fm: str,
    revision: str,
    split: str,
    location: str | None,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Filesystem path of one cached embedding shard."""
    base = _model_dir(cache_root(root), fm, revision)
    if split == "test":
        return base / "test" / "all.npz"
    if location is None:
        raise ValueError(f"split={split!r} requires a location ({TRAIN_LOCATIONS})")
    return base / split / f"{location}.npz"


def save_embedding_set(
    X: np.ndarray,
    y: np.ndarray | None,
    *,
    fm: str,
    revision: str,
    split: str,
    location: str | None,
    pooling: str = "mean",
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Write one embedding shard to the cache, creating parent dirs."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = embedding_path(fm, revision, split, location, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "X": np.asarray(X, dtype=np.float32),
        "fm": np.array(fm),
        "revision": np.array(revision),
        "split": np.array(split),
        "location": np.array(location or ""),
        "pooling": np.array(pooling),
    }
    if y is not None:
        payload["y"] = np.asarray(y).astype(np.int16)
    np.savez(path, **payload)
    return path


def _load_one(path: Path) -> EmbeddingSet:
    if not path.exists():
        raise FileNotFoundError(
            f"missing embedding shard: {path}. "
            f"Either the extraction job has not run or the fm/revision is wrong."
        )
    with np.load(path, allow_pickle=False) as npz:
        loc = str(npz["location"]) if "location" in npz.files else None
        return EmbeddingSet(
            X=npz["X"].astype(np.float32),
            y=npz["y"].astype(int) if "y" in npz.files else None,
            fm=str(npz["fm"]),
            revision=str(npz["revision"]),
            split=str(npz["split"]),
            location=loc or None,
            pooling=str(npz["pooling"]) if "pooling" in npz.files else "mean",
        )


def embeddings(
    fm: str,
    split: str,
    location: str | None = None,
    *,
    revision: str | None = None,
    root: str | os.PathLike[str] | None = None,
) -> EmbeddingSet:
    """Load cached frozen-FM embeddings — the one call students need.

    Args:
        fm: foundation-model id (e.g. ``"moment"`` or a HF repo id).
        split: ``"train"``, ``"validation"``, or ``"test"``.
        location: ``Bag``/``Hips``/``Torso``/``Hand`` for train/validation.
            ``None`` concatenates all available locations. Ignored for test.
        revision: cache revision; auto-detected when only one is present.
        root: cache root override (defaults to ``$SHL_EMB_CACHE``).

    Returns:
        An :class:`EmbeddingSet`. ``y`` is ``None`` for the test split, which
        ships without labels.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    rt = cache_root(root)
    rev = _resolve_revision(rt, fm, revision)

    if split == "test":
        return _load_one(embedding_path(fm, rev, "test", None, root))

    if location is not None:
        return _load_one(embedding_path(fm, rev, split, location, root))

    # Concatenate every available location for this split into one set.
    parts = [
        _load_one(p)
        for loc in TRAIN_LOCATIONS
        if (p := embedding_path(fm, rev, split, loc, root)).exists()
    ]
    if not parts:
        raise FileNotFoundError(
            f"no {split} embeddings for fm={fm!r}@{rev} under {rt}."
        )
    X = np.concatenate([p.X for p in parts], axis=0)
    ys = [p.y for p in parts]
    y = np.concatenate(ys, axis=0) if all(v is not None for v in ys) else None
    return EmbeddingSet(
        X=X, y=y, fm=fm, revision=rev, split=split, location=None, pooling=parts[0].pooling
    )


def list_available(root: str | os.PathLike[str] | None = None) -> list[tuple[str, str]]:
    """List ``(fm, revision)`` pairs present in the cache."""
    rt = cache_root(root)
    if not rt.exists():
        return []
    out: list[tuple[str, str]] = []
    for d in sorted(rt.iterdir()):
        if d.is_dir() and "@" in d.name:
            fm, _, rev = d.name.partition("@")
            out.append((fm.replace("__", "/"), rev))
    return out
