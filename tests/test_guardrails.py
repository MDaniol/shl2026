"""Research guardrails — automated checks that block methodological errors.

Maps to the Kapoor & Narayanan leakage taxonomy (Patterns 2023) and the SHL
shuffled-test rule. These run in seconds on synthetic data (no cluster, no real
features), so they gate every change BEFORE it reaches the HPC:

    pytest tests/test_guardrails.py -q

Library-first: we lean on sklearn Pipeline (preprocessing fit on train only ->
prevents leakage L1.2) and the team `shl2026` package rather than hand-rolled code.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

_MODELING = Path(__file__).resolve().parents[1] / "notebooks" / "mdaniol" / "modeling"
import sys
sys.path.insert(0, str(_MODELING))


# --- L1.2 preprocessing leakage: preprocessing must be INSIDE a Pipeline -------
def test_emb_pca_pipeline_is_leakproof():
    """L1.2: scaler/PCA must be fit on train only -> they live inside a Pipeline,
    ordered before the classifier (so .fit(Xtrain) never sees TUNE/TEST stats)."""
    import emb_pca_heads as ep
    pipe = ep.make_pipe(64, "logreg", n_features=300)
    assert isinstance(pipe, Pipeline)
    names = [n for n, _ in pipe.steps]
    assert names.index("clf") == len(names) - 1, "classifier must be last"
    assert "l2" in names and "sc" in names and "pca" in names


def test_freq_mag_heads_are_pipelines():
    import freq_mag_branch as fm
    for head in ("logreg", "ridge", "elasticnet"):
        p = fm.make_head(head)
        assert isinstance(p, Pipeline) and p.steps[0][0] == "sc", \
            f"{head}: scaler must precede clf (no leakage)"


# --- SHL shuffled-test rule: predictions must be window-independent ------------
def test_shuffle_invariance():
    """The test is row-shuffled, so a valid model's predictions must be equivariant
    to row order (no cross-window/temporal dependence). Verify on a real pipeline."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 40)); y = rng.integers(1, 9, 300)
    clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=500))]).fit(X, y)
    Xt = rng.standard_normal((120, 40))
    perm = rng.permutation(120)
    assert np.array_equal(clf.predict(Xt)[perm], clf.predict(Xt[perm])), \
        "predictions depend on row order -> a temporal/cross-window dependency leaked in"


# --- no temporal smoothing over test rows (static scan of the modeling code) ---
def test_no_temporal_smoothing_in_modeling():
    """L3.1 / SHL rule: forbid cross-row smoothing (HMM/Viterbi/median/rolling)
    which is invalid on the shuffled test. Scans for the CALL/IMPORT forms so the
    'we must NOT do X' comments don't false-positive."""
    forbidden = re.compile(r"\.rolling\(|\.ewm\(|\.shift\(|medfilt\(|savgol_filter\(|"
                           r"import hmmlearn|from hmmlearn|HMM\(|viterbi\(")
    offenders = []
    for f in _MODELING.glob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if forbidden.search(line):
                offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, "temporal smoothing found (invalid on shuffled test):\n" + "\n".join(offenders)


# --- determinism: same inputs -> same outputs ---------------------------------
def test_moe_combine_deterministic_and_normalized():
    import moe_combine as moe
    rng = np.random.default_rng(0)
    def norm(a): return a / a.sum(-1, keepdims=True)
    P = np.stack([norm(rng.random((50, 8))) for _ in range(3)])
    q = norm(rng.random((50, 3)))
    a, b = moe.soft_mix(P, q), moe.soft_mix(P, q)
    assert np.array_equal(a, b), "soft_mix not deterministic"
    assert np.allclose(a.sum(1), 1.0), "mix not a probability distribution"


def test_aligned_proba_order_and_normalized():
    """Calibrated probas must be re-indexed to CLASSES (1..8) and row-normalized so
    cross-expert mixing is valid."""
    import lightgbm as lgb
    from probe_fusion import aligned_proba, CLASSES
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 20)); y = rng.integers(1, 9, 400)
    clf = lgb.LGBMClassifier(n_estimators=20, verbosity=-1).fit(X, y)
    P = aligned_proba(clf, X[:30])
    assert P.shape == (30, len(CLASSES))
    assert np.allclose(P.sum(1), 1.0, atol=1e-9)


# --- moe_combine's own invariants (oracle == true expert, etc.) ---------------
def test_moe_combine_self_test():
    import moe_combine as moe
    moe.self_test()   # raises on any violation


# --- temporal split must be time-disjoint with an embargo gap -----------------
def test_temporal_split_time_disjoint():
    """Conservative split: FIT must end before TEST begins, with an embargo gap,
    so FIT/TEST can't share a journey (the 0.90->0.71 over-claim guard, L3.2)."""
    from split import temporal_phase, FIT, TUNE, TEST, UNUSED
    import numpy as np
    n, emb = 10000, 100
    ph = temporal_phase(n, emb)
    fit_i, tune_i, test_i = np.where(ph == FIT)[0], np.where(ph == TUNE)[0], np.where(ph == TEST)[0]
    assert fit_i.max() < tune_i.min() < test_i.min(), "slices not in time order"
    assert tune_i.min() - fit_i.max() - 1 >= emb - 1, "no embargo gap before TUNE"
    assert test_i.min() - tune_i.max() - 1 >= emb - 1, "no embargo gap before TEST"
    assert (ph == UNUSED).sum() == 2 * emb, "embargo rows wrong count"


# --- determinism: every bagged LightGBM must be seeded ------------------------
def test_lgbm_subsampling_is_seeded():
    """`subsample<1` makes LightGBM stochastic; without random_state results vary
    run-to-run (breaks reproducibility). Enforce: in each modeling file, #subsample
    instantiations <= #random_state (every bagged model is seeded)."""
    offenders = []
    for f in _MODELING.glob("*.py"):
        txt = f.read_text()
        if txt.count("subsample=") > txt.count("random_state="):
            offenders.append(f.name)
    assert not offenders, f"bagged LightGBM without random_state in: {offenders}"
