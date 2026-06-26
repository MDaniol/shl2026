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
def test_temporal_split_time_disjoint_and_class_complete():
    """Conservative split must be BOTH (a) time-disjoint per class — that class's
    FIT occurrences end before its TEST occurrences, with an embargo gap (the
    0.90->0.71 over-claim guard, L3.2) — AND (b) class-complete: every class present
    in FIT/TUNE/TEST (a global cut drops session-clustered classes -> F1=0)."""
    from split import temporal_phase, FIT, TUNE, TEST, UNUSED
    import numpy as np
    # 8 classes, each a contiguous time-run of 500 rows (mimics session bouts)
    y = np.repeat(np.arange(1, 9), 500)
    ph = temporal_phase(y, embargo=10)
    for c in range(1, 9):
        idx = np.where(y == c)[0]
        f, t = idx[ph[idx] == FIT], idx[ph[idx] == TEST]
        assert len(f) and len(t), f"class {c} missing from FIT or TEST"
        assert f.max() < t.min(), f"class {c}: FIT not before TEST"
    for code in (FIT, TUNE, TEST):                       # class-complete
        assert set(np.unique(y[ph == code])) == set(range(1, 9)), "a class is missing from a slice"
    assert (ph == UNUSED).sum() > 0, "no embargo gap"


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


def test_prior_adapt_recovers_shift_and_identities():
    """Tier-1 label-shift: MLLS recovers a known induced prior shift; adapt/logit are
    identities under no shift; outputs stay row-normalized."""
    import prior_adapt as pa
    pa.self_test()                       # raises on failure (MLLS recovery + identities)
    import numpy as np
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    pi = np.array([0.5, 0.3, 0.2])
    assert np.allclose(pa.adapt(P, pi, pi), P, atol=1e-9)
    assert np.allclose(pa.logit_adjust(P, pi, 0.0), P, atol=1e-9)
    assert np.allclose(pa.adapt(P, pi, pi[::-1]).sum(1), 1.0, atol=1e-6)


def test_loaders_route_test_to_single_all_file():
    """Regression: the test set is one merged 'all' file (no per-location, no Hand), so
    the loaders must NOT iterate Bag/Hips/Torso/Hand for split=='test' (that was a
    FileNotFoundError on dataset_parquet_features/test/Bag.parquet). train/validation
    still load per body location."""
    import probe_fusion as pf
    assert pf.split_locs("test") == ("all",)
    assert pf.split_locs("train") == pf.LOCATIONS
    assert pf.split_locs("validation") == pf.LOCATIONS


def test_voting_head_combiner_math():
    """Soft-voting head: equal vote == moe_combine.uniform_mix; the einsum weighted
    vote with uniform weights == equal vote; weight_search concentrates on the FM that
    perfectly predicts TUNE (selection-on-TUNE works); a vote of identical probas is a
    no-op. Guards the hand-written vote/weight math against a library reference."""
    import numpy as np
    import voting_head as vh
    import moe_combine as mc
    rng = np.random.default_rng(0)
    K, n = 2, 300
    def norm(a): return a / a.sum(-1, keepdims=True)
    P = np.stack([norm(rng.random((n, 8))) for _ in range(K)], 0)   # (K,n,8)

    # equal vote == library uniform_mix (moe_combine stacks experts the same way)
    assert np.allclose(P.mean(0), mc.uniform_mix(P), atol=1e-12)
    # einsum weighted vote with uniform weights == equal vote
    w_unif = np.ones(K) / K
    assert np.allclose(np.einsum("knc,k->nc", P, w_unif), P.mean(0), atol=1e-12)
    # vote of identical probas is a no-op (whatever the weights)
    Pid = np.stack([P[0], P[0]], 0)
    assert np.allclose(np.einsum("knc,k->nc", Pid, np.array([0.3, 0.7])), P[0], atol=1e-12)

    # weight_search selects on TUNE: FM0 predicts y, FM1 is ADVERSARIAL (confident on a
    # wrong class), so equal weights degrade and the search must concentrate on FM0.
    y = np.asarray(vh.CLASSES)[rng.integers(0, 8, n)]
    wrong_idx = y % 8                                               # != y-1 for every y in 1..8
    Pgood = norm(0.99 * np.eye(8)[y - 1] + 0.01 / 8)
    Pbad = norm(0.99 * np.eye(8)[wrong_idx] + 0.01 / 8)
    Pmix = np.stack([Pgood, Pbad], 0)
    score = lambda ww: vh.macro_f1(
        y, np.asarray(vh.CLASSES)[np.einsum("knc,k->nc", Pmix, ww / ww.sum()).argmax(1)])
    w = vh.weight_search(Pmix, y, K)
    assert np.isclose(w.sum(), 1.0, atol=1e-9)
    assert score(w) >= score(np.ones(K)) - 1e-9                     # never worse than equal weights
    assert w[0] > w[1]                                              # concentrates on the good FM


def test_embed_per_channel_keeps_channels_separate():
    """Per-channel extraction stacks each channel's embedding -> (n, C, d), and the mean
    over channels reproduces a channel-pooled embedding (the property that makes per-channel
    a strict generalization for channel-independent FMs). Uses an identity embed_fn so the
    result must equal the input exactly."""
    import numpy as np
    import extract_embeddings as ee
    n, C, T = 5, 9, 32
    X = np.random.default_rng(0).standard_normal((n, C, T)).astype(np.float32)
    ident = lambda x: x[:, 0, :]                       # (n,1,T) -> (n,T): identity per channel
    out = ee.embed_per_channel(X, ident)
    assert out.shape == (n, C, T)
    assert np.allclose(out, X, atol=1e-6)             # channel c -> X[:, c]
    assert np.allclose(out.mean(1), X.mean(1), atol=1e-6)  # mean over C == pooled


def test_lgbm_subsampling_is_deterministic():
    """Reproducibility: bagged LightGBM (subsample<1) is non-deterministic under
    multithreading unless deterministic=True (+force_row_wise). Enforce: every file
    using subsample= must set deterministic= at least as often (paper-grade reproducible
    numbers; this is what caused 0.8157-vs-0.8213 on the same recipe)."""
    offenders = []
    for f in _MODELING.glob("*.py"):
        txt = f.read_text()
        if txt.count("subsample=") > txt.count("deterministic="):
            offenders.append(f.name)
    assert not offenders, f"bagged LightGBM without deterministic= in: {offenders}"


def test_lgbm_deterministic_config_reproduces():
    """Functional check: with deterministic=True + force_row_wise=True + fixed seed, two
    fits on identical data give identical predictions (the config we now ship)."""
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 25)); y = rng.integers(1, 9, 600)
    def fit():
        return lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=80,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  n_jobs=-1, verbosity=-1, random_state=0,
                                  deterministic=True, force_row_wise=True).fit(X, y).predict(X)
    assert np.array_equal(fit(), fit()), "deterministic LGBM config not reproducible"


def test_eval_metrics_ece_and_robustness():
    """ECE is ~0 for perfectly-calibrated probs and large for over-confident ones;
    channel-dropout robustness is 0 for a channel-invariant predictor and largest for
    the informative channel. Guards the paper's two differentiating eval axes."""
    import eval_metrics as em
    em.self_test()        # raises on any violation
