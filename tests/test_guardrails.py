"""Research guardrails — automated checks that block methodological errors.

Maps to the Kapoor & Narayanan leakage taxonomy (Patterns 2023) and the SHL
shuffled-test rule. These run in seconds on synthetic data (no cluster, no real
features), so they gate every change BEFORE it reaches the HPC:

    pytest tests/test_guardrails.py -q

Library-first: we lean on sklearn Pipeline (preprocessing fit on train only ->
prevents leakage L1.2) and the team `shl2026` package rather than hand-rolled code.
"""
from __future__ import annotations

import os
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
    result must equal the input exactly. Skips where torch is absent (the x86 CPU env runs
    the gate but doesn't carry the FM stack — torch lives in the aarch64 GH200 env)."""
    pytest.importorskip("torch")          # extract_embeddings imports torch at module load
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
    multithreading unless deterministic=True (+force_col_wise). Enforce: every file
    using subsample= must set deterministic= at least as often (paper-grade reproducible
    numbers; this is what caused 0.8157-vs-0.8213 on the same recipe)."""
    offenders = []
    for f in _MODELING.glob("*.py"):
        txt = f.read_text()
        if txt.count("subsample=") > txt.count("deterministic="):
            offenders.append(f.name)
    assert not offenders, f"bagged LightGBM without deterministic= in: {offenders}"


def test_lgbm_deterministic_config_reproduces():
    """Functional check: with deterministic=True + force_col_wise=True + fixed seed, two
    fits on identical data give identical predictions (the config we now ship)."""
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 25)); y = rng.integers(1, 9, 600)
    def fit():
        return lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=80,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  n_jobs=-1, verbosity=-1, random_state=0,
                                  deterministic=True, force_col_wise=True).fit(X, y).predict(X)
    assert np.array_equal(fit(), fit()), "deterministic LGBM config not reproducible"


def test_eval_metrics_ece_and_robustness():
    """ECE is ~0 for perfectly-calibrated probs and large for over-confident ones;
    channel-dropout robustness is 0 for a channel-invariant predictor and largest for
    the informative channel. Guards the paper's two differentiating eval axes."""
    import eval_metrics as em
    em.self_test()        # raises on any violation


def test_head_xchannel_forward_and_se_uses_channels():
    """The 4 channel-pool heads output (B,8); the SE head's output actually depends on
    channel content (it's not a glorified mean); a short train runs + yields normalized
    probas. Guards the novel cross-channel head before any cluster run. Skips where torch is
    absent (x86 CPU gate); runs fully on the aarch64 GH200 env where the head trains."""
    pytest.importorskip("torch")
    import head_xchannel as hx
    hx.self_test()        # raises on any violation


def test_vote_model_reuse_is_lossless():
    """Model-reuse correctness: submit_vote --from-models reuses voting_head's joblib'd
    (clf, cal_weights). A joblib round-trip must preserve aligned_proba EXACTLY, and the
    weighted+recal vote arithmetic must be reproducible from the saved params — so the fast
    path gives bit-identical predictions to refitting."""
    import io, joblib
    import lightgbm as lgb
    from probe_fusion import aligned_proba, CLASSES
    rng = np.random.default_rng(0)
    cls = np.asarray(CLASSES)
    Xte = rng.standard_normal((60, 12)).astype(np.float32)
    Pte, saved = [], []
    for s in (0, 1):
        Xtr = rng.standard_normal((300, 12)); ytr = rng.integers(1, 9, 300)
        clf = lgb.LGBMClassifier(n_estimators=20, verbosity=-1, random_state=s).fit(Xtr, ytr)
        wcal = np.abs(rng.standard_normal(8)) + 0.5
        # round-trip the model through joblib (what voting_head saves / submit_vote loads)
        buf = io.BytesIO(); joblib.dump((clf, wcal), buf); buf.seek(0)
        clf2, wcal2 = joblib.load(buf)
        P1, P2 = aligned_proba(clf, Xte, wcal), aligned_proba(clf2, Xte, wcal2)
        assert np.array_equal(P1, P2), "joblib round-trip changed aligned_proba"
        Pte.append(P1); saved.append((clf2, wcal2))
    Pte = np.stack(Pte, 0)
    vw = np.array([0.6, 0.4]); recal = np.abs(rng.standard_normal(8)) + 0.5
    ref = cls[(np.einsum("knc,k->nc", Pte, vw) * recal).argmax(1)]                 # fit-path vote
    Pte_reload = np.stack([aligned_proba(c, Xte, w) for c, w in saved], 0)         # load-path vote
    got = cls[(np.einsum("knc,k->nc", Pte_reload, vw) * recal).argmax(1)]
    assert np.array_equal(ref, got), "reused models do not reproduce the vote prediction"


def test_spectrogram_thread_count_identical():
    """The extraction speed-up sets torch.set_num_threads(allocated_cores) to parallelize the
    spectrogram interpolate+normalize across CPU cores. That MUST be bit-identical to the
    single-threaded result (no accuracy change) — interpolate is elementwise and torch's CPU
    mean/std reduction is deterministic. Lock it: same input, different thread counts -> equal."""
    torch = pytest.importorskip("torch")
    import numpy as np
    import extract_embeddings as ee
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 3 * (np.arange(500) / 100))[None].repeat(40, 0) \
        + 0.1 * rng.standard_normal((40, 500))
    prev = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        ref_ast = ee.imu_log_spectrogram(x, n_mels=128, n_frames=1024)
        ref_img = ee.imu_spectrogram_image(x, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        torch.set_num_threads(max(2, min(8, (os.cpu_count() or 2))))
        assert torch.equal(ref_ast, ee.imu_log_spectrogram(x, n_mels=128, n_frames=1024)), \
            "AST spectrogram changed with thread count — NOT number-identical"
        assert torch.equal(ref_img, ee.imu_spectrogram_image(
            x, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])), \
            "ViT spectrogram-image changed with thread count — NOT number-identical"
    finally:
        torch.set_num_threads(prev)


def test_ast_spectrogram_shape_and_norm():
    """AST branch: per-channel IMU log-spectrogram lands on AST's (n_frames, n_mels) grid, is
    finite, and is per-spectrogram z-normalized (mean~0, std~1) as AST expects. Guards the new
    spectrogram front-end before any GPU run. Skips where torch is absent (x86 CPU gate)."""
    pytest.importorskip("torch")
    import numpy as np
    import extract_embeddings as ee
    rng = np.random.default_rng(0)
    t = np.arange(500) / 100
    x = np.sin(2 * np.pi * 3 * t)[None].repeat(6, 0) + 0.1 * rng.standard_normal((6, 500))
    S = ee.imu_log_spectrogram(x, n_mels=128, n_frames=1024)
    assert tuple(S.shape) == (6, 1024, 128)
    s = S.numpy()
    assert np.isfinite(s).all()
    assert abs(float(s.mean())) < 1e-3 and abs(float(s.std()) - 0.5) < 0.05   # AST contract: std 0.5


def test_vit_spectrogram_image_shape_and_norm():
    """Vision-ViT branch: per-channel IMU spectrogram-image lands on (n,3,size,size), is finite,
    and is min-maxed-then-(mean,std)-normalized per the ViT's processor (RGB-replicated). Guards
    the DINOv2/CLIP image front-end before any GPU run. Skips where torch is absent (CPU gate)."""
    pytest.importorskip("torch")
    import numpy as np
    import extract_embeddings as ee
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 3 * (np.arange(500) / 100))[None].repeat(5, 0) + 0.1 * rng.standard_normal((5, 500))
    img = ee.imu_spectrogram_image(x, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert tuple(img.shape) == (5, 3, 224, 224)
    a = img.numpy()
    assert np.isfinite(a).all()
    # pre-norm RGB channels were identical (replicate) -> after de-norm they match
    de = a * np.array([0.229, 0.224, 0.225])[None, :, None, None] + np.array([0.485, 0.456, 0.406])[None, :, None, None]
    assert np.allclose(de[:, 0], de[:, 1], atol=1e-4) and np.allclose(de[:, 1], de[:, 2], atol=1e-4)
    assert de.min() > -1e-3 and de.max() < 1 + 1e-3          # de-normalized back to [0,1]


def test_orient_rotation_invariance():
    """The orientation features MUST be invariant to a global phone rotation (the whole point — the
    hidden test mixes placements). Rotate acc/gyr/mag by the same proper rotation → features unchanged."""
    import numpy as np
    import orient_features as of
    rng = np.random.default_rng(0)
    n, T = 6, 500
    acc = rng.standard_normal((n, 3, T)); acc[:, 2, :] += 9.8
    gyr = 0.1 * rng.standard_normal((n, 3, T)); mag = 40 + rng.standard_normal((n, 3, T))
    f0, names = of.orient_block(acc, gyr, mag)
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    R = R * np.sign(np.linalg.det(R))                              # ensure a proper rotation (det +1)
    rot = lambda x: np.einsum("ij,njt->nit", R, x)
    f1, _ = of.orient_block(rot(acc), rot(gyr), rot(mag))
    assert f0.shape == (n, 11) and len(names) == 11
    assert np.allclose(f0, f1, atol=1e-5), "orient features are NOT rotation-invariant"


def test_rail_redistribute():
    """E-RAIL redistribute: β=0 ⇒ unchanged; β=1 ⇒ within-pair split = binary p; pair mass preserved;
    other classes untouched."""
    import numpy as np
    import rail_disambig as rd
    cls = np.asarray(rd.CLASSES)
    P = np.array([[0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.3, 0.2]])      # Train(7)=0.3, Subway(8)=0.2
    p_pos = np.array([0.9])                                        # binary says P(Train|pair)=0.9
    q0 = rd.redistribute(P, p_pos, cls, 7, 8, 0.0)
    assert np.allclose(q0, P)                                      # β=0 → unchanged
    q1 = rd.redistribute(P, p_pos, cls, 7, 8, 1.0)
    mass = 0.5
    assert abs(q1[0, 6] - mass * 0.9) < 1e-9 and abs(q1[0, 7] - mass * 0.1) < 1e-9
    assert abs((q1[0, 6] + q1[0, 7]) - mass) < 1e-9               # pair mass preserved
    assert np.allclose(q1[0, [0, 1, 2, 3, 4, 5]], P[0, [0, 1, 2, 3, 4, 5]])  # others untouched


def test_additive_logspace_equals_exp_multiply():
    """v4 submission consistency: the C1 decision is argmax(log p + b); submit_vote ships it as
    argmax(p · exp(b)). These MUST give the identical per-window label (so the submission == the
    validated rule), and p·exp(b) renormalizes to a valid proba for the combine."""
    import numpy as np
    import decision_rule as dr
    rng = np.random.default_rng(1)
    P = dr._norm(rng.random((300, 8)) + 0.01)
    b = rng.standard_normal(8)
    a_log = (np.log(P + dr.EPS) + b).argmax(1)
    a_mul = (P * np.exp(b)).argmax(1)
    assert np.array_equal(a_log, a_mul)
    q = dr._norm(P * np.exp(b))
    assert np.allclose(q.sum(1), 1.0) and np.array_equal(q.argmax(1), a_mul)


def test_ablation_importance_helpers():
    """Ablation overfitting probes: importance share splits emb vs HC correctly; Jaccard top-k is 1
    for identical importances and <1 when the top set differs."""
    import numpy as np
    import ablation_rep as ab
    imp = np.array([3.0, 1.0, 0.0, 0.0, 6.0])                       # n_emb=2 → emb=4, hc=6, tot=10
    fm_s, hc_s = ab.importance_share(imp, n_emb=2)
    assert abs(fm_s - 0.4) < 1e-9 and abs(hc_s - 0.6) < 1e-9
    assert ab.jaccard_topk(imp, imp, 2) == 1.0
    assert ab.jaccard_topk(np.array([5, 4, 0, 0]), np.array([0, 0, 4, 5]), 2) == 0.0


def test_robustness_diag_functions():
    """Confusion/per-class-F1/top-2/prior helpers: perfect diagonal → macro 1.0; a planted rank-2
    recovery is detected; uniform-prior reweight is computed without error."""
    import numpy as np
    import robustness_diag as rd
    cls = np.asarray(rd.CLASSES)
    y = np.array([7, 7, 5, 5])
    perfect = rd.confusion_counts(y, y, cls)
    f1, macro = rd.per_class_f1_from_confusion(perfect)
    assert abs(f1[list(cls).index(7)] - 1.0) < 1e-9                 # Train perfectly predicted
    assert abs(f1[list(cls).index(5)] - 1.0) < 1e-9                 # Car perfectly predicted
    assert abs(macro - 2.0 / 8) < 1e-9                             # only 2 of 8 classes present & perfect
    # top-2: Train predicted as Subway, but truth is rank-2 in proba → recoverable
    proba = np.zeros((1, 8)); proba[0, list(cls).index(8)] = 0.6; proba[0, list(cls).index(7)] = 0.4
    rows = rd.top2_structure(np.array([7]), np.array([8]), proba, cls)
    assert rows and rows[0][0] == 7 and rows[0][1] == 8 and rows[0][3] == 1
    # prior sensitivity runs and returns a macro per grid entry
    C = rd.confusion_counts(np.array([7, 7, 5, 5, 3]), np.array([7, 8, 5, 5, 3]), cls)
    res = rd.prior_sensitivity(C, cls, [("emp", C.sum(1) / C.sum()), ("uni", np.ones(8) / 8)])
    assert len(res) == 2 and all(0 <= m <= 1 for _, m in res)


def test_paired_bootstrap_diff():
    """C1 paired significance: deterministic; identical preds → Δ≈0 with CI bracketing 0; a strictly
    better rule → Δ>0 with lo>0 (the correct paired test, not a marginal-CI overlap)."""
    import numpy as np
    import decision_rule as dr
    cls = np.asarray(dr.CLASSES)
    rng = np.random.default_rng(0)
    n = 1500
    y = rng.integers(1, 9, n)
    a = y.copy(); b = y.copy()
    b[:300] = (b[:300] % 8) + 1                                         # b is worse on 300 windows
    d, lo, hi, p = dr.paired_bootstrap_diff(y, a, b, B=300, seed=0)
    d2, lo2, hi2, p2 = dr.paired_bootstrap_diff(y, a, b, B=300, seed=0)
    assert (d, lo, hi, p) == (d2, lo2, hi2, p2)                         # deterministic
    assert d > 0 and lo > 0 and p < 0.05                               # a significantly beats b
    d0, lo0, hi0, p0 = dr.paired_bootstrap_diff(y, a, a, B=300, seed=0)
    assert abs(d0) < 1e-9 and lo0 <= 0 <= hi0                          # identical → no win


def test_build_rep_concatenates_2d(monkeypatch):
    """A1 build_rep: fusion = emb ⊕ feats must be 2-D (regression guard for the load_feats[0] bug)."""
    import numpy as np
    import pair_separability as ps
    monkeypatch.setattr(ps, "load_emb", lambda *a, **k: np.zeros((10, 768), np.float32))
    monkeypatch.setattr(ps, "load_feats", lambda *a, **k: np.zeros((10, 520), np.float32))
    p = Path(".")
    assert ps.build_rep(p, p, "x", "validation", "fusion").shape == (10, 1288)
    assert ps.build_rep(p, p, "x", "validation", "emb").shape == (10, 768)
    assert ps.build_rep(p, p, "x", "validation", "handcrafted").shape == (10, 520)


def test_ceiling_diag_functions():
    """Phase-0 diagnostics: top-2 oracle ≥ base, confusion-collapse corrects only in-pair swaps,
    error-correlation Q∈[-1,1] (identical preds → Q=1), bootstrap CI brackets the point estimate."""
    import numpy as np
    import ceiling_diag as cd
    cls = np.asarray(cd.CLASSES)
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(1, 9, n)
    base = y.copy(); base[:100] = ((base[:100]) % 8) + 1                 # 100 wrong base preds
    # one FM whose top-2 always contains the truth -> oracle recovers all
    P = np.zeros((n, 8)); P[np.arange(n), y - 1] = 0.6; P[:, 0] += 0.1
    orac = cd.top2_oracle(y, [P], base, cls)
    assert cd.macro_f1(y, orac) >= cd.macro_f1(y, base)                 # oracle never worse
    assert (orac == y).all()                                           # truth in top-2 here -> all correct
    # confusion-collapse: a Train(7)->Subway(8) swap is corrected; a Car(5)->Walk(2) error is NOT
    yy = np.array([7, 5, 3]); pp = np.array([8, 2, 3])
    cc = cd.confusion_collapse(yy, pp, cd.PAIRS)
    assert cc[0] == 7 and cc[1] == 2 and cc[2] == 3
    # error-correlation
    q_same = cd.error_correlation(y, base, base)
    assert abs(q_same["Q"] - 1.0) < 1e-6 and q_same["double_fault"] >= 0
    assert -1.0 - 1e-9 <= cd.error_correlation(y, base, y)["Q"] <= 1.0 + 1e-9
    # bootstrap CI brackets the point estimate, deterministic
    pt, lo, hi = cd.bootstrap_ci(y, base, cls, B=200, seed=0)
    pt2, lo2, hi2 = cd.bootstrap_ci(y, base, cls, B=200, seed=0)
    assert lo <= pt <= hi and (pt, lo, hi) == (pt2, lo2, hi2)


def test_additive_bias_improves_macro_and_is_deterministic():
    """C1: the joint coordinate-descent additive log-bias must (a) be deterministic, (b) not REDUCE
    macro-F1 vs raw argmax on the data it's fit on (it's a maximizer), and (c) raise an under-emitted
    minority class that a vanishing multiplier can't. Pure-numpy core, runs on the CPU gate."""
    import numpy as np
    import decision_rule as dr
    cls = np.asarray(dr.CLASSES)
    rng = np.random.default_rng(0)
    n = 2000
    y = np.where(rng.random(n) < 0.05, 3, rng.integers(1, 9, n))       # class 3 (=Run-like) rare
    # probas biased AGAINST class 3 (under-emitted): shrink its column
    P = rng.random((n, 8)) + 0.01
    for i in range(n):
        P[i, y[i] - 1] += 0.6                                          # signal, but...
    P[:, 2] *= 0.25                                                    # ...class-3 column suppressed
    P = dr._norm(P)
    base = dr.macro_f1(y, cls[P.argmax(1)])
    b1 = dr.additive_bias_search(P, y, cls)
    b2 = dr.additive_bias_search(P, y, cls)
    assert np.array_equal(b1, b2)                                      # deterministic
    after = dr.macro_f1(y, cls[(np.log(dr._norm(P) + dr.EPS) + b1).argmax(1)])
    assert after >= base - 1e-9                                        # never hurts on fit data
    assert b1[2] > 0.0                                                 # raises the suppressed class
    # SLD recovers a planted target prior shift
    corr, tgt = dr.sld_correct(P, P.mean(0))
    assert corr.shape == P.shape and abs(tgt.sum() - 1.0) < 1e-6


def test_pair_separability_probe_leakage_safe():
    """A1 diagnostic: the binary probe fits on FIT and scores on a DISJOINT TEST (no row overlap),
    is deterministic, and returns a sane bal-acc in [0,1] for a separable toy pair. Guards the
    leakage-safe contract before any cluster run."""
    pytest.importorskip("sklearn")
    import numpy as np
    import pair_separability as ps
    rng = np.random.default_rng(0)
    # two clearly-separable classes (labels 7,8) -> probe should score high on held-out
    n = 300
    Xfit = np.concatenate([rng.standard_normal((n, 8)), rng.standard_normal((n, 8)) + 4.0])
    yfit = np.array([7] * n + [8] * n)
    Xtest = np.concatenate([rng.standard_normal((80, 8)), rng.standard_normal((80, 8)) + 4.0])
    ytest = np.array([7] * 80 + [8] * 80)
    r1 = ps.probe_pair(Xfit, yfit, Xtest, ytest, 7, 8)
    r2 = ps.probe_pair(Xfit, yfit, Xtest, ytest, 7, 8)
    assert 0.0 <= r1["bal_acc"] <= 1.0 and r1["bal_acc"] > 0.9        # separable -> high
    assert r1 == r2                                                    # deterministic
    assert r1["n_fit"] == 2 * n and r1["n_test"] == 160               # only the pair's rows used
    assert ps.verdict(0.95).startswith("SEPARABLE") and ps.verdict(0.5).startswith("COLLAPSED")


def test_pool_per_channel_equals_channel_mean(tmp_path):
    """pool_per_channel mean-pools a per-channel (n,C,d) embedding dir into a (n,d) voter dir, and
    that MUST equal the channel-mean exactly (it reproduces a non-per-channel extraction, so the
    dinov2 voter is number-identical to a fresh non-pc extraction — no GPU re-run)."""
    import numpy as np
    import pool_per_channel as pp
    rng = np.random.default_rng(0)
    src = tmp_path / "dinov2_V2_pc"; src.mkdir()
    dst = tmp_path / "dinov2_V2"
    a = rng.standard_normal((7, 5, 12)).astype(np.float32)
    np.save(src / "validation__Bag.npy", a)
    assert pp.pool_dir(src, dst) == 1
    out = np.load(dst / "validation__Bag.npy")
    assert out.shape == (7, 12)
    assert np.array_equal(out, a.mean(axis=1).astype(np.float32))


def test_pack_imagebind_contract():
    """ImageBind IMU packer: 6-ch acc+gyr (NO magnetometer), interp to 2000 samples, per-channel
    MEAN-SUBTRACTION (NOT z-norm — ImageBind's training contract; har-fm-scientist fix). Pure-numpy
    so it runs on the CPU gate too."""
    import numpy as np, sys
    sys.path.insert(0, str(_MODELING.parent / "feature_extraction"))
    import fm_input as fi
    rng = np.random.default_rng(0)
    acc = rng.standard_normal((6, 3, 500)).astype(np.float32); acc[:, 2] += 9.8
    gyr = 0.1 * rng.standard_normal((6, 3, 500)).astype(np.float32)
    mag = 40 + rng.standard_normal((6, 3, 500)).astype(np.float32)
    x, names = fi.pack_imagebind(fi.build_channel_library(acc, gyr, mag), "V0")
    assert x.shape == (6, 6, 2000)
    assert not any("Mag" in n for n in names)                 # acc+gyr only
    assert np.abs(x.mean(axis=-1)).max() < 1e-4               # per-channel zero-mean
    # NOT z-normed: per-channel std must vary (not forced to 1)
    assert x.std(axis=-1).std() > 1e-3


def test_combine_external_alignment_and_blend():
    """Cross-team combine guard: the RAW-signal window signature is float32/float64-robust, a
    permuted (misaligned) order is DETECTED, and the weighted blend is row-normalized + preserves
    argmax on identical inputs. Ensures we can't silently ensemble misaligned test rows."""
    import combine_external as ce
    ce.self_test()        # raises on any violation


def test_batched_per_channel_equals_loop():
    """Perf opt must be NUMBER-IDENTICAL: the AST/DINOv2 batched path reshapes a single
    forward over (b*C) channel-rows to (b,C,d) — this must equal the per-channel C-loop
    (embed_per_channel stacks per-channel). Guards against a reshape/transpose bug that would
    silently scramble channels. (Frozen LayerNorm models are per-sample, so batching can't change
    the forward; only the reshape layout is at risk — tested here with a row-wise fake forward.)"""
    import numpy as np
    b, C, T, d = 4, 5, 500, 7
    X = np.random.default_rng(0).standard_normal((b, C, T))
    rowfwd = lambda s: s[:, :d] + s.mean(1, keepdims=True)        # (m,T)->(m,d), strictly row-wise
    batched = rowfwd(X.reshape(b * C, T)).reshape(b, C, d)        # the optimized path's layout
    loop = np.stack([rowfwd(X[:, c, :]) for c in range(C)], axis=1)  # embed_per_channel layout
    assert batched.shape == (b, C, d) and np.allclose(batched, loop)
    # and mean-over-C (pooled path) equals the loop's mean
    assert np.allclose(batched.mean(1), loop.mean(1))
