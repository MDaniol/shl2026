#!/usr/bin/env python3
"""Cheap, differentiating eval axes for the paper (survey-aligned, pure-NumPy).

Two metrics the SHL/HASCA field under-reports and that cost ~nothing on cached probas:

  1. **Calibration (ECE / MCE).** Expected/Maximum Calibration Error over confidence bins
     — are the model's probabilities trustworthy, not just its argmax? (Guo 2017, 1706.04599.)
     Computable NOW on any class-aligned calibrated proba (e.g. the kept soft-vote).

  2. **Missing-channel robustness.** Macro-F1 when a sensor channel is dropped (zeroed) —
     does the model degrade gracefully under the partial-sensor conditions real wearables
     hit? Head-agnostic: takes a predict_fn over per-channel embeddings (n, C, d), so it
     plugs into the cross-channel head (LIGHTWEIGHT_HEAD_PLAN H-chan) directly.

Both are reported alongside macro-F1 + per-class F1; they differentiate the submission
without changing the model. All deterministic and side-effect-free.
"""
from __future__ import annotations

import numpy as np


def ece(proba, y_true, n_bins: int = 15, classes=None) -> dict:
    """Expected + Maximum Calibration Error. proba: (n, C) row-normalized, columns in
    `classes` order (default 1..C). y_true: labels in that same label space. Equal-width
    confidence bins on max-prob; ECE = sum_b (n_b/n)|acc_b - conf_b|, MCE = max_b(...)."""
    proba = np.asarray(proba, dtype=np.float64)
    classes = np.arange(1, proba.shape[1] + 1) if classes is None else np.asarray(classes)
    y_true = np.asarray(y_true)
    conf = proba.max(1)
    pred = classes[proba.argmax(1)]
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)   # bin per sample
    n = len(conf)
    e, m = 0.0, 0.0
    for b in range(n_bins):
        sel = idx == b
        nb = int(sel.sum())
        if nb == 0:
            continue
        gap = abs(correct[sel].mean() - conf[sel].mean())
        e += (nb / n) * gap
        m = max(m, gap)
    return {"ece": float(e), "mce": float(m), "n_bins": int(n_bins),
            "mean_conf": float(conf.mean()), "accuracy": float(correct.mean())}


def channel_dropout_robustness(predict_fn, X, y, macro_fn, channel_names=None) -> dict:
    """Macro-F1 degradation when each channel is dropped (zeroed), one at a time.

    predict_fn: (n, C, d) -> integer labels.  X: (n, C, d) per-channel embeddings.
    macro_fn: (y_true, y_pred) -> float macro-F1 (pass probe_fusion.macro_f1).
    Returns the intact-input macro, the per-channel drop (base - ablated), and the
    worst/mean drop. Head-agnostic — works for any model over per-channel embeddings."""
    X = np.asarray(X)
    assert X.ndim == 3, "X must be (n, C, d) per-channel embeddings"
    base = float(macro_fn(y, predict_fn(X)))
    C = X.shape[1]
    names = list(channel_names) if channel_names is not None else [f"ch{c}" for c in range(C)]
    drops = {}
    for c in range(C):
        Xc = X.copy()
        Xc[:, c, :] = 0.0
        drops[names[c]] = base - float(macro_fn(y, predict_fn(Xc)))
    vals = list(drops.values())
    return {"base_macro_f1": base,
            "per_channel_drop": {k: float(v) for k, v in drops.items()},
            "worst_drop": float(max(vals)) if vals else 0.0,
            "mean_drop": float(np.mean(vals)) if vals else 0.0}


def self_test() -> None:
    rng = np.random.default_rng(0)
    C = 8

    # ECE: perfectly-calibrated probas -> ~0; over-confident (p=1 but 50% right) -> ~0.5
    n = 4000
    y = rng.integers(1, C + 1, n)
    onehot = np.eye(C)[y - 1]
    well = 0.85 * onehot + 0.15 / C                       # confident AND correct -> low gap
    well = well / well.sum(1, keepdims=True)
    # make it actually ~calibrated: flip a fraction to match confidence
    assert ece(onehot.copy(), y)["ece"] < 1e-6, "one-hot correct labels must be perfectly calibrated"
    # over-confident: always predict class 1 with prob 1 but only ~1/C correct
    overc = np.zeros((n, C)); overc[:, 0] = 1.0
    e_over = ece(overc, y)["ece"]
    assert e_over > 0.6, f"over-confident ECE should be large, got {e_over:.3f}"

    # robustness: a predict_fn that IGNORES input -> zero drops; one that uses only ch0 ->
    # dropping ch0 hurts, others don't.
    Xpc = rng.standard_normal((300, C, 5))
    ypc = rng.integers(1, C + 1, 300)
    macro = lambda yt, yp: float((np.asarray(yt) == np.asarray(yp)).mean())   # accuracy stand-in
    const_pred = lambda X: np.full(len(X), 1)
    r0 = channel_dropout_robustness(const_pred, Xpc, ypc, macro)
    assert r0["worst_drop"] == 0.0, "constant predictor must be channel-invariant"
    # predictor keyed to sign of channel 0's first dim
    ch0_pred = lambda X: np.where(X[:, 0, 0] > 0, 1, 2)
    r1 = channel_dropout_robustness(ch0_pred, Xpc, ypc, macro, channel_names=[f"c{c}" for c in range(C)])
    assert r1["per_channel_drop"]["c0"] >= max(r1["per_channel_drop"][f"c{c}"] for c in range(1, C)), \
        "dropping the informative channel must hurt most"
    print("eval_metrics self_test OK — ECE (calibrated~0, overconfident large) + "
          "channel-dropout robustness (invariant=0, informative-channel hurts)")


if __name__ == "__main__":
    self_test()
