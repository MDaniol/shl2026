# Lightweight-head development (pre-registration)

**Status:** pre-registered 2026-06-25, before building or peeking at lock-test. This is the
challenge's *sanctioned* innovation surface (frozen FM + **lightweight trainable heads**), so it's
where ranking-edge and paper-novelty align. Authority: `AI_GUIDELINES.md` (8-stage, leakage gate,
MLflow §8), `WINNING_STRATEGY.md`. Metric = macro-F1, temporal+embargo split, Bag/Hips/Torso,
per-window (shuffled test → no smoothing).

## Motivation — we feed the head a LOSSY summary
v1 uses the FM's **mean-pooled 512-d vector** ⊕ 520 handcrafted → LightGBM. But MOMENT/Mantis emit a
**full per-patch, per-channel token sequence**, most of which mean-pooling discards. Evidence this
matters: COMODO (arXiv:2503.07259) — per-channel **concat** > mean by +1.8–6.8 on IMU; attention-
pooling > mean across vision/audio/speech. A better *head* therefore **accesses information we
currently throw away** (possibly the rail-discriminative signal). This is a *different lever* than
Tier-1 (which reweighted fixed probabilities and is tapped out): the head produces *better*
probabilities from *richer* input.

## Hypotheses (falsifiable)
- **H-pool:** learned pooling (attention / multi-stat mean⊕max⊕std⊕GeM) over FM tokens beats
  mean-pool.
- **H-chan:** a **per-channel attention** head (learn to weight/combine the independently-embedded
  channels) recovers cross-axis coupling that channel-independent FMs discard. *(Novel for IMU FMs.)*
- **H-fuse:** token-level **cross-FM** fusion (lightweight cross-attention over 2 FMs' tokens) beats
  late-fusion voting.

## Candidate lightweight heads (all differentiable, frozen FM — NO backbone backprop)
1. mean-pool + tree (v1 baseline).
2. **Attention pooling** (small learned query over patches) → linear.
3. **Per-channel attention** (weights over channel-embeddings) → small MLP.
4. **Multi-stat pooling** (mean⊕max⊕std⊕GeM) → MLP.
5. **Cross-FM token cross-attention** (top-2 FMs from the bake-off).
6. **Handcrafted-fusion layer** (concat the head output with the 520 features → final linear/MLP).

## Protocol (identical to the rest of the pipeline)
Fit on User1 + val[FIT]; **select head/hparams on TUNE; read TEST once**. Per-class + macro reported;
BHT eval; per-window. All runs MLflow-logged + artifact-snapshotted (rule §8).

## Overfitting guards (the XGBoost-scar discipline — non-negotiable)
- **Tiny heads** (target < ~50k trainable params; lightweight is the point and the regularizer).
- Dropout + weight decay + **early-stop on TUNE** (never on TEST).
- **≥3 seeds**, report mean ± std; a gain within seed noise is **not** a gain.
- Keep the **handcrafted lane** (it's strong); the neural head must *complement*, not replace.
- Prefer the **simplest head that wins**; every extra head tried is a selection → discipline.
- ADOPT only if macro-F1 CI lower bound > v1 **and** no per-class regression on the locked split.

## What to build
1. `extract_embeddings.py --tokens` — cache per-patch/per-channel token embeddings (GPU, Helios).
2. `heads/` — a small PyTorch trainable-head module + honest-split training loop + calibration +
   per-class eval + MLflow. Reuse `shl2026`, `split.py`, `fit_cal_eval` conventions.
3. Ablation runner over the candidate heads → `HEAD_RESULTS.md`.

## Success criteria / outcomes
- **Win:** a lightweight head beats mean-pool+tree on the locked split (CI-positive, no per-class
  regression) → ship + it's the paper's method.
- **Flat (also publishable):** "mean-pool is sufficient for frozen IMU FMs on this data" — a clean
  negative + the rigorous eval is the contribution.

## Open: literature (filled by the 2026-06-25 deep-research pass)
What pooling/attention/channel/parameter-efficient head designs work for frozen TS/IMU FMs and HAR;
macro-F1/imbalance-aware head training; small-data overfit control. → see agent reports.
