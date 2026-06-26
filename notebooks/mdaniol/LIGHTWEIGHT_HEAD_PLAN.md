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

## Bake-off inputs (locked 2026-06-26)
Best single fused FM = **utica_V2 (0.8157)**; **top-2 = utica_V2 + mantisv2_V1 (0.8138)**, distinct
families → real diversity. These are the inputs for the cross-FM heads (#5) and the soft-vote.
*(MASTER dropped from the candidate FM pool: multimodal, no released weights, train-from-scratch.)*

## Candidate lightweight heads (all differentiable, frozen FM — NO backbone backprop)
1. mean-pool + tree (v1 baseline).
2. **Attention pooling** (small learned query over patches) → linear.
3. **Per-channel attention** (weights over channel-embeddings) → small MLP.
4. **Multi-stat pooling** (mean⊕max⊕std⊕GeM) → MLP.
5. **Cross-FM token cross-attention** (top-2 FMs from the bake-off).
6. **Handcrafted-fusion layer** (concat the head output with the 520 features → final linear/MLP).

**0. Calibrated soft-vote of the top-2 (BUILT — `modeling/voting_head.py`).** The safe, no-training
baseline for #5: per-FM fusion+calibration → equal / TUNE-weighted / +recal vote, gated KEEP iff it
beats the best single FM on the lock TEST by >+0.001. Run it first — it sets the bar cross-attention
(#5) must clear to justify its complexity. Job: `hpc/voting_head.sbatch`.

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

## Literature synthesis (deep research, 2026-06-25 — 3 HAR-FM agents)
Convergent, evidence-backed findings (citations are the load-bearing ones):
1. **Cheap fixed pooling is the workhorse; learned attention is the fragile part.** `mean⊕max⊕std⊕GeM`
   reliably beats mean at ~0 params, very low overfit risk (Okabe 1803.10963: *std more effective than
   attention*; adding attention alone *hurt*). Learned/attention pooling **overfits small data**
   (Unmute, arXiv:2509.24901: ±9 seed variance, falls below linear at 64-shot). → multi-stat pooling
   FIRST, not attention pooling.
2. **Per-channel concat > mean** (COMODO 2503.07259, +1.8/+6.8 pp on IMU) — the FM embeds channels
   independently; keep them separate.
3. **THE novel head = cross-channel mixing** (SE-block / channel self-attention over the 9 per-channel
   embeddings) — recovers cross-axis coupling a channel-independent FM *provably discards*; a tree can't.
   SE = low-param/low-risk version; attention = higher-variance upgrade. (THAT AAAI'21; SE/scSE 1808.08127.)
4. **Fusion = calibrated soft-voting over the top-2 FMs** (convex weights, calibrate-then-average):
   **+1–3 pp, lowest risk**, the SHL-2025-winner recipe. **NOT** token-level cross-attention (overfit;
   MBT 2107.00135 shows dense cross-attn redundant). "Diversity, not routing" — explains our router ≈0.
5. **Imbalance reframe:** Train/Subway is a **confusion** problem, not frequency → balanced-softmax/LDAM
   are off-target (Run already 0.96). Post-hoc logit-adjustment ≈ our calibration → **already captured**
   (Tier-1 tapped out, confirmed). The one targeted loss = **focal** (targets confusion *and* improves
   calibration; Mukhoti 2002.09437).
6. **FiLM fusion** of embedding + handcrafted (handcrafted steers/denoises the embedding; 1709.07871) —
   safer than concat on small data.
7. **Rail's only real shot:** gravity-canonicalization + **RAW mag axes** (EqNIO 2408.06321) — keeps
   orientation-robustness while restoring the directional mag signature that magnitude-streams destroyed
   (explains our H2 negative). High-variance.
8. **The 0.9→0.71 gap is likely EVALUATION (subject leakage), not head overfit** — HAR record-wise vs
   leave-subjects-out inflation ~10–14 pp ≈ our gap. Honest subject-disjoint selection is the real fix.
   (Our temporal+embargo + train-User1/test-Users2&3 mostly covers it; TUNE/TEST share users 2&3, but so
   does the challenge test, so it's appropriate here.)
9. **Zero-param robust baselines / ceiling check:** NCM+CL2N (SimpleShot 1911.04623) + k-NN geometry
   audit (DINO) tell us whether frozen MOMENT even separates the 8 classes across users; Pro² (2302.05441)
   bottleneck head for shift (+5–15%).

**DON'T** (convergent negatives): patch-attention pooling first · token-level cross-FM cross-attention ·
soft-F1 surrogates · temporal-attention heads (target the easy cadence classes) · graph-nets over axes ·
frequency-imbalance losses · kitchen-sink ensembling.

## Ranked build order (lowest-regret first)
1. **Enable token + per-channel extraction** (`extract_embeddings.py --per-channel` / `--tokens`). [GPU, prereq]
2. **Multi-stat pooling (mean⊕max⊕std⊕GeM) + per-channel concat → LGBM**, plus a **k-NN/NCM geometry
   audit** (the ceiling). Cheap, reliable, tree-compatible. [CPU]
3. **Cross-channel SE head** (the novel paper contribution); channel-attention as the upgrade — small
   PyTorch head harness, heavy regularization + seeds. [CPU/light-GPU]
4. **After the bake-off:** calibrated **soft-voting top-2 FMs** + **FiLM fusion** (embedding+handcrafted). [CPU]
5. **Targeted rail bets (high-variance, only if 2–4 plateau):** focal loss · gravity-canon + raw-mag ·
   cost-sensitive Train↔Subway penalty.

All under: select on TUNE / lock TEST once, ≥3 seeds (attention heads are high-variance), per-class +
the Train↔Subway off-diagonal reported, calibrate-before-average. Agent reports in session log + agent
memory (`reference_frozen_head_pooling.md`, `reference_har_head_architectures.md`,
`reference_imbalance_macrof1_heads.md`, `reference_cross_fm_fusion_heads.md`).

---

## Pre-registered runs (frozen before peeking at TEST)

### E-VOTE-01 — calibrated soft-voting of the bake-off top-2 (registered 2026-06-26)
Stage-1/2 of the 8-stage workflow, written **before** submitting; results go to
`VOTING_HEAD_RESULTS.md` + MLflow run `voting_utica_V2+mantisv2_V1`.

- **Hypothesis (H-fuse, late-fusion form):** a calibrated convex vote of the two best fused FMs
  (utica_V2 0.8157, mantisv2_V1 0.8138 — distinct families) beats the best single FM on the lock
  TEST, because different families make different errors (diversity, not routing — consistent with
  synthesis #4 and the SHL-2025 winner; explains our location-router ≈0).
- **Code / job:** `modeling/voting_head.py` → `hpc/voting_head.sbatch` (job `shl-vote`, Helios CPU).
- **Protocol:** mirrors `probe_fusion.py --use-split` exactly — fit on User1+val[FIT], calibrate +
  select weights on **TUNE**, read **TEST once**. Bag/Hips/Torso, per-window, macro + per-class
  (Train/Subway called out). Each single-FM row must reproduce its `BAKEOFF_SPLIT.md` value (built-in
  sanity check). Variants: equal vote · TUNE-weighted (coordinate-ascent simplex) · weighted+per-class-recal.
- **Decision rule (gated, fixed in advance):** **KEEP** the vote iff the TUNE-selected variant beats
  the best single FM on TEST by **> +0.001** macro-F1 **and** no per-class F1 regresses materially
  (esp. Train/Subway/Run); else **DISABLE** (single utica_V2 stays the base). Selection is on TUNE only.
- **Expected:** +0.005…+0.02 macro (synthesis #4: "+1–3 pp, lowest risk"). A flat/negative result is
  itself publishable ("late-fusion diversity exhausted; the gain must come from richer pooling").
- **Traceability:** registered in git (code auto-snapshotted by `track`); MLflow params (embs, split,
  weights, best_single) + bare `macro_f1` (lock) + per-class; artifacts = `VOTING_HEAD_RESULTS.md`
  + `.json`. Diff-test `tests/test_guardrails.py::test_voting_head_combiner_math` guards the vote math.
- **Next gate:** the KEEP/DISABLE TEST macro becomes the **bar the cross-channel SE head (#3) must
  clear** to justify its added complexity.
