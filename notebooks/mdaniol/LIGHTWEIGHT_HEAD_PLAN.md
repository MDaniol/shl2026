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
1. ✅ **Per-channel extraction BUILT** (`extract_embeddings.py --per-channel` → `(n,C,d)`, `_pc` tag;
   job `extract_per_channel_helios.sbatch`, utica+mantisv2 V1; diff-tested). Mantis/MOMENT are
   channel-independent so per-channel embed = pre-pool rep, mean-over-C reproduces the pooled vector.
   *(Token-level `--tokens` still TODO; per-channel is the load-bearing one for the cross-channel head.)*
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

**RESULT (concluded 2026-06-26 → KEEP):** `vote:weighted+recal` TEST **0.8342** vs best single
utica_V2 **0.8213** (**+0.0129**, TUNE-selected 0.8614). Per-class clean (Run+0.02, Car+0.03,
Bus+0.05, Train+0.03; Still −0.02 only). **The bar for the cross-channel SE head is now 0.8342.**
⚠️ single utica_V2 here (0.8213) ≠ `BAKEOFF_SPLIT.md` (0.8157): pre-registered reproduce-check
failed → bake-off likely stale vs regenerated features; re-confirm before citing absolute numbers
(vote Δ unaffected — all rows share data). Submission: the vote (0.8342) > single, so ship the vote
(needs `submit_vote.py`, Phase-B), not single utica_V2.

### E-HEAD-01 — cross-channel lightweight heads (registered 2026-06-26)
Pre-registered before any cluster run; code `modeling/head_xchannel.py` (+ `hpc/head_xchannel_helios.sbatch`,
job `shl-head`), gate green, diff-tested (`test_head_xchannel_forward_and_se_uses_channels`).

- **Hypothesis (H-chan):** a head that learns to MIX the C independently-embedded channels
  (Squeeze-Excitation over channels) recovers cross-axis coupling that channel-independent FMs
  discard, beating the channel-mean baseline. Inputs = per-channel `(n,C,d)` `_pc` embeddings.
- **Heads (all → [pooled ⊕ 520 handcrafted] → MLP → 8, frozen FM, no backprop):** `mean` (baseline,
  reproduces single-FM fusion) · `concat` (COMODO) · `multistat` (mean⊕max⊕std⊕GeM, the workhorse)
  · **`se` (novel)**. Tiny heads (SE pool ≈ C²/r params); dropout 0.3 + weight-decay 1e-4 + early-stop
  on TUNE; **≥3 seeds**, report mean±std (attention/learned pooling is high-variance).
- **Protocol:** fit User1+val[FIT]; calibrate + select head on **TUNE**; lock **TEST** once. Per-class
  + macro + **ECE** + **missing-channel robustness** (`eval_metrics`). Handcrafted StandardScaler fit
  on FIT only (leak-safe).
- **Decision rule (frozen):** **KEEP** the cross-channel SE head iff its TEST mean − std (CI lower
  bound) **> the mean baseline** AND no per-class regression (esp. Train/Subway/Run). To become the
  submission it must additionally clear the **E-VOTE-01 bar 0.8342** (likely via a head-vote of two FMs).
- **Expected:** SE > mean by +0.5–2 pp if H-chan holds (COMODO-scale); a flat result is publishable
  ("mean-pool sufficient for frozen IMU FMs"). multistat is the low-risk fallback win.
- **Prereq:** `extract_per_channel_helios.sbatch` (job `shl-extract-pc`) → `embeddings/<fm>_V1_pc/`.
- **Traceability:** MLflow run `head_<emb_pc>` (bare macro_f1 = TUNE-winner lock + per-head + per-class
  + ECE) + `HEAD_RESULTS.md` artifact; pre-reg here; commit; diary conclusion on completion.

**RESULT (concluded 2026-06-27):** H-chan **weakly confirmed but does NOT beat the vote.** SE >
channel-mean by **+0.006** (utica_V1_pc 0.8058 vs 0.8001, CI-positive → KEEP) and **+0.004**
(mantisv2_V1_pc 0.8137 vs 0.8100, within-noise → DISABLE). Best head 0.814 ≪ E-VOTE-01 vote **0.834**
and below even utica_V2 fusion (0.82) — heads ran on the weaker V1 (axes) variant, and a small MLP on
per-channel V1 ≈ V1 fusion + a sliver from SE; it can't beat LGBM-fusion + the calibrated vote. SE &
multistat are **well-calibrated** (ECE 0.022–0.026 vs concat 0.052); rail (Train/Subway) unmoved. **Net:
cross-channel mixing is a real-but-small, FM-dependent effect → a paper finding (channel-mean nearly
sufficient for frozen IMU FMs; good calibration), NOT a submission component.** v3 vote stays best.
Head-vote (E-HEAD-02) would land ~0.81 < 0.834 → not worth running.

### E-FMDIV-HEAD — cross-channel SE head on the spectrogram-FM voters (registered 2026-06-27)
Pre-registered before peeking. Reuses `head_xchannel.py` (E-HEAD-01 machinery, validation-cleared)
on the NEW per-channel spectrogram-FM embeddings; `hpc/head_xchannel_helios.sbatch` (writes
per-FM `HEAD_RESULTS_<tag>.md`). Research basis: `VISION_SOUND_FM_RESEARCH.md` (TiViT 2506.08641).

- **Inputs:** `ast_V2_pc`, `dinov2_V2_pc` — `(n, 5, d)` per-channel embeddings (5 orientation-invariant
  magnitude channels embedded independently by the frozen AST / DINOv2). (ImageBind is native-joint
  `(n,1024)`, not channel-independent → it goes through the bake-off, not this head.)
- **Hypothesis (H-chan, transferred):** learning to MIX the 5 frozen spectrogram-FM channel-embeddings
  (SE block) beats their channel-mean, recovering cross-sensor coupling the mean discards — same
  mechanism that gave +0.006 on the temporal FMs (E-HEAD-01).
- **Protocol:** `head_xchannel` runs mean / concat / multistat / **SE** over the 5 channels ⊕ 520
  handcrafted → MLP → 8 classes. Fit FIT, select head+seed on TUNE, lock TEST once, ≥3 seeds (mean±std),
  per-class + ECE + missing-channel robustness, BHT-only (TEST excludes Hand by split construction).
- **Decision rule (frozen):** (1) per FM, KEEP the SE head iff its TEST mean−std > the mean baseline
  (does cross-channel mixing help this FM?); (2) is any spectrogram-FM head **competitive** with the
  0.834 vote / its single-FM fusion? Only if a head is competitive do we build the **head-vote**
  (combine its probas with the UTICA+MantisV2 vote) — evidence-gated, to avoid E-HEAD-01's outcome
  (head real-but-small, didn't beat the vote). A flat result is a clean publishable ablation.
- **Run:** `EMB_PC=ast_V2_pc sbatch …/head_xchannel_helios.sbatch` ; `EMB_PC=dinov2_V2_pc sbatch …`.
- **Traceability:** MLflow run `head_<emb_pc>` (bare macro_f1 + per-head + per-class + ECE) +
  `HEAD_RESULTS_<tag>.md` + split snapshot; pre-reg here; diary conclusion on completion.

### E-FMDIV-VOTE — fold the spectrogram-FM(s) into the calibrated vote as diversity voters (registered 2026-06-27)
Pre-registered **before peeking at the N-way TEST**. Reuses the E-VOTE-01 machinery unchanged
(`voting_head.py` / `hpc/voting_head.sbatch`); the only new artifact is the **mean-pooled voter dir**
built by `hpc/pool_emb_helios.sbatch` → `pool_per_channel.py`. Motivated by the E-FMDIV-HEAD result:
DINOv2 **fails the bar standalone** (best cross-channel head TEST 0.797 ≪ 0.834) BUT its per-class
profile is relatively strong on the **vehicle pair** (Car 0.92 / Bus 0.79, best seed) — a cross-domain
(vision-ViT) error pattern that *may* decorrelate from the temporal FMs. Standalone strength is the
wrong test for a voter; diversity is.

- **Inputs:** `dinov2_V2` — the channel-mean of `dinov2_V2_pc`, which is **number-identical** to a
  non-per-channel extraction (`emb = pooled.mean(1)`), locked by
  `tests/test_guardrails.py::test_pool_per_channel_equals_channel_mean`. Fused as `emb ⊕ 520
  handcrafted → LightGBM`, the **same recipe** as the utica_V2 / mantisv2_V1 voters (apples-to-apples).
- **Hypothesis (H-div):** adding the decorrelated cross-domain DINOv2 fusion voter to the
  utica_V2 + mantisv2_V1 calibrated vote lifts macro-F1 — concentrated on **Car/Bus**, where DINOv2 is
  relatively strong — without hurting Run or the rail pair.
- **Protocol (identical to E-VOTE-01):** per FM, fit `emb⊕520→LGBM` on FIT (User-1 + validation[FIT]),
  per-class calibrate on TUNE; convex vote **weights + per-class recal selected on TUNE**; **lock TEST
  once** (BHT-only). Report macro-F1 on TUNE+TEST, **per-class F1**, ECE, and the **selection-lock gap**.
  Deterministic (`deterministic=True, force_col_wise=True`).
- **Decision rule (frozen):** KEEP the 3-way vote into the submission **iff its TEST macro > 0.8342**
  (the E-VOTE-01 bar) — measured on TEST, never selected on TUNE. Secondary (does not gate shipping):
  record the **Car/Bus per-class delta** vs the 2-way vote as a diversity finding — a flat macro with a
  real pair lift is a clean publishable ablation, not a submission. **Overfit guard:** more voters = more
  TUNE-selected weights → if the 3-way selection-lock gap inflates materially vs the 2-way (≈0.027),
  treat the TUNE gain as overfit and DISABLE regardless of TUNE.
- **Honest prior:** the voter is mostly the 520 handcrafted features + a weak DINOv2 add-on, so expect
  **≈0.834 ± a little**; marginal-to-neutral is the likely outcome, a clear lift would be a pleasant
  surprise. A null result still answers the E-FMDIV diversity question.
- **Pre-registered variants (same gate):** **4-way** `+imagebind_V0` (already extracted) and **5-way**
  `+ast_V2` (after the AST extraction + its own `pool_emb_helios` finish) — to test all diversity voters
  jointly, not just DINOv2.
- **Run:** `POOL=$(sbatch --parsable hpc/pool_emb_helios.sbatch)` ;
  `EMBS=utica_V2,mantisv2_V1,dinov2_V2 sbatch --dependency=afterok:$POOL hpc/voting_head.sbatch`.
- **Traceability:** `shl-pool` log + `embeddings/dinov2_V2/` on group storage; MLflow run `vote_*`
  (bare macro_f1 = TEST lock so `leaderboard()` ranks it, + per-class + weights + split scheme) +
  `VOTING_HEAD_RESULTS.md` row + split snapshot; this pre-reg; diary conclusion on completion.
- **OUTCOME (2026-06-28): DISABLE.** 3-way +dinov2 0.8281, +imagebind 0.8310 — both < 0.8342 (all of
  equal/weighted/+recal). Overfit guard fired (selection-lock gap +0.0345/+0.0320 vs 2-way's +0.028);
  rejected even on the vehicle pair (mantis already > both). E-FMDIV program closed; 2-way 0.8342 stands.

### A1 — pair-separability diagnostic for the two caps (registered 2026-06-28)
Pre-registered **before peeking**. Decides whether the hard-pair program is worth building, so we don't
repeat the rail-expert failure. Code `modeling/pair_separability.py` (+ `hpc/pair_separability_helios.sbatch`),
guarded by `tests/test_guardrails.py::test_pair_separability_probe_leakage_safe`. Basis: deep-research
top-3 (har-fm + har-dl agents, 2026-06-27) — both lanes converged on "the rail/vehicle ceiling is an
information question; measure it first."

- **Question:** can a binary **linear probe** on the submission representation separate **Train↔Subway**
  and **Car↔Bus** on HELD-OUT data? High ⇒ the separating info is in the frozen rep and the multiclass
  head leaves it on the table ⇒ build **B1** (Fisher-axis feature) / **B2** (gated OvO specialist + Bayes
  override, Ashqar 2006.06945). Low ⇒ the encoder collapsed the pair ⇒ DISABLE the hard-pair program.
- **Protocol (leakage-safe, identical to submit_vote):** probe fit ONLY on FIT = train(all) +
  validation[FIT]; scored on the locked TEST = validation[TEST] (BHT-only); per-window; deterministic
  (LogisticRegression, standardized, class-balanced, seed 0). Never selects on TUNE/TEST. Reps: `fusion`
  (emb⊕520, the submission rep — default), `emb` (FM only), `handcrafted` (520 only, no-FM control).
- **Decision rule (frozen):** report held-out **balanced-accuracy + F1** per pair. **≥0.80 ⇒ SEPARABLE**
  → proceed to B1 then B2 (gated on nested-TUNE-CV gain > fold-std). **≤0.65 ⇒ COLLAPSED** → DISABLE the
  hard-pair program; document the ceiling as an information limit for the paper. **0.65–0.80 ⇒ MARGINAL**
  → decide by ROI. Run `fusion` and `handcrafted` to see whether the FM adds pair-separating signal.
- **Run:** `EMB=utica_V2 REP=fusion sbatch hpc/pair_separability_helios.sbatch` (then `REP=handcrafted`).
- **Traceability:** MLflow run `pairsep_<emb>_<rep>` (per-pair bal-acc + F1) + `PAIR_SEPARABILITY.md`
  + split snapshot; this pre-reg; diary conclusion on completion.

### C1 — macro-F1-aware additive-logit decision rule (registered 2026-06-28)
`modeling/decision_rule.py` (+ `hpc/decision_rule_helios.sbatch`). The multiplicative `calibrate()`
can't lift an under-emitted class at p→0; the additive log-bias `argmax(log p_k + b_k)` (GLA
2310.08106, joint coordinate-descent on TUNE per Lipton 1402.1892) can.
- **Gate (frozen):** KEEP iff best-on-TUNE rule beats the within-run champion on TEST **and** the
  **paired-bootstrap Δ CI excludes 0** (the correct test, not marginal-CI overlap) **and** the
  selection-lock gap ≤ champion+0.01. SLD informational-only (transductive → rules-gated).
- **OUTCOME 2026-06-28: KEEP.** `mult+additive` TEST 0.8383 vs 0.8339, **paired Δ +0.0044
  CI[+0.0026,+0.0063]**, gap *smaller* than champion; Run rebalanced 0.95→0.97 → shipped as **v4**
  (`submit_vote --decision additive`), gated on the prior-robustness check (`robustness_diag`).

### E-RAIL — gated Train↔Subway disambiguator (registered 2026-06-28, before peeking)
`modeling/rail_disambig.py` (+ `hpc/rail_disambig_helios.sbatch`). Data-justified by `robustness_diag`:
Subway→Train errors are 99% top-2-recoverable, Train→Subway 80% — the model ranks the pair adjacent
and picks the wrong one.
- **Hypothesis:** a binary Train-vs-Subway head (emb⊕520, FIT-only) re-splitting the champion's
  within-pair mass (blend β on TUNE) recovers a share of those flips without touching other classes.
- **Gate (frozen):** KEEP iff the rail TEST macro beats the champion with **paired Δ CI excluding 0**.
  Same discipline that correctly DISABLEd the earlier rail expert — a good TUNE number alone ships nothing.
- **Traceability:** MLflow `rail_*` (macro + paired CI + bin pair-acc + β) + `RAIL_DISAMBIG_RESULTS.md`.

### orient — orientation-invariant features vs Torso fragility (registered 2026-06-28, before peeking)
`modeling/orient_features.py` (+ `hpc/orient_features_helios.sbatch`). Motivated by `robustness_diag`:
Torso macro 0.766 vs Bag/Hips ~0.86 (Bike=0.44); the hidden test has no location label.
- **Hypothesis:** gravity-V/H + cross-sensor SO(3) invariants (rotation-invariant by construction —
  unit-tested) appended to the 520 improve generalization across placements, especially **Torso**.
- **Gate (frozen):** KEEP iff emb⊕520⊕orient beats emb⊕520 on the honest TEST (report per-location;
  the Torso Δ is the target). Honest-eval, TEST locked once.
- **Traceability:** MLflow `orient_*` (macro + per-location Torso) + `ORIENT_FEATURES_RESULTS.md`.

### P0 diagnostics — ceiling / robustness / ablation (registered 2026-06-28)
`ceiling_diag.py` (oracle + confusion-collapse ceiling + FM error-correlation + bootstrap CI),
`robustness_diag.py` (confusion structure + per-location + prior-sensitivity), `ablation_rep.py`
(fm/hc/fusion + gap + HC importance share + top-k stability). Analysis-only on the locked TEST
(select nothing). All MLflow-traced; outcomes recorded in EXPERIMENT_LOG diary 2026-06-28.
