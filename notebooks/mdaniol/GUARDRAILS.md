# SHL 2026 — Research Guardrails & Best Practices

**Purpose:** a quality firewall so we never "send bullshit to the HPC." Every
experiment must pass the **pre-HPC checklist** below before `sbatch`. Grounded in
the ML-leakage taxonomy (Kapoor & Narayanan, *Patterns* 2023), REFORMS reporting
standards (arXiv:2308.07832), and the Deep-Learning-for-HAR tutorial
(`TUTORIAL/dl-for-har`). Companion to `DATA_SPLITS.md` and `MOE_ARCHITECTURE.md`.

---

## 0. Prime directives
1. **Library-first.** Prefer a validated library to hand-written code. If you must
   hand-write numerics, **validate it against an independent library** (we do:
   features vs scipy/statsmodels/tsfel in `validate_features.py`; augmentations vs
   the analytic Rodrigues formula in `validate_augment.py`).
2. **The lock-test (TEST) is sacred.** Select/tune on TUNE; read TEST **once** to
   confirm. Never choose a config, threshold, β, or feature by TEST.
3. **Mirror the deployment distribution.** Evaluate on **Bag/Hips/Torso** (the real
   test has no Hand) and **per-window** (the test is shuffled — no temporal context).
4. **Determinism or it didn't happen.** Fixed seeds everywhere; record git SHA +
   `val_split.json` with every result (via `shl2026.track`).
5. **Keep the strong baseline.** Don't drop Branch A (handcrafted+LGBM) without
   lock-test evidence that something beats it.

---

## 1. Leakage taxonomy → how we prevent each (Kapoor & Narayanan)
| Type | Leak | Our prevention | Enforced by |
|---|---|---|---|
| L1.1 | no test set | held-out **TEST** lock slice | `split.py` |
| L1.2 | preprocessing on train+test | scaler/PCA inside an **sklearn `Pipeline`**, fit on FIT only | `emb_pca_heads`, `freq_mag_branch`; `test_*_pipeline_is_leakproof` |
| L1.3 | feature selection on train+test | features fixed a priori; freq+mag subset by **domain prefix**, never data-driven on TEST | `select_freq_mag` (static) |
| L1.4 | duplicate rows across split | adjacent (near-duplicate) windows kept together via **contiguous per-class blocks** | `split.py` |
| L2 | illegitimate features | per-window only; no label-/future-derived features | feature bank design |
| L3 | test ≠ distribution of interest | **BHT-only** eval mirrors the real test | all runners (audited) |
| L3.1 | temporal leakage | contiguous blocks; **no HMM/Viterbi/median/rolling** on shuffled test | `test_no_temporal_smoothing_in_modeling` |
| L3.2 | train/test non-independence | contiguous time-blocks (correlated neighbours stay together) | `split.py` |
| L3.3 | sampling bias in test | class-stratified blocks; every class in every slice | `split.py` |
| (+) | **selection on the lock-test** (multiple comparisons) | **select on TUNE, confirm on TEST**; oracle excluded (not deployable) | `moe_experiment`, `rail_expert` (audited) |

## 2. Library-first policy (what to reuse, not rewrite)
| Need | Use (library) | Not |
|---|---|---|
| Metrics / macro-F1 / confusion | `shl2026.evaluate_predictions` (sklearn) | hand-rolled F1 |
| MLflow logging + provenance | `shl2026.track` (no-op safe) | a custom wrapper |
| Preprocessing (scale/PCA/L2) | sklearn `Pipeline`+`StandardScaler`/`PCA`/`Normalizer` | manual fit/transform (leak risk) |
| Linear/tree heads | sklearn `LogisticRegression`/`RidgeClassifier`/`SGDClassifier`, LightGBM | bespoke optimizers |
| DSP (PSD, filters, rotation, interp) | scipy (`signal`, `spatial.transform`, `interpolate`) | hand math (unless validated) |
| Tests | **pytest** (team suite in `tests/`) | ad-hoc `assert` scripts |
| Submission shape/labels | `shl2026.submission.validate` | manual checks |
| Embedding cache / heads zoo | `shl2026.embeddings` / `shl2026.heads` (where applicable) | re-loaders |
Optional, if a real need arises (add to your PERSONAL env, never the team lock):
`pandera`/`deepchecks` (data-schema checks), `aeon`/`sktime` (ROCKET), `hypothesis`
(property tests).

## 3. Validation discipline (REFORMS-aligned reporting)
- Report **per-class F1 + macro** (never macro alone) and the **selection-lock gap**
  = macro(TUNE) − macro(TEST). Large gap ⇒ overfit the selection set.
- Watch the **rare/hard classes**: Run (~4.3%), Train/Subway, Car/Bus.
- A change ships **only if** it beats the global baseline **on TEST**.
- Touch TEST as little as possible: many experiments each peeking at TEST is itself
  a multiple-comparisons leak — decide on TUNE, confirm the single winner on TEST.

## 4. Determinism & reproducibility
- `shl2026.track(seed=0)` seeds Py/NumPy/Torch; LightGBM/PCA `random_state=0`;
  `split.py` is RNG-free; augmentation RNG is explicitly seeded.
- Every MLflow run auto-stamps git SHA, dirty flag, Slurm job, node, grant, env.
- Commit + push before runs you keep (dirty tree ⇒ only a code-snapshot artifact).

## 5. Pre-HPC checklist (run locally before every `sbatch`)
```bash
# 1. guardrail + unit tests (seconds, synthetic data)
SHL-local/bin/python -m pytest tests/test_guardrails.py -q
# 2. numeric validation suites (libraries as ground truth)
SHL-local/bin/python notebooks/mdaniol/feature_extraction/validate_features.py
SHL-local/bin/python notebooks/mdaniol/modeling/validate_augment.py
SHL-local/bin/python notebooks/mdaniol/modeling/moe_combine.py
# 3. the script imports + argparse resolve (catch typos before the queue)
SHL-local/bin/python notebooks/mdaniol/modeling/<script>.py --help >/dev/null
# 4. sbatch is well-formed + scheduler accepts it (consumes nothing)
bash -n notebooks/mdaniol/hpc/<job>.sbatch
sbatch --test-only notebooks/mdaniol/hpc/<job>.sbatch
```
Green on all four ⇒ safe to submit. Red ⇒ fix before spending GPU-hours.

## 6. When adding any NEW experiment
1. Reuse loaders/metrics/track/Pipeline — don't duplicate (see §2).
2. Fit all preprocessing on FIT; calibrate/select on TUNE; confirm on TEST.
3. Restrict eval to Bag/Hips/Torso; keep it window-independent.
4. Add a guardrail test if it introduces a new invariant.
5. Log via `track` + write a versioned `*_RESULTS.md`; commit before submit.

## References
- Kapoor & Narayanan, *Leakage and the reproducibility crisis in ML-based science*,
  Patterns 4(9), 2023 — arXiv:2207.07048.
- *REFORMS: Reporting Standards for ML-Based Science*, arXiv:2308.07832.
- Bock et al., *Deep Learning for HAR* tutorial — `TUTORIAL/dl-for-har` (subject-wise
  validation; window-before-split; no shuffle-split).
- Internal: `DATA_SPLITS.md`, `MOE_ARCHITECTURE.md`, `validate_features.py`,
  `validate_augment.py`.
