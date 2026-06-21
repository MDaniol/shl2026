# Working rules for `notebooks/mdaniol/` (SHL 2026)

**Before doing anything here, follow these governing docs (read in this order):**
1. `AI_GUIDELINES.md` — the mandatory 8-stage dev workflow + coding standards.
2. `GUARDRAILS.md` — leakage taxonomy, invariants, library-first policy, pre-HPC checklist.
3. `DATA_SPLITS.md` — the train/eval/test split contract.
4. `MOE_ARCHITECTURE.md` — architecture map, file map, versioning, MLflow convention.

## Non-negotiables (every change)
- **8-stage workflow** (research → plan → implement → audit → validate →
  **data-leakage audit** → methodological assessment → commit+log). Stage 6 is a
  **hard gate**; tick the `AI_GUIDELINES.md §3` checklist in the commit message.
- **Select on TUNE, confirm on TEST once.** Never tune/select on the lock-test.
- **Evaluate on Bag/Hips/Torso** (no Hand) and **per-window** (test is shuffled — no
  temporal smoothing / cross-window features).
- **Library-first.** Reuse `shl2026.*` (`track`, `evaluate_predictions`, `embeddings`,
  `make_head`, `write_submission`) + existing `modeling/` helpers (`fit_cal_eval`,
  `calibrate`, `aligned_proba`, `moe_combine`, `load_split_with_location_map`). Hand-
  written numerics need a differential test vs a library.
- **Self-explanatory, simple code.** Plain over clever; docstrings state the invariant.
- **Reuse the team infra**: `env_mdaniol.sh`, the `hpc/*.sbatch` structure, MLflow via
  `track`, the `tests/` suite. Don't build parallel pipelines or touch the team
  `pyproject.toml`/`uv.lock`.

## Pre-HPC gate (must be green before any `sbatch`)
```bash
SHL-local/bin/python -m pytest tests/test_guardrails.py -q
SHL-local/bin/python notebooks/mdaniol/feature_extraction/validate_features.py
SHL-local/bin/python notebooks/mdaniol/modeling/validate_augment.py
bash -n notebooks/mdaniol/hpc/<job>.sbatch && sbatch --test-only notebooks/mdaniol/hpc/<job>.sbatch
```
