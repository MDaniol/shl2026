# SHL 2026 — AI Development Guidelines

How AI-assisted development must proceed in this project so every change meets the
documented research standards **before it reaches the HPC**. This is the process
doc; the technical invariants live in `GUARDRAILS.md`, the split contract in
`DATA_SPLITS.md`, the architecture in `MOE_ARCHITECTURE.md`. **Read this first.**

---

## 1. The development workflow (mandatory stages, in order)
No stage is skippable. A change is not "done" until stage 8.

1. **Research.** Understand the task; find prior art in the repo and the literature.
   Cite sources for any methodological choice. Prefer evidence over assumption.
2. **Plan (reuse-first).** Identify the existing functions/utilities to extend
   (`shl2026.*`, `probe_fusion`, `split`, `metrics`, `moe_combine`). Do **not** build
   a parallel pipeline. State what you will reuse vs. what is genuinely new.
3. **Implement (library-first + simple).** See §2 coding standards. Use a validated
   library wherever one exists; write the minimum new code, as plainly as possible.
4. **Audit (code review).** Re-read the diff: correct, minimal, no duplication, names
   and docstrings clear, error paths sane.
5. **Validate (numeric).** Run the unit/property tests and the validation suites
   (`pytest tests/`, `validate_features.py`, `validate_augment.py`, `moe_combine.py`).
   Hand-written numerics must be checked against an independent library.
6. **★ Data-leakage audit (gate — never skip).** Explicitly walk the leakage
   checklist in §3 and the Kapoor–Narayanan map in `GUARDRAILS.md §1`. State, in the
   commit/PR message, that each item passed. If any fails, stop and fix.
7. **Methodological assessment.** Confirm the change still satisfies the
   `GUARDRAILS.md` prime directives (select-on-TUNE, BHT eval, determinism, baseline
   kept) and the pre-HPC checklist is green.
8. **Commit + log (full traceability).** Versioned result file **and** a `shl2026.track`
   MLflow run that is reproducible on its own. The run MUST:
   - log `params=` including the **split scheme** (`"split": args.split.stem`) + the run's
     knobs, and `tags=` the phase + KEEP/DISABLE decision;
   - log `log_eval(r_te, prefix="test_")` + `log_eval(r_tu, prefix="tune_")`, a **bare
     `macro_f1`** (= lock-test macro, so `leaderboard()` ranks it), and per-class F1;
   - **snapshot artifacts** via `run.log_artifact(...)`: the result file (`*_RESULTS.md`/
     JSON), the fitted model + calibration weights, and the split file (`track` already
     auto-snapshots the code).
   A missing `track`/artifact snapshot fails this gate, exactly like a leakage miss.
   Commit with a message recording the leakage-audit result and the git SHA discipline.

Only after stage 8 may a job be `sbatch`-ed.

---

## 2. Coding standards — self-explanatory & simple
The reader (human or AI, six months later) must understand the code without guessing.

- **Library over hand-rolled.** If scipy/sklearn/`shl2026` does it, call it. Hand-
  written numerics require a differential test vs the library (see `validate_*`).
- **Plain beats clever.** Small functions, one responsibility, descriptive names
  (`load_split_with_location_map`, not `lsm`). No dense one-liners that hide intent.
- **Docstrings state the contract, not the obvious.** Inputs, output shapes, and the
  *invariant* the function guarantees (e.g. "fits PCA on FIT only — no TEST leakage").
- **Comments explain WHY, not WHAT.** The *what* should be readable from the code.
- **Make shapes explicit.** Print/annotate array shapes; assert split lengths match.
- **Fail loudly, early.** Assert preconditions; never silently degrade results.
- **Determinism by default.** Pass/seed RNGs; no `Date.now()`-style nondeterminism.
- **No duplication.** Import the existing helper; don't copy it.

---

## 3. ★ Data-leakage audit checklist (the required gate, stage 6)
Tick every item against the change (mirrors Kapoor & Narayanan, *Patterns* 2023):

- [ ] **Train/Tune/Test separation** — TEST never used to fit, scale, PCA, calibrate,
      select features, or choose a config/threshold/β. (L1.1–L1.3)
- [ ] **Preprocessing inside a Pipeline** — scaler/PCA/normalizer fit on FIT only. (L1.2)
- [ ] **No data-driven feature selection on TEST** — subsets are a-priori/by-domain. (L1.3)
- [ ] **No near-duplicate split** — adjacent/correlated windows stay in one slice
      (contiguous blocks, not random). (L1.4, L3.2)
- [ ] **Legitimate features only** — per-window, no label-/future-derived inputs. (L2)
- [ ] **Eval mirrors the test** — Bag/Hips/Torso only; class-stratified. (L3, L3.3)
- [ ] **No temporal smoothing** — no HMM/Viterbi/median/rolling over test rows; the
      model is window-independent (test is shuffled). (L3.1)
- [ ] **Select on TUNE, confirm on TEST once** — no best-on-TEST cherry-picking; the
      lock-test is read once for the single chosen config.
- [ ] **Determinism** — seeds fixed; git SHA + split counts logged.

Automated coverage: `pytest tests/test_guardrails.py` checks the Pipeline-leakage,
shuffle-invariance, no-temporal-smoothing, normalization, and determinism items.
The select-on-TUNE and eval-mirrors-test items are enforced in the runners and must
be re-confirmed by reading the diff.

---

## 4. Definition of done
- Stages 1–8 complete; the §3 checklist ticked in the commit message.
- `pytest tests/` + the three `validate_*` suites pass locally.
- Pre-HPC checklist (`GUARDRAILS.md §5`) green; `sbatch --test-only` accepted.
- Result logged to MLflow (`shl2026.track`) **with artifacts snapshotted** (result file,
  model + calibration weights, split; code auto-snapshotted) and a versioned `*_RESULTS.md`
  written — split scheme + bare `macro_f1` present on the run.
- The strong baseline is intact unless lock-test evidence supersedes it.

## References
`GUARDRAILS.md` (invariants + library policy), `DATA_SPLITS.md` (split contract),
`MOE_ARCHITECTURE.md` (architecture + versioning). External: Kapoor & Narayanan
(arXiv:2207.07048), REFORMS (arXiv:2308.07832), Bock et al. DL-for-HAR tutorial.
