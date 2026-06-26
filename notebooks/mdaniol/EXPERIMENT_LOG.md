# SHL 2026 — experiment log (progress, results, why)

Concise running log. Detail lives in `WINNING_STRATEGY.md`, `VEHICLE_EXPERIMENT_PLAN.md`,
`DATA_SPLITS.md`, and the per-experiment `*_RESULTS.md` + MLflow. Metric = macro-F1, honest
temporal+embargo split, Bag/Hips/Torso eval, per-window (test is shuffled → no smoothing).

## Headline status
- **New best overall (E-VOTE-01, 2026-06-26): calibrated soft-vote of utica_V2 + mantisv2_V1
  (weighted+recal) = 0.8342 macro-F1** (temporal lock; **+0.013** over best single). KEPT.
- **Best single base:** frozen **UTICA (V2)** ⊕ 520 handcrafted → LightGBM + calibration =
  **0.8213** (current features; bake-off table said 0.8157 — ⚠️ re-confirm, features regenerated).
  Beats old v1 (MOMENT-small_V1 ≈ 0.806). UTICA, not MOMENT, is the strongest single FM.
- **All 11 FMs add positive Δ vs handcrafted-only (0.7909)** (+0.003…+0.025); embeddings *alone*
  lose to handcrafted → the FM value is in **fusion, not replacement** (confirms the survey's
  reconstruction-pretraining prior). Top-2 for soft-voting = **utica_V2 + mantisv2_V1** (0.8138).
- **Realistic ceiling ~80s** (smoothing is dead on a shuffled test), so ~0.82 is competitive.
- Tier-1 post-hoc is **tapped out** (KEEP v1; oracle ceiling < v1). Remaining edge = the
  **lightweight head** (soft-voting top-2, then cross-channel) + rigor for the paper.
- **New-FM hunt closed (verify-before-invest paid off 4×):** oneHAR (not in repo), MASTER (no
  weights, train-from-scratch), **SensorLM** (Google — training code only, no weights, proprietary
  Fitbit data, sensor-language not IMU embedder), **LSM** (Google — proprietary minute-level wrist
  HR/EDA/skin-temp, no weights, wrong modality) — **all dropped**. **CrossHAR** modality-OK but
  small/low-upside + weights unconfirmed → low priority. The bake-off settled the FM lane (~+0.025
  ceiling); gains now come from the **head + vote**. Only **UniMTS** (public weights, packer stubbed)
  is worth a future bet — gated behind the head.

## Job tracker (sbatch / driver / status)
`squeue --me` shows the **job-name**. GPU extraction = Athena (x86 A100) **or Helios GH200 (aarch64)**;
CPU work = Ares / Helios-CPU (x86). Helpers: `env_mdaniol.sh` (arch-aware), `link_data.sh`,
`stage_to_group.sh`, `setup_env.sh` (x86) / `setup_env_helios.sh` (aarch64), `add_fm_deps[_helios].sh`.

| job-name | sbatch | driver | cluster | status | output |
|---|---|---|---|---|---|
| `shl-tier1` | `tier1.sbatch` / `_ares` / `_helios` | `tier1_experiment.py` | Ares/Helios CPU | ✅ done — **KEEP v1** (post-hoc tapped out) | `TIER1_RESULTS.md` |
| `shl-extract-all` | `extract_all_helios.sbatch` (array) | `extract_embeddings.py` (MLflow-logged) | **Helios GH200** | ✅ done — all 11 FM sets extracted (~1800 win/s) | `embeddings/<fm>_<var>/` + MLflow |
| `shl-extract` | `extract_fm.sbatch` / `extract_fm_helios.sbatch` | `extract_embeddings.py` | Athena / Helios GPU | (per-model variant) | `embeddings/<fm>_<var>/` |
| `shl-probe` | `probe_fusion.sbatch` / `_ares` | `probe_fusion.py` | Helios CPU | ✅ done — **utica_V2 = 0.8157** (new best base) | `BAKEOFF_SPLIT.md` |
| `shl-vote` | `voting_head.sbatch` | `voting_head.py` | Helios CPU | 🔨 ready — soft-vote top-2 (utica_V2+mantisv2_V1) | `VOTING_HEAD_RESULTS.md` |
| `shl-extract-pc` | `extract_per_channel_helios.sbatch` (array) | `extract_embeddings.py --per-channel` | **Helios GH200** | 🔨 ready — per-channel V1 (utica+mantisv2) for the cross-channel head | `embeddings/<fm>_V1_pc/` (n,C,d) + MLflow |
| `shl-head` | `head_xchannel_helios.sbatch` | `head_xchannel.py` (MLflow) | **Helios GH200** | 🔨 ready — E-HEAD-01 cross-channel heads (after `_pc`) | `HEAD_RESULTS.md` |
| `shl-tta` | `tta_embeddings.sbatch` | `extract_embeddings.py --tta-k` | Athena/Helios GPU | 🔨 ready | `embeddings/..._tta*/` |
| `shl-submit` | `submit_helios.sbatch` / `submit.sbatch` | `submit_fusion.py` (MLflow-tracked) | Helios CPU / Athena | ✅ v1 done; 🔨 v2 ready (utica_V2, 0.8213) | `AGH_predictions_v2_utica_V2-fusion.txt` |
| `shl-submit-vote` | `submit_vote_helios.sbatch` | `submit_vote.py` (MLflow-tracked) | Helios CPU | 🔨 **v3 ready — ship the vote 0.8342** (utica_V2+mantisv2_V1) | `AGH_predictions_v3_vote.txt` |
| `shl-vib` | `vibration_psd.sbatch` | `vibration_psd_diagnostic.py` | either | ✅ done (H1) | `VIBRATION_DIAGNOSTIC.md` |
| `shl-vexpert` | `vehicle_expert.sbatch` | `vehicle_expert.py` | Athena | ✅ done (V4 DISABLE) | `VEHICLE_EXPERT_RESULTS.md` |
| _Tier 2 (gravity-canon, layer/pool)_ | _not built_ | — | Athena GPU | ⏳ planned | — |
| _new FMs (UniMTS/NormWear)_ | _not built_ | — | Athena GPU | ⏳ planned | — |

Done/banked earlier (results in the table below): `shl-split`, `shl-base`, `shl-vsweep`,
`shl-moe`, `shl-rail`, `shl-fusion`, `shl-freqmag`, `shl-embpca`, `shl-temporal`, `shl-diagnose`.

## Done — confirmed results
| # | experiment | result | why we did it |
|---|---|---|---|
| 1 | **Baseline pipeline** (520 handcrafted + FM embeddings → LGBM fusion) | v1 ≈ 0.80 | frozen-FM transport-mode recognition |
| 2 | **Split protocol / leakage audit** (blocked vs temporal+embargo) | blocked **inflates ~0.87** → honest **~0.80** (+0.07 gap) | avoid last year's 0.9→0.71 over-claim |
| 3 | **FM bake-off v2** (full 11-FM × temporal split, calibrated) | **utica_V2 = 0.8157** best (Δ+0.025 vs hc 0.7909); all 11 FMs help in fusion; emb-alone < hc | which FM to use (UTICA wins, not MOMENT) |
| 4 | **Per-class calibration** (macro-F1 multipliers on TUNE) | **~+0.16 macro** — single biggest lever, rescues Run | macro-F1 weights all classes equally |
| 5 | **Location Mixture-of-Experts** (Bag/Hips/Torso + router) | **≈0 gain** (negative) — routing per se is an ensemble effect | test location unknown |
| 6 | **Rail expert** (Train/Subway, magnetometer magnitude) | **DISABLE** | disambiguate the rail pair |
| 7 | **VehicleExpert V4** (gated subset corrector, mag-magnitude) | **DISABLE** — all 48 configs negative | vehicle classes = 52% of test |
| 8 | **Vibration diagnostic H1** (engine band 18–40 Hz) | **supported but modest** — Car 0.088 vs Walk 0.019; MI-class 0.44 ≫ placement 0.058 | can we hear the engine in acc? |
| 9 | **H2 rail magnetometer** (gravity-referenced, leave-one-bout-out) | **NEGATIVE** — LOBO 0.67 ≈ majority 0.60; cue is route- not mode-specific | bottom(metro)-vs-top(train) power hypothesis |
| 10 | **Tier-1 post-hoc** (calibration / logit-adj / MLLS prior) | **KEEP v1** — no post-hoc beats calibration on TUNE; **oracle-prior ceiling 0.8003 < v1 0.8029** → label-shift HURTS macro-F1. Run already ≈0.96 (calibrated); cap is rail. | squeeze macro-F1 via the metric-aligned decision rule |
| 11 | **Soft-voting head E-VOTE-01** (calibrated vote of utica_V2+mantisv2_V1) | **KEEP** — `vote:weighted+recal` TEST **0.8342** vs best single utica_V2 0.8213 (**+0.0129**, TUNE-selected). Per-class: Run+0.02, Car+0.03, Bus+0.05, **Train+0.03**, no material regression (Still −0.02). The SHL-2025 winning family pays off. | diversity-not-routing late fusion |

## Banked negatives — do NOT re-chase
Temporal smoothing (dead on shuffled test) · location-MoE · rail/magnetometer Train-Subway (H2)
· vehicle/engine disambiguation (modest) · **FM embeddings _alone_ (< handcrafted) — but in
_fusion_ all 11 FMs help (+0.003…+0.025); utica_V2 fusion 0.8157 is the new best, see #3**. The Train↔Subway
residual (Subway→Train = 38% of Subway) is likely irreducible here without a **barometer** (absent).

## In progress / built, not yet concluded
| experiment | what | status |
|---|---|---|
| **Per-channel extraction** (`extract_embeddings.py --per-channel`, job `shl-extract-pc`) | (n,C,d) V1 axes for utica+mantisv2 → cross-channel head input | **ready** (GH200); diff-tested |
| **Cross-channel head E-HEAD-01** (`head_xchannel.py`, job `shl-head`) | mean/concat/multistat/**se** heads over (n,C,d)⊕520; ECE+robustness; ≥3 seeds; novel = SE channel-mixing | **BUILT + pre-registered** 2026-06-26 (diff-tested); needs `_pc` extraction first; bar 0.8342 |
| **Rotation-TTA** (`tta_embeddings.py`) | mean FM embedding over K reorientations | built; ready to run after the head |

**E-VOTE-01 CONCLUDED 2026-06-26 → KEEP** (result row #11). ⚠️ **Traceability flag + ROOT CAUSE:**
single utica_V2 in the vote run = 0.8213 but `BAKEOFF_SPLIT.md` said 0.8157 (+0.0056, same recipe) —
the pre-registered "single reproduces bake-off" check FAILED. **Not stale features** (feature/FM-input
code unchanged since 5d77713, 2026-06-20 → regen is identical). **Cause: LightGBM multithread
non-determinism** (our configs used `subsample`/`colsample` + `n_jobs=-1`, no `deterministic=True` →
~±0.005 run-to-run). **FIXED:** added `deterministic=True, force_row_wise=True` to all 6 LGBM configs
(probe_fusion, submit_fusion, train_split, train_baseline, moe_experiment, diagnose_perclass) +
guardrail `test_lgbm_subsampling_is_deterministic`. Re-run the bake-off (utica_V2, mantisv2_V1) WITH
the deterministic config to lock canonical numbers. Vote Δ unaffected (all rows shared data). New best
overall = vote+recal **0.8342**.

## Infra / correctness work
- **MLflow traceability** mandated (track + artifact snapshot per experiment); split scheme + bare
  macro_f1 logged. — full provenance.
- **Test-loader bug** fixed (test = single shuffled `all.parquet`, not per-location). — submission crashed.
- **Submission versioning** documented (`SUBMISSIONS.md`). — provenance.
- **Tier-1 methodology fix**: source prior = model's *implied* prior (class_weight=balanced ≠ label
  freqs). — correct label-shift.
- **Helios GH200 (aarch64) extraction lane** stood up (Athena queue dead): hybrid-arch handling
  (x86 login/CPU vs aarch64 GH200), `setup_env_helios.sh` (fresh aarch64 resolve, not the x86 lock),
  aarch64 `uv` + torch 2.5.1/cu12.4 on Grace-Hopper (~1800 win/s), arch-aware `env_mdaniol.sh`,
  optional group `env.sh`. — use GH200 when Athena is busy.
- **extract_embeddings now MLflow-logged** (per model×variant: params + windows/sec + manifest);
  fixed a `del lib,X` runtime bug from the TTA refactor (caught by the Helios smoke test).
- **torch-gate fix (2026-06-26):** the new head/per-channel guardrail tests import torch, but the
  **x86 CPU venv has no torch** (it lives only in the aarch64 GH200 venv) → the gate failed and
  aborted `submit_vote`/`submit` on CPU. Fixed with `pytest.importorskip("torch")` so those tests
  **skip on CPU, run fully on GH200**. (CPU venv = `$SCRATCH/venvs/shl2026`; aarch64 = `…-gh200`.)
- **link_data storage-path fix (2026-06-26):** `link_data.sh` defaulted `EMB_SRC`/`FEAT_SRC` to
  `$SCRATCH` (purges / empty) → submit jobs linked an empty `embeddings/` → false "test__all.npy
  missing" (the file IS on group storage). Fixed defaults → **group storage** (single source of truth)
  + submit/head sbatch now export `RAW_SRC/FEAT_SRC/EMB_SRC` explicitly (mirrors the bake-off job).
- **determinism cost + force_col_wise speedup (2026-06-26):** `deterministic=True` made LightGBM fits
  **~3–4× slower** (non-det `shl-vote` = 2 fits <45 min; det `submit_vote` timed out at 2 h). Kept
  reproducibility; bumped submit_vote `--time`→6 h, submit→3.5 h, probe→12 h. **OPTIMIZED:** switched
  all 6 LGBM configs `force_row_wise`→**`force_col_wise`** — **verified bit-identical** (row vs col,
  `max prob diff 0.0`, predictions equal — both build the same histograms) and **~1.5× faster locally**
  (more on the 24-core cluster: col_wise parallelizes across our ~1k features). Zero change to any
  number; speeds up every fit (submit/probe/baseline). `test_lgbm_deterministic_config_reproduces` updated.
- **model-reuse for submissions (2026-06-26):** `voting_head.py` now snapshots its fitted models +
  locked vote params (`vote_models_*.joblib`, also an MLflow artifact — closes a §8 gap); `submit_vote.py
  --from-models <bundle>` **reuses** them → predicts test in **minutes, no refit**, bit-identical to the
  fit path (guarded by `test_vote_model_reuse_is_lossless`: joblib round-trip preserves `aligned_proba`).
  Fit path stays default; reuse is opt-in. **Skipped hc-only caching** — all 11 probe tasks start
  simultaneously so the cache isn't ready in time (would need a 2-step pre-compute); `force_col_wise`
  + the 2-task re-confirm already cover the probe cost.
- **MLflow file-store maintenance-mode (2026-06-27):** newer MLflow refuses the `file://` backend by
  default → recent runs (incl. submission v3) logged "MLflow unavailable; continuing without tracking"
  → **silently untracked** (a §8 gap). Fixed: `export MLFLOW_ALLOW_FILE_STORE=true` in `env_mdaniol.sh`
  (we deliberately use a group-storage file store, no DB server). Re-runs now log again.
- **Bake-off table NOT clean-deterministic yet (2026-06-27):** the deterministic re-run left a MIXED
  table — `utica_V2` shows a *third* value 0.8229 (vs 0.8157 orig, 0.8213 vote-run), and the array
  never finished 5 tasks (mantisv2_V1, mantis8m_V0/V1, utica_V0/V1). Ranking robust (utica_V2 top,
  mantisv2_V1 2nd) so the DECISION + shipped vote are unaffected, but **don't cite this table** — needs
  one clean full 12 h run (with MLflow now fixed) to be a paper artifact.

## Next (prioritized)
1. ✅ **Tier 1 done** → KEEP v1 (post-hoc tapped out; oracle ceiling < v1).
2. ✅ **FM bake-off done** → **utica_V2 = 0.8157 new best base** (all 11 FMs help in fusion).
3. **Soft-voting head (active)** — `voting_head.sbatch` (job `shl-vote`): calibrated vote of the
   top-2 (utica_V2+mantisv2_V1). The safe gain + the 2025-winning-family realization. Sets the bar
   the cross-channel head (LIGHTWEIGHT_HEAD_PLAN #5) must clear.
4. **Lightweight head** — cross-channel SE / per-channel attention (the *novel* contribution).
5. **Tier 2** — gravity-frame canonicalization + MOMENT layer-10 / concat-pooling.
6. **Gated FMs (CrossHAR / UniMTS)** — only if the head stalls (oneHAR/MASTER dropped).
7. **Paper** — leakage audit (0.87→0.80) + banked negatives + calibration-as-surviving-lever
   + frozen-FM-ensemble framing (HASCA rewards methodology/characterization).

## Cluster division of labor
- **Athena (x86, GPU/A100):** FM extraction — proven, but the queue can be dead.
- **Helios GH200 (aarch64, GPU):** FM extraction fallback/primary — `extract_all_helios.sbatch` + the
  aarch64 env (`setup_env_helios.sh` + `add_fm_deps_helios.sh`; aarch64 `uv`). ~1800 win/s, MLflow-logged.
- **Ares / Helios-CPU (x86, CPU):** everything on *cached* embeddings — bake-off, tier1, fusion,
  post-hoc. `setup_env.sh` + `uv pip install lightgbm`.
- **Storage:** group storage is per-cluster (Athena `pr2`, Helios `pr3` — separate); raw + embeddings
  live on group storage, moved by `rsync` / `stage_to_group.sh`. `$SCRATCH` purges (Helios: 30 d).
  Full Helios runbook: `hpc/HELIOS_SETUP.md`.

## Runbook — FM bake-off (current: Helios GH200 + CPU) [extraction running]
Goal: settle MantisV2 / Mantis8M / UTICA / MOMENT under temporal+embargo. Athena queue dead → Helios.
```
# --- Helios GH200 (aarch64): extract ALL FMs -> group storage, MLflow-logged ---
cd ~/shl2026 && git pull
sbatch notebooks/mdaniol/hpc/extract_all_helios.sbatch   # array: 1 model/task; ~30 min wall-clock

# --- Helios-CPU / Ares (x86): LightGBM bake-off over the staged embeddings ---
./scripts/setup_env.sh && source notebooks/mdaniol/hpc/env_mdaniol.sh && uv pip install lightgbm
export MLFLOW_TRACKING_URI="file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios"
sbatch notebooks/mdaniol/hpc/probe_fusion_ares.sbatch    # -> BAKEOFF_SPLIT.md (temporal, Δ-vs-handcrafted)
```
(Old Athena path still valid: `extract_fm.sbatch` + `stage_to_group.sh` + `probe_fusion_ares.sbatch`.)

## Diary
- **2026-06-25** — Tier-1 post-hoc ablation (Ares): **KEEP v1**, a clean negative — oracle-prior
  ceiling 0.8003 < v1 0.8029 (label-shift hurts a class-uniform metric); Run already ≈0.96, cap is
  the rail pair. Post-hoc lever closed (also rules out a Bayesian variant). Athena GPU queue went
  dead → stood up the **Helios GH200 (aarch64) extraction lane**: hybrid-arch env, aarch64 `uv` +
  torch 2.5.1/cu12.4 on Grace-Hopper (smoke ~1800 win/s), MLflow-logged extraction, group-storage
  outputs. FM bake-off (MantisV2/Mantis8M/UTICA vs MOMENT) now extracting on Helios while Athena sleeps.

## Runbook — soft-voting head (job `shl-vote`, registered 2026-06-26) [pending submit]
Goal: does a calibrated late-fusion vote of the bake-off top-2 beat the best single FM on the
lock TEST? Registered (committed), MLflow-tracked, gated KEEP iff Δ>+0.001. CPU on Helios.
```
cd ~/shl2026 && git pull                                   # pull the registered job + head
SHL-local/bin/python -m pytest tests/test_guardrails.py -q # pre-HPC gate (12/12 expected)
sbatch --test-only notebooks/mdaniol/hpc/voting_head.sbatch
JID=$(sbatch --parsable notebooks/mdaniol/hpc/voting_head.sbatch); echo "submitted $JID"
# --- monitor ---
watch -n 30 "squeue --me; echo ---; tail -n 8 notebooks/mdaniol/modeling/logs/vote_${JID}.out"
# --- on completion: results + MLflow ---
cat notebooks/mdaniol/VOTING_HEAD_RESULTS.md
source notebooks/mdaniol/hpc/env_mdaniol.sh
python -c "from shl2026 import leaderboard; print(leaderboard())" | head
```
