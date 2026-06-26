# SHL 2026 — experiment log (progress, results, why)

Concise running log. Detail lives in `WINNING_STRATEGY.md`, `VEHICLE_EXPERIMENT_PLAN.md`,
`DATA_SPLITS.md`, and the per-experiment `*_RESULTS.md` + MLflow. Metric = macro-F1, honest
temporal+embargo split, Bag/Hips/Torso eval, per-window (test is shuffled → no smoothing).

## Headline status
- **New best base (bake-off 2026-06-26):** frozen **UTICA (V2)** embeddings ⊕ 520 handcrafted →
  LightGBM fusion + per-class calibration = **0.8157 macro-F1** (temporal lock). Beats the old v1
  (**MOMENT-small_V1 = 0.8060**) by **+0.010** — a free upgrade from the full 11-FM bake-off.
- **All 11 FMs add positive Δ vs handcrafted-only (0.7909)** (+0.003…+0.025); embeddings *alone*
  lose to handcrafted → the FM value is in **fusion, not replacement** (confirms the survey's
  reconstruction-pretraining prior). Top-2 for soft-voting = **utica_V2 + mantisv2_V1** (0.8138).
- **Realistic ceiling ~80s** (smoothing is dead on a shuffled test), so ~0.82 is competitive.
- Tier-1 post-hoc is **tapped out** (KEEP v1; oracle ceiling < v1). Remaining edge = the
  **lightweight head** (soft-voting top-2, then cross-channel) + rigor for the paper.
- **Survey FM-hunt closed:** oneHAR (not in companion repo) and MASTER (multimodal, *no released
  weights*, train-from-scratch → not a frozen FM) are **dropped**. Only **CrossHAR / UniMTS**
  survive verification, kept as **gated** integration bets (not started unless the head stalls).

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
| `shl-tta` | `tta_embeddings.sbatch` | `extract_embeddings.py --tta-k` | Athena/Helios GPU | 🔨 ready | `embeddings/..._tta*/` |
| `shl-submit` | `submit.sbatch` | `submit_fusion.py` | Athena | ✅ v1 done | `AGH_predictions_v1_*.txt` |
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

## Banked negatives — do NOT re-chase
Temporal smoothing (dead on shuffled test) · location-MoE · rail/magnetometer Train-Subway (H2)
· vehicle/engine disambiguation (modest) · **FM embeddings _alone_ (< handcrafted) — but in
_fusion_ all 11 FMs help (+0.003…+0.025); utica_V2 fusion 0.8157 is the new best, see #3**. The Train↔Subway
residual (Subway→Train = 38% of Subway) is likely irreducible here without a **barometer** (absent).

## In progress / built, not yet concluded
| experiment | what | status |
|---|---|---|
| **Soft-voting head** (`voting_head.py` → `voting_head.sbatch`, job `shl-vote`) | calibrated late-fusion vote of top-2 (utica_V2+mantisv2_V1): equal / TUNE-weighted / +recal; gated KEEP iff > best single +0.001 on lock TEST. MLflow-tracked (run `voting_utica_V2+mantisv2_V1`), artifact-snapshotted, diff-tested (`test_voting_head_combiner_math`). | **registered 2026-06-26** (committed, gate green 12/12); pending submit on Helios CPU |
| **Rotation-TTA** (`tta_embeddings.py`) | mean FM embedding over K reorientations | built; ready to run after the head |

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
