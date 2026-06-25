# SHL 2026 — experiment log (progress, results, why)

Concise running log. Detail lives in `WINNING_STRATEGY.md`, `VEHICLE_EXPERIMENT_PLAN.md`,
`DATA_SPLITS.md`, and the per-experiment `*_RESULTS.md` + MLflow. Metric = macro-F1, honest
temporal+embargo split, Bag/Hips/Torso eval, per-window (test is shuffled → no smoothing).

## Headline status
- **Best deployable (v1):** frozen **MOMENT-small (V1)** embeddings ⊕ 520 handcrafted features →
  LightGBM fusion → per-class calibration ≈ **0.80 macro-F1** (honest; bracket 0.725–0.803).
- **Realistic ceiling ~80s** (smoothing is dead on a shuffled test), so 0.80 is competitive.
- Edge to chase = **metric-aligned post-hoc (Tier 1) + rigor**, not a flashier model.

## Job tracker (sbatch / driver / status)
`squeue --me` shows the **job-name**. Athena = GPU, Ares = CPU. Helpers (not jobs):
`env_mdaniol.sh`, `link_data.sh`, `stage_to_group.sh`, `setup_env.sh`, `add_fm_deps.sh` (Athena only).

| job-name | sbatch (Athena / Ares) | driver | cluster | status | output |
|---|---|---|---|---|---|
| `shl-tier1` | `tier1.sbatch` / `tier1_ares.sbatch` / `tier1_helios.sbatch` | `tier1_experiment.py` | Ares/Helios CPU | ✅ done — **KEEP v1** (post-hoc tapped out) | `TIER1_RESULTS.md` |
| `shl-probe` | `probe_fusion.sbatch` / `probe_fusion_ares.sbatch` | `probe_fusion.py` | Ares CPU | ⏳ pending (stage embeds first) | `BAKEOFF_SPLIT.md` |
| `shl-extract` | `extract_fm.sbatch` | `extract_embeddings.py` | Athena GPU | ⏳ pending (Mantis8M) | `embeddings/<fm>_<var>/` |
| `shl-tta` | `tta_embeddings.sbatch` | `extract_embeddings.py --tta-k` | Athena GPU | 🔨 ready | `embeddings/..._tta*/` |
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
| 3 | **FM bake-off** (MantisV2/UTICA/MOMENT × V0/V1/V2) | MantisV2 **0.70 < 0.745** handcrafted; **MOMENT-small_V1 = only FM that helps** | which FM to use |
| 4 | **Per-class calibration** (macro-F1 multipliers on TUNE) | **~+0.16 macro** — single biggest lever, rescues Run | macro-F1 weights all classes equally |
| 5 | **Location Mixture-of-Experts** (Bag/Hips/Torso + router) | **≈0 gain** (negative) — routing per se is an ensemble effect | test location unknown |
| 6 | **Rail expert** (Train/Subway, magnetometer magnitude) | **DISABLE** | disambiguate the rail pair |
| 7 | **VehicleExpert V4** (gated subset corrector, mag-magnitude) | **DISABLE** — all 48 configs negative | vehicle classes = 52% of test |
| 8 | **Vibration diagnostic H1** (engine band 18–40 Hz) | **supported but modest** — Car 0.088 vs Walk 0.019; MI-class 0.44 ≫ placement 0.058 | can we hear the engine in acc? |
| 9 | **H2 rail magnetometer** (gravity-referenced, leave-one-bout-out) | **NEGATIVE** — LOBO 0.67 ≈ majority 0.60; cue is route- not mode-specific | bottom(metro)-vs-top(train) power hypothesis |
| 10 | **Tier-1 post-hoc** (calibration / logit-adj / MLLS prior) | **KEEP v1** — no post-hoc beats calibration on TUNE; **oracle-prior ceiling 0.8003 < v1 0.8029** → label-shift HURTS macro-F1. Run already ≈0.96 (calibrated); cap is rail. | squeeze macro-F1 via the metric-aligned decision rule |

## Banked negatives — do NOT re-chase
Temporal smoothing (dead on shuffled test) · location-MoE · rail/magnetometer Train-Subway (H2)
· vehicle/engine disambiguation (modest) · MantisV2/UTICA alone (< handcrafted). The Train↔Subway
residual (Subway→Train = 38% of Subway) is likely irreducible here without a **barometer** (absent).

## Built, validated, not yet run
| experiment | what | why |
|---|---|---|
| **Tier 1 post-hoc** (`tier1_experiment.py`) | calibration / logit-adjust / **MLLS test-prior estimation** → per-class F1 thresholds | the differentiated edge; targets Run (weak from scarcity) |
| **Rotation-TTA** (`tta_embeddings.py`) | mean FM embedding over K reorientations | robustness to unknown test orientation |
| **Temporal FM bake-off** (`probe_fusion.sbatch`) | all FMs under the honest split + MLflow | settle FM ranking, gate cross-FM ensembling |

## Infra / correctness work
- **MLflow traceability** mandated (track + artifact snapshot per experiment); split scheme + bare
  macro_f1 logged. — full provenance.
- **Test-loader bug** fixed (test = single shuffled `all.parquet`, not per-location). — submission crashed.
- **Submission versioning** documented (`SUBMISSIONS.md`). — provenance.
- **Tier-1 methodology fix**: source prior = model's *implied* prior (class_weight=balanced ≠ label
  freqs). — correct label-shift.

## Next (prioritized)
1. **Run Tier 1** → read `TIER1_RESULTS.md` (does MLLS/logit-adjust beat v1 + lift Run; oracle ceiling).
2. **Tier 2** — gravity-frame canonicalization + MOMENT layer-10 / concat-pooling (one GPU re-extract).
3. **Re-check all FMs** on the temporal split with upgraded features (add spectral centroid/entropy/
   flatness) → definitive `BAKEOFF_SPLIT.md`.
4. **Paper** — leakage audit (0.87→0.80) + banked negatives + calibration-as-surviving-lever
   (HASCA rewards methodology/characterization).

## Cluster division of labor
- **Athena (GPU):** FM embedding *extraction* only (runs the frozen FM) — `extract_fm.sbatch`,
  `tta_embeddings.sbatch`, `variant_sweep.sbatch`.
- **Ares (CPU):** everything that consumes *cached* embeddings — `tier1_ares.sbatch`,
  `probe_fusion_ares.sbatch`, fusion, post-hoc. Slim venv (no torch): `setup_env.sh` + `uv pip
  install lightgbm`. Data reaches Ares via **group storage** (Athena `$SCRATCH` is invisible to Ares).

## Runbook — Tier 1 (CPU, Ares) [running]
```
# Ares: env = setup_env.sh + lightgbm; data staged to group storage; then:
sbatch notebooks/mdaniol/hpc/tier1_ares.sbatch   # -> TIER1_RESULTS.md (read cal(v1) vs logit_adj/mlls; Run F1; oracle ceiling)
```

## Runbook — temporal FM bake-off (re-check ALL FMs on the honest split) [pending]
Goal: settle MantisV2 / Mantis8M / UTICA / MOMENT under temporal+embargo with current features.
```
# --- on ATHENA (GPU): extract the missing FM + publish cached ones to group storage ---
cd ~/shl2026 && git pull
MODEL=mantis8m VARIANTS="V0 V1 V2" sbatch notebooks/mdaniol/hpc/extract_fm.sbatch   # the only un-extracted family
bash notebooks/mdaniol/hpc/stage_to_group.sh        # rsync features + ALL cached embeddings -> group storage
#   (re-run stage_to_group.sh after mantis8m finishes to publish it)

# --- on ARES (CPU): the heavy LightGBM bake-off over whatever is staged (missing FMs skip) ---
cd ~/shl2026 && git pull
sbatch notebooks/mdaniol/hpc/probe_fusion_ares.sbatch   # -> BAKEOFF_SPLIT.md (temporal, Δ-vs-handcrafted) + MLflow
```
Priority if GPU/time tight: UTICA + MantisV2 V1/V2 (UTICA loads the Mantis8M arch anyway); plain
Mantis8M is completeness-only (MantisV2 already beat it conceptually). New FMs (UniMTS/NormWear)
are a separate integration step — the novelty/upside play, not part of this bake-off.
