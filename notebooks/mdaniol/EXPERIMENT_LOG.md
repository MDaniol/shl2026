# SHL 2026 — experiment log (progress, results, why)

Concise running log. Detail lives in `WINNING_STRATEGY.md`, `VEHICLE_EXPERIMENT_PLAN.md`,
`DATA_SPLITS.md`, and the per-experiment `*_RESULTS.md` + MLflow. Metric = macro-F1, honest
temporal+embargo split, Bag/Hips/Torso eval, per-window (test is shuffled → no smoothing).

## Headline status
- **Best deployable (v1):** frozen **MOMENT-small (V1)** embeddings ⊕ 520 handcrafted features →
  LightGBM fusion → per-class calibration ≈ **0.80 macro-F1** (honest; bracket 0.725–0.803).
- **Realistic ceiling ~80s** (smoothing is dead on a shuffled test), so 0.80 is competitive.
- Edge to chase = **metric-aligned post-hoc (Tier 1) + rigor**, not a flashier model.

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
