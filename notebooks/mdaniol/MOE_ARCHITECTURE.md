# SHL 2026 — Location-aware MoE architecture, file map & versioning

Companion to `DATA_SPLITS.md` and the approved plan. Tracks **what is what and where**, the
branch/fusion flow, and the versioning convention. Test location is **unknown** → per-location
models are *experts* combined via router/soft-MoE, never hard-routed in the submission.

## Architecture (branches → fusion → rail → submission)

```mermaid
flowchart TD
    F["5s frame 9x500<br/>(Acc/Gyr/Mag)"] --> PRE["existing preprocessing<br/>feature_extraction/ + cached embeddings"]

    subgraph BR["Branches (probabilities per class)"]
      A["Branch A: 520 handcrafted -> LGBM<br/>train_baseline.py / train_split.py · EXISTS"]
      FM["Global FM heads<br/>UTICA/MOMENT/MantisV2 emb -> head<br/>probe_fusion.py · EXISTS"]
      FQ["Freq+Mag branch (linear)<br/>curated spectral+mag subset + entropy/centroid/spike<br/>freq_mag_branch.py · NEW (reuses shl_features)"]
      EX["Per-location experts (Bag/Hips/Torso)<br/>probe_fusion.py --per-location · NEW loop, reuses fit_cal_eval"]
      RT["Location router q(Bag,Hips,Torso)<br/>probe_fusion.py --router · NEW, reuses fit_cal_eval"]
    end

    PRE --> A & FM & FQ & EX
    PRE --> RT
    EX --> MOE["Soft MoE: P_moe = q·P_experts<br/>moe_combine.py · NEW (pure numpy)"]
    RT --> MOE
    A & FM & FQ & MOE --> FUS["Fusion: weighted avg / logit stacker<br/>moe_combine.py + probe_fusion.py · NEW"]
    FUS --> GF["Global fallback<br/>P_final = β·P_global + (1-β)·P_moe<br/>β∈{.25,.40,.50,.70} + adaptive"]
    GF --> RAIL{"Rail gate:<br/>P(train)+P(subway)>τ_rail<br/>& conf>τ_conf?"}
    RAIL -->|yes| RE["Train-vs-Subway expert<br/>rail_expert.py · NEW (mag features)"]
    RAIL -->|no| OUT["8-class label"]
    RE --> OUT
    OUT --> SUB["Submission 92726x500<br/>predict_submit.py + validate.py · EXISTS (extend)"]

    classDef exists fill:#d4f4dd,stroke:#2a7;
    classDef new fill:#fde0c0,stroke:#e80;
    class A,FM,SUB exists;
    class FQ,EX,RT,MOE,FUS,RE new;
```

## Decision gates (evidence-based; lock-test = held-out TEST)
- **G1 oracle:** per-location oracle gain over global is *large*? else experts = analysis-only.
- **G2 router:** router reliable enough that soft-MoE approaches oracle?
- **G3 submission:** location-aware / rail / any branch ships **only if it beats the global
  baseline on lock-test (TEST)**, not the selection split (TUNE).

## File map (extend, don't rewrite)
| Component | File | Status |
|---|---|---|
| Split (FIT/TUNE/TEST) | `modeling/split.py`, `artifacts/val_split.npy` | EXISTS |
| Branch A (520 feat LGBM) | `modeling/train_baseline.py`, `train_split.py` | EXISTS |
| Global FM heads + fusion | `modeling/probe_fusion.py` | EXISTS |
| Metrics / confusion | `modeling/metrics.py` (`class_report`) | EXISTS |
| Submission + validate | `modeling/predict_submit.py`, `src/shl2026/submission/validate.py` | EXISTS |
| Freq+Mag linear branch | `modeling/freq_mag_branch.py` | NEW |
| Per-location experts + router | `modeling/probe_fusion.py` (`--per-location`, `--router`, `--rep`) | NEW |
| MoE / β-fallback / fusion combiner | `modeling/moe_combine.py` | NEW |
| Rail expert | `modeling/rail_expert.py` | NEW |
| Subject column (leakage check) | `feature_extraction/extract_features.py` | NEW |
| Result tables | `BAKEOFF_SPLIT.md`, `MOE_RESULTS.md` | EXISTS / NEW |

## MLflow logging — use the EXISTING team API (`shl2026.track`)
Do NOT roll our own MLflow wrapper. The team package already provides it:
`from shl2026 import track, evaluate_predictions` (impl `src/shl2026/tracking/autolog.py`).
`track()` reads `MLFLOW_TRACKING_URI` (group `env.sh` → `env_mdaniol.sh`), opens one run,
auto-stamps git SHA/dirty, container, Slurm job, node, grant, student, python_env, logs a
code_snapshot if dirty, and **no-ops gracefully** if MLflow is unreachable. Server =
`scripts/mlflow_server.sh` (sqlite+proxied artifacts), client+server mlflow 3.13.

Convention for our runs:
- **Experiment = `"mdaniol"`** (student-name convention; shows on the team `leaderboard()`).
- **`run_name = <config>`** (e.g. `global+softmoe_b0.40`), **`seed=0`** (track seeds Py/NumPy/Torch),
  **`params_path=None`** (our scripts use argparse, not the root `params.yaml`).
- **params:** rep, model/variant, location, β, τ_rail/τ_conf, PCA dim, n_estimators.
- **tags:** `phase` (baseline/freqmag/emb/oracle/router/moe/fusion/rail), `branch`.
- **metrics:** score with `evaluate_predictions(y_true, y_pred)` → `EvalResult`, then
  `run.log_eval(result, prefix="test_")` / `"tune_"` (logs macro_f1 + per-class incl.
  Train/Subway); add `run.log_metrics({"selection_lock_gap": ...})`.
- **artifacts:** `run.log_artifact(<per-config json>)`.
Pattern:
```python
from shl2026 import track, evaluate_predictions
with track("mdaniol", run_name=cfg, seed=0, params_path=None,
           params={...}, tags={"phase": "moe", "branch": "fusion"}) as run:
    run.log_eval(evaluate_predictions(y_test, pred_test), prefix="test_")
    run.log_metrics({"selection_lock_gap": gap})
```
The JSON/markdown tables stay as the offline record alongside MLflow.

## Versioning convention
- **Code:** git, one commit per step; message `feat(moe): <step> — <what/why>`.
- **Results:** append-only `MOE_RESULTS.md` (git-tracked) + per-config JSON `artifacts/<config>.json`
  via `save_report`. Never overwrite a prior config's row.
- **Models:** `artifacts/<branch>_<rep>_<config>.joblib` (e.g. `expert_Bag_handcrafted.joblib`).
- **Submissions:** `AGH_predictions_v<N>_<config>.txt` (e.g. `v4_global+softmoe_b0.40`) + a
  `SUBMISSIONS.md` log row: version, date, config, held-out macro-F1, file, git SHA. The chosen
  final is copied to `AGH_predictions.txt`.
- **Determinism:** fixed seeds everywhere; `split.py` already RNG-free; record git SHA + `val_split.json` counts with every result.

## Changelog
- v0 — global Branch A + FM split bake-off (baseline, current run).
- (next) v1 — freq+mag branch · v2 — emb L2/PCA heads · v3 — per-location oracle · v4 — router+MoE · v5 — rail expert.
