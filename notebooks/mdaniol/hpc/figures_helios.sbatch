#!/bin/bash
# PAPER FIGURES (CPU): re-run E-COMBINE with --dump-eval to snapshot the HONEST labelled arrays, then
# render every paper figure (confusion matrices, per-class F1 bars, blend-weight sweep) from them.
# Reuses env_mdaniol.sh + the cached embeddings/features + his staged results. Idempotent: it also
# rewrites the identical v5 submission (deterministic) — harmless.
#
#   sbatch notebooks/mdaniol/hpc/figures_helios.sbatch
#
#SBATCH --job-name=shl-figures
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/figures_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
EMBS="${EMBS:-utica_V2,mantisv2_V1}"
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios}"
SPLIT="$PROJECT_DIR/notebooks/mdaniol/modeling/artifacts/val_split_temporal.npy"
PDUSZA_DIR="${PDUSZA_DIR:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/pdusza_results}"
M="$PROJECT_DIR/notebooks/mdaniol/modeling"
FIGDIR="$PROJECT_DIR/notebooks/mdaniol/figures"
DUMP="$FIGDIR/combine_eval_dump.npz"
HIS_HOLDOUT="$PDUSZA_DIR/02_adaptation_selection/selected_untouched_target_holdout/predictions.npz"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs "$FIGDIR"

# pre-HPC gate (same as combine)
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
[ -f "$SPLIT" ] || python "$M/split.py" --scheme temporal --embargo 100 \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --out "$SPLIT"
[ -f "$PROJECT_DIR/notebooks/mdaniol/preds_test_v4_vote.npy" ] || { echo "FATAL: our v4 test proba missing — run submit_vote v4 first" >&2; exit 1; }
[ -d "$PDUSZA_DIR/03_final_all_labelled_ensemble" ] || { echo "FATAL: $PDUSZA_DIR not staged (his files)" >&2; exit 1; }
[ -f "$HIS_HOLDOUT" ] || echo "WARN: $HIS_HOLDOUT missing — Lane B holdout CM will be skipped"

echo "[$(date)] STEP 1/2 E-COMBINE --dump-eval -> $DUMP"
python "$M/combine_pdusza.py" --embs "$EMBS" --split "$SPLIT" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --emb-root "$PROJECT_DIR/embeddings" \
    --pdusza-dir "$PDUSZA_DIR" \
    --our-test-proba "$PROJECT_DIR/notebooks/mdaniol/preds_test_v4_vote.npy" \
    --dump-eval "$DUMP" || { echo "combine failed" >&2; exit 1; }

echo "[$(date)] STEP 2/2 rendering figures -> $FIGDIR"
python "$FIGDIR/make_paper_figures.py" --dump "$DUMP" \
    $( [ -f "$HIS_HOLDOUT" ] && echo "--his-holdout $HIS_HOLDOUT" ) \
    --out-dir "$FIGDIR" --mlflow || { echo "figure rendering failed" >&2; exit 1; }

echo "[$(date)] done — PNGs + $DUMP in $FIGDIR:"
ls -la "$FIGDIR"/*.png
