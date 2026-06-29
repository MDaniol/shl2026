#!/bin/bash
# E-COMBINE (CPU): leakage-clean soft-vote of our v4 champion ⊕ collaborator's frozen-DINoV2 pipeline.
# w tuned on (his target_holdout ∩ our validation[TEST]); blend hidden test -> v5. His 4 files staged at
# $PDUSZA_DIR (group storage). Reuses our cached embeddings + 520 features + env_mdaniol.sh.
#
#   sbatch notebooks/mdaniol/hpc/combine_pdusza_helios.sbatch
#
#SBATCH --job-name=shl-combine
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/combine_%j.out
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

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
[ -f "$SPLIT" ] || python "$M/split.py" --scheme temporal --embargo 100 \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --out "$SPLIT"
[ -f "$PROJECT_DIR/notebooks/mdaniol/preds_test_v4_vote.npy" ] || { echo "FATAL: our v4 test proba missing — run submit_vote v4 first" >&2; exit 1; }
[ -d "$PDUSZA_DIR/03_final_all_labelled_ensemble" ] || { echo "FATAL: $PDUSZA_DIR not staged (his 4 files)" >&2; exit 1; }

echo "[$(date)] E-COMBINE EMBS=$EMBS PDUSZA=$PDUSZA_DIR on $(hostname)"
HIS_FULLVAL="${HIS_FULLVAL:-$PDUSZA_DIR/03_SELECTED_ADAPTATION_full_official_validation_predictions.npz}"
python "$M/combine_pdusza.py" --embs "$EMBS" --split "$SPLIT" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --emb-root "$PROJECT_DIR/embeddings" \
    --pdusza-dir "$PDUSZA_DIR" \
    --our-test-proba "$PROJECT_DIR/notebooks/mdaniol/preds_test_v4_vote.npy" \
    $( [ -f "$HIS_FULLVAL" ] && echo "--his-fullval $HIS_FULLVAL" )
echo "[$(date)] done — see COMBINE_PDUSZA_RESULTS.md + AGH_predictions_v5_combine.txt + MLflow combine_pdusza_v5."
