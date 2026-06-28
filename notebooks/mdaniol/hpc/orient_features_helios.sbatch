#!/bin/bash
# Orientation-invariant features vs the Torso placement fragility (CPU): does emb⊕520⊕orient beat
# emb⊕520 on the honest TEST, especially on Torso? Reads RAW dataset_parquet (streamed per location).
#
#   EMB=utica_V2 sbatch notebooks/mdaniol/hpc/orient_features_helios.sbatch
#
#SBATCH --job-name=shl-orient
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/orient_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
EMB="${EMB:-utica_V2}"
export RAW_SRC="${RAW_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet}"
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios}"
SPLIT="$PROJECT_DIR/notebooks/mdaniol/modeling/artifacts/val_split_temporal.npy"
M="$PROJECT_DIR/notebooks/mdaniol/modeling"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
[ -f "$SPLIT" ] || python "$M/split.py" --scheme temporal --embargo 100 \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --out "$SPLIT"
[ -d "$PROJECT_DIR/dataset_parquet" ] || { echo "FATAL: dataset_parquet (raw) not linked" >&2; exit 1; }

echo "[$(date)] orient-feature ablation EMB=$EMB on $(hostname)"
python "$M/orient_features.py" --emb "$EMB" --split "$SPLIT" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --emb-root "$PROJECT_DIR/embeddings" \
    --data-dir "$PROJECT_DIR/dataset_parquet"
echo "[$(date)] done — see ORIENT_FEATURES_RESULTS.md + MLflow run orient_${EMB}."
