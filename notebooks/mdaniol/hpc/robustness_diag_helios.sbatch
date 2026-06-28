#!/bin/bash
# Confusion-structure + generalizability diagnostics (CPU): where the top-2 oracle gap lives +
# per-location robustness (Bag/Hips/Torso) + prior-sensitivity. Analysis-only on locked TEST.
#
#   EMBS=utica_V2,mantisv2_V1 sbatch notebooks/mdaniol/hpc/robustness_diag_helios.sbatch
#
#SBATCH --job-name=shl-robust
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/robust_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
EMBS="${EMBS:-utica_V2,mantisv2_V1}"
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

echo "[$(date)] robustness diagnostics EMBS=$EMBS on $(hostname)"
python "$M/robustness_diag.py" --embs "$EMBS" --split "$SPLIT" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --emb-root "$PROJECT_DIR/embeddings"
echo "[$(date)] done — see ROBUSTNESS_DIAG.md + MLflow run robust_${EMBS}."
