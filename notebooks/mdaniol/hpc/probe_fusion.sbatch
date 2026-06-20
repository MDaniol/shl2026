#!/bin/bash
# Athena CPU array job: probe + fusion per cached embedding set (one per task).
# CPU-only (LightGBM / logreg); big RAM for the 784k x 2304 train embeddings.
#
#SBATCH --job-name=shl-probe
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/probe_%A_%a.out
## --- EDIT for your allocation (CPU grant) ---
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1   # Athena is GPU-only
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026}"
VENV="${VENV:-$PROJECT_DIR/SHL-local}"

EMBS=(mantisv2_V0 utica_V0 mantis8m_V0 moment-small_V0)
E="${EMBS[$SLURM_ARRAY_TASK_ID]}"

source "$VENV/bin/activate"
cd "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
echo "[$(date)] probe $E on $(hostname)"
python notebooks/mdaniol/modeling/probe_fusion.py --emb "$E" \
    --emb-root "$PROJECT_DIR/embeddings" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features"
echo "[$(date)] done $E"
