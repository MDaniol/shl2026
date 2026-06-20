#!/bin/bash
# Athena CPU job: handcrafted Track-A baseline → calibrate → submission.
#
#SBATCH --job-name=shl-base
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/baseline_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1   # Athena is GPU-only
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026}"
VENV="${VENV:-$PROJECT_DIR/SHL-local}"
source "$VENV/bin/activate"
cd "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling

python $M/train_baseline.py     --feat-dir "$PROJECT_DIR/dataset_parquet_features"
python $M/threshold_calibrate.py --feat-dir "$PROJECT_DIR/dataset_parquet_features"
python $M/predict_submit.py      --feat-dir "$PROJECT_DIR/dataset_parquet_features"
echo "[$(date)] baseline + calibration + submission done"
