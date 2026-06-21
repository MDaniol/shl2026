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
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # single env source: group env + my venv + HF/uv/tmp caches
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"   # ensure data symlinks
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling

python $M/train_baseline.py     --feat-dir "$PROJECT_DIR/dataset_parquet_features"
python $M/threshold_calibrate.py --feat-dir "$PROJECT_DIR/dataset_parquet_features"
python $M/predict_submit.py      --feat-dir "$PROJECT_DIR/dataset_parquet_features"
echo "[$(date)] baseline + calibration + submission done"
