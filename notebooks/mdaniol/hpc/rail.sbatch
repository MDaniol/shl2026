#!/bin/bash
# Athena job: Step 8 — global Train-vs-Subway rail expert over the global base.
# CPU; reuses cached features. Writes RAIL_RESULTS.md + MLflow. Gated correction
# kept only if it beats the base macro-F1 (printed decision).
#
#SBATCH --job-name=shl-rail
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/rail_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1            # Athena is GPU-only (job is CPU-bound)
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling
FEAT="$PROJECT_DIR/dataset_parquet_features"
[ -f "$M/artifacts/val_split.npy" ] || python $M/split.py --feat-dir "$FEAT"

echo "[$(date)] global Train-vs-Subway rail expert"
python $M/rail_expert.py --feat-dir "$FEAT"
echo "[$(date)] done — see notebooks/mdaniol/RAIL_RESULTS.md + MLflow"
