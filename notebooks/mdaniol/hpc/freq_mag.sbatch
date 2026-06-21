#!/bin/bash
# Athena job: Step 2 — frequency + magnetometer branch (linear heads on the
# controlled freq+mag subset of the existing 520 features). CPU, reuses cached
# features (no re-extraction). Writes FREQMAG_RESULTS.md + MLflow.
#
#SBATCH --job-name=shl-freqmag
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/freqmag_%j.out
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

echo "[$(date)] frequency + magnetometer branch"
python $M/freq_mag_branch.py --feat-dir "$FEAT"
echo "[$(date)] done — see notebooks/mdaniol/FREQMAG_RESULTS.md + MLflow"
