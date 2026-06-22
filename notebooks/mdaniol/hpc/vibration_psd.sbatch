#!/bin/bash
# Athena job: Phase-1 vibration diagnostic (VEHICLE_EXPERIMENT_PLAN.md).
# Training-free, leakage-free per-class Welch PSD on raw acc + raw-axis magnetometer to GATE
# the engine-band (H1) and raw-mag rail (H2) feature work. CPU-bound; reads raw dataset_parquet.
#
#   sbatch notebooks/mdaniol/hpc/vibration_psd.sbatch
#
#SBATCH --job-name=shl-vib
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/vib_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1            # Athena is GPU-only (this job is CPU-bound)
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

echo "[$(date)] vibration diagnostic on $(hostname)"
python notebooks/mdaniol/modeling/vibration_psd_diagnostic.py \
    --data-dir "$PROJECT_DIR/dataset_parquet" \
    --locs "${LOCS:-Bag,Hips,Torso}" --max-per-class-loc "${MAXPC:-1500}"
echo "[$(date)] done — see VIBRATION_DIAGNOSTIC.md / .json + MLflow (vibration_psd_diagnostic)"
