#!/bin/bash
# SLURM array job: one task per (split, location) Parquet → feature Parquet.
# Feature extraction is CPU-only — request a CPU grant/partition, not GPU.
#
#SBATCH --job-name=shl-feat
#SBATCH --array=0-8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/feat_%A_%a.out
#SBATCH --error=logs/feat_%A_%a.err
## --- EDIT for your allocation (Cyfronet Athena CPU grant) ---
#SBATCH --account=plgshl26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1   # Athena is GPU-only

set -euo pipefail

# --- paths (override via env, e.g. `PROJECT_DIR=$SCRATCH/SHL_2026 sbatch ...`) ---
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/dataset_parquet}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/dataset_parquet_features}"
SCRIPT_DIR="$PROJECT_DIR/notebooks/mdaniol/feature_extraction"

# --- index → input file (9 tasks: 4 train + 4 validation + 1 test) ---
INPUTS=( train/Bag train/Hips train/Torso train/Hand \
         validation/Bag validation/Hips validation/Torso validation/Hand \
         test/all )
REL="${INPUTS[$SLURM_ARRAY_TASK_ID]}"
SPLIT="${REL%%/*}"
LOC="${REL##*/}"

source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"   # group paths + UV_CACHE_DIR
source "${SHL_VENV:-$SCRATCH/venvs/shl2026}/bin/activate" # YOUR env (team lock + FM add-ons), overrides env.sh's group venv
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$OUT_DIR/$SPLIT" "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

echo "[task $SLURM_ARRAY_TASK_ID] $REL  ($(date))"
python extract_features.py \
    --input  "$DATA_DIR/$REL.parquet" \
    --output "$OUT_DIR/$REL.parquet" \
    --split "$SPLIT" --location "$LOC" \
    --chunk-size 20000
echo "[task $SLURM_ARRAY_TASK_ID] done ($(date))"
