#!/bin/bash
# Athena job: the 60/20/20 validation-split experiment + robustness diagnostic.
#   split.py            -> blocked 60/20/20 of validation -> artifacts/val_split.npy
#   train_split.py      -> Phase A honest held-out macro-F1 (per-class + macro)
#                          Phase B submission (User1 + ALL validation)
#   robustness_sweep.py -> rotation/noise/time-warp severity on the held-out TEST
#                          slice (rotation curve must be FLAT = Branch-A invariance)
# Needs the handcrafted features (extract_features.sbatch) to exist first.
#
#SBATCH --job-name=shl-split
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/split_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1            # Athena is GPU-only
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026}"
source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"   # group paths + UV_CACHE_DIR
source "${SHL_VENV:-$SCRATCH/venvs/shl2026}/bin/activate" # YOUR env (team lock + FM add-ons)
cd "$PROJECT_DIR"; mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling
FEAT="$PROJECT_DIR/dataset_parquet_features"

echo "[$(date)] build 60/20/20 split"
python $M/split.py --feat-dir "$FEAT"

echo "[$(date)] Phase A held-out estimate + Phase B submission"
python $M/train_split.py --feat-dir "$FEAT"

echo "[$(date)] robustness sweep on held-out TEST slice"
python $M/robustness_sweep.py \
    --raw-dir "$PROJECT_DIR/dataset_parquet/validation" \
    --split   "$M/artifacts/val_split.npy" \
    --model   "$M/artifacts/split_lgbm.joblib"

echo "[$(date)] done — see $M/artifacts/{split_results.json,robustness.json}"
