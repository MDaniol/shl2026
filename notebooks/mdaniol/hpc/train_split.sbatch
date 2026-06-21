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
#SBATCH --cpus-per-task=16      # per-GPU fair share; ~2x faster LightGBM, still backfills
#SBATCH --mem=32G
#SBATCH --time=01:00:00         # headroom for 3 multiclass LightGBM fits + robustness sweep
#SBATCH --output=notebooks/mdaniol/modeling/logs/split_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1            # Athena is GPU-only (job doesn't use the GPU)
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # single env source: group env + my venv + HF/uv/tmp caches
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"   # ensure data symlinks
mkdir -p notebooks/mdaniol/modeling/logs
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
