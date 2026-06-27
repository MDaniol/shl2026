#!/bin/bash
# CPU job: Phase-0 subject-leakage diagnostic (why blocked/random splits inflate).
# Pure CPU (LightGBM domain classifiers on the 520 handcrafted features). Helios plgrid.
#
#   sbatch notebooks/mdaniol/hpc/subject_leakage.sbatch
#
#SBATCH --job-name=shl-subjleak
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/subjleak_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
# stage from GROUP storage (persistent); $SCRATCH purges
export RAW_SRC="${RAW_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet}"
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

echo "[$(date)] subject-leakage diagnostic on $(hostname)"
python notebooks/mdaniol/modeling/subject_leakage.py \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features"
echo "[$(date)] done -> SUBJECT_LEAKAGE.md"
