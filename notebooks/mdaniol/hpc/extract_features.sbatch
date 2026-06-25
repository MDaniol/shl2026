#!/bin/bash
# CPU job: build the 520 handcrafted feature parquets from raw dataset_parquet -> group storage.
# x86 CPU partition (numpy/scipy/FFT; no GPU). Reads raw + writes features on GROUP storage so the
# bake-off (probe_fusion_ares) finds them. Helios CPU = plgrid/plgshl26-cpu (also works on Ares).
#
#   sbatch notebooks/mdaniol/hpc/extract_features.sbatch
#   (then: sbatch probe_fusion_ares.sbatch for the bake-off)
#
#SBATCH --job-name=shl-feat
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/feat_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # x86 core env (numpy/scipy/pandas/pyarrow)
cd "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios}"
DATA="${RAW_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet}"
FEAT="${FEAT_DST:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
FE=notebooks/mdaniol/feature_extraction/extract_features.py
mkdir -p "$FEAT/train" "$FEAT/validation" "$FEAT/test"

echo "[$(date)] features: $DATA -> $FEAT"
for loc in Bag Hips Torso Hand; do
  for split in train validation; do
    out="$FEAT/$split/$loc.parquet"
    [ -f "$out" ] && { echo "  skip $split/$loc (exists)"; continue; }
    python "$FE" --input "$DATA/$split/$loc.parquet" --output "$out" --split "$split" --location "$loc"
  done
done
[ -f "$FEAT/test/all.parquet" ] || \
  python "$FE" --input "$DATA/test/all.parquet" --output "$FEAT/test/all.parquet" --split test --location all
echo "[$(date)] done. Next: sbatch notebooks/mdaniol/hpc/probe_fusion_ares.sbatch"
