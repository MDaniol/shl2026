#!/bin/bash
# Helios GH200 (aarch64): PER-CHANNEL extraction for the cross-channel head.
# Keeps the 9 V1 axes SEPARATE -> (n, C, d) instead of pooled (n, d). One FM per array
# task. Writes to a *_pc tag on group storage (does NOT clobber the pooled bake-off sets).
# Inputs for LIGHTWEIGHT_HEAD_PLAN #2/#3 (multi-stat pooling, cross-channel SE head).
#
#   sbatch notebooks/mdaniol/hpc/extract_per_channel_helios.sbatch         # utica + mantisv2, V1
#   (smoke first:  LIMIT=2000 EMBDIR=$SCRATCH/pc_smoke sbatch ...)
#
# Per-channel is C× the embed calls (V1 -> 9), so slower than the pooled run; 4 h is generous.
# Prereq: aarch64 FM env (setup_env_helios.sh + add_fm_deps_helios.sh) + raw data on Helios pr3.
#
#SBATCH --job-name=shl-extract-pc
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/extractpc_%A_%a.out
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgshl26-gpu-gh200

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# top-2 FM families from BAKEOFF_SPLIT.md; V1 = the 9 physical axes (cross-axis coupling lives here)
MODELS=(utica mantisv2)
M="${MODELS[$SLURM_ARRAY_TASK_ID]}"
VARIANT="${VARIANT:-V1}"

GROUP="$PLG_GROUPS_STORAGE/plggmhealth/shl2026"
DATA="$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet"
EMB="${EMBDIR:-$GROUP/embeddings}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$GROUP/mlruns-helios}"
LIMIT_FLAG=""; [ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # aarch64 venv + aarch64 uv
cd "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs "$EMB"

echo "[$(date)] HELIOS GH200 PER-CHANNEL extract $M $VARIANT -> $EMB/${M}_${VARIANT}_pc (MLflow: $MLFLOW_TRACKING_URI)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

python notebooks/mdaniol/modeling/extract_embeddings.py --model "$M" --prefetch
python notebooks/mdaniol/modeling/extract_embeddings.py --model "$M" --variant "$VARIANT" --per-channel \
    --data-dir "$DATA" --emb-dir "$EMB" --device cuda --chunk-size "${CHUNK:-4000}" --tf-batch "${TFB:-512}" \
    $LIMIT_FLAG
echo "[$(date)] done $M ${VARIANT}_pc. Next (CPU): multi-stat pooling + cross-channel head."
