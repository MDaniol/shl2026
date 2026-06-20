#!/bin/bash
# Athena GPU array job: one frozen-FM embedding extraction per array task.
# A100 is ~5-10x faster than the laptop MPS and has robust CUDA memory mgmt.
#
#SBATCH --job-name=shl-emb
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/emb_%A_%a.out
## --- EDIT for your allocation ---
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026}"

# one model per array index (extend freely; bump --array to match)
MODELS=(mantisv2 utica mantis8m moment-small)
M="${MODELS[$SLURM_ARRAY_TASK_ID]}"

source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"   # activates YOUR env (team lock + your FM add-ons)
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

echo "[$(date)] embedding model=$M on $(hostname)"
python notebooks/mdaniol/modeling/extract_embeddings.py \
    --model "$M" --variant "${VARIANT:-V0}" \
    --data-dir "$PROJECT_DIR/dataset_parquet" \
    --emb-dir  "$PROJECT_DIR/embeddings" \
    --device cuda --chunk-size 8000 --tf-batch 512
echo "[$(date)] done $M"
