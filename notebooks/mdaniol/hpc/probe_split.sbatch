#!/bin/bash
# Athena array: re-probe each cached FM embedding set under the train_split.py
# protocol (fit = User1 + validation[FIT], calibrate on TUNE, eval on held-out
# TEST) -> apples-to-apples vs the handcrafted lane. One (model:variant) per task.
# Writes BAKEOFF_SPLIT.md with the decisive "Δ vs handcrafted" column.
#
# Prereqs: variant_sweep/extract_embeddings produced the embeddings, and
# train_split.sbatch (or split.py) produced artifacts/val_split.npy.
#
#SBATCH --job-name=shl-psplit
#SBATCH --array=0-5
#SBATCH --gres=gpu:1            # Athena is GPU-only (this job is CPU/LightGBM)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/psplit_%A_%a.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgshl26-gpu-a100

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# the six cached embedding sets from the sweep (one per array task)
EMBS=(mantisv2_V0 mantisv2_V1 mantisv2_V2 utica_V1 moment-small_V1 moment-small_V2)
E="${EMBS[$SLURM_ARRAY_TASK_ID]}"

source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"   # group paths + UV_CACHE_DIR
source "${SHL_VENV:-$SCRATCH/venvs/shl2026}/bin/activate" # YOUR env (team lock + FM add-ons)
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"   # ensure data symlinks
mkdir -p notebooks/mdaniol/modeling/logs

echo "[$(date)] split-probe $E"
python notebooks/mdaniol/modeling/probe_fusion.py --emb "$E" --use-split \
    --emb-root "$PROJECT_DIR/embeddings" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" \
    --split "$PROJECT_DIR/notebooks/mdaniol/modeling/artifacts/val_split.npy" || {
        echo "[$(date)] FAILED $E" >&2; exit 1; }
echo "[$(date)] done $E"
