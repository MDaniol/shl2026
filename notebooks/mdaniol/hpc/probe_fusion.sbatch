#!/bin/bash
# Athena CPU array job: probe + fusion per cached embedding set (one per task).
# CPU-only (LightGBM / logreg); big RAM for the 784k x 2304 train embeddings.
# Default protocol = train_split (USE_SPLIT=1): fit on User1 + validation[FIT],
# calibrate on TUNE, eval on held-out TEST -> apples-to-apples vs the handcrafted
# lane (writes BAKEOFF_SPLIT.md with the decisive "Δ vs handcrafted" column).
# Set USE_SPLIT=0 to run the old train->validation probe (BAKEOFF_RESULTS.md).
#
#SBATCH --job-name=shl-probe
#SBATCH --array=0-6
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/probe_%A_%a.out
## --- EDIT for your allocation (CPU grant) ---
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1   # Athena is GPU-only
#SBATCH --account=plgshl26-gpu-a100

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# the cached embedding sets to probe (one per array task; bump --array to match)
EMBS=(mantisv2_V0 mantisv2_V1 mantisv2_V2 utica_V1 utica_V2 moment-small_V1 moment-small_V2)
E="${EMBS[$SLURM_ARRAY_TASK_ID]}"
SPLIT_FLAG=""; [ "${USE_SPLIT:-1}" = "1" ] && SPLIT_FLAG="--use-split \
    --split $PROJECT_DIR/notebooks/mdaniol/modeling/artifacts/val_split.npy"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # single env source: group env + my venv + HF/uv/tmp caches
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"   # ensure data symlinks
mkdir -p notebooks/mdaniol/modeling/logs

# skip (not fail) embedding sets that haven't been extracted yet — e.g. utica_V2
# until variant_sweep regenerates it. A genuine probe error still -> FAILED below.
EMB_DIR="$PROJECT_DIR/embeddings/$E"
for sp in train validation; do for loc in Bag Hips Torso Hand; do
  if [ ! -f "$EMB_DIR/${sp}__${loc}.npy" ]; then
    echo "[$(date)] SKIP $E — missing $EMB_DIR/${sp}__${loc}.npy (extract it first)"
    exit 0
  fi
done; done

echo "[$(date)] probe $E (USE_SPLIT=${USE_SPLIT:-1}) on $(hostname)"
python notebooks/mdaniol/modeling/probe_fusion.py --emb "$E" $SPLIT_FLAG \
    --emb-root "$PROJECT_DIR/embeddings" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" || {
        echo "[$(date)] FAILED $E" >&2; exit 1; }
echo "[$(date)] done $E"
