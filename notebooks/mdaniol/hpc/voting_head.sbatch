#!/bin/bash
# CPU job: calibrated soft-voting head over the top-K fused FMs from BAKEOFF_SPLIT.md.
# Pure CPU (LightGBM fusion per FM + NumPy vote) on CACHED embeddings -> runs on the
# x86 CPU partition (Helios plgrid / Ares). Reuses env_mdaniol.sh + group-storage data.
#
#   EMBS=utica_V2,mantisv2_V1 sbatch notebooks/mdaniol/hpc/voting_head.sbatch
#   (default EMBS = the bake-off top-2)
#
#SBATCH --job-name=shl-vote
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/vote_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
EMBS="${EMBS:-utica_V2,mantisv2_V1}"

export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios}"

SPLIT="$PROJECT_DIR/notebooks/mdaniol/modeling/artifacts/val_split_temporal.npy"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

[ -f "$SPLIT" ] || { echo "[$(date)] FATAL: split $SPLIT missing (run the bake-off first)" >&2; exit 1; }

# require every FM's cached embeddings (skip-not-fail mirrors the probe job)
for e in ${EMBS//,/ }; do
  for sp in train validation; do for loc in Bag Hips Torso Hand; do
    if [ ! -f "$PROJECT_DIR/embeddings/$e/${sp}__${loc}.npy" ]; then
      echo "[$(date)] SKIP — missing embeddings/$e/${sp}__${loc}.npy (stage it to group storage first)"; exit 0
    fi
  done; done
done

echo "[$(date)] voting head EMBS=$EMBS on $(hostname)"
python notebooks/mdaniol/modeling/voting_head.py --embs "$EMBS" \
    --split "$SPLIT" \
    --emb-root "$PROJECT_DIR/embeddings" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" || {
        echo "[$(date)] FAILED voting head" >&2; exit 1; }
echo "[$(date)] done voting head"
