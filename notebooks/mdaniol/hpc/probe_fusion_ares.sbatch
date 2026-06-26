#!/bin/bash
# Ares (CPU-only) version of probe_fusion.sbatch — the temporal FM bake-off.
# Pure CPU (LightGBM / logreg) on CACHED embeddings, so it runs natively on Ares.
# One cached embedding set per array task; sets not on group storage are SKIPPED.
#
# PREREQ: copy the cached embeddings you want probed to GROUP storage (from Athena), e.g.:
#   rsync -a $SCRATCH/shl2026/embeddings/{mantisv2_V0,utica_V0,moment-small_V1} \
#         $PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings/
#   rsync -a $SCRATCH/shl2026/dataset_parquet_features \
#         $PLG_GROUPS_STORAGE/plggmhealth/shl2026/
#
# Run:  sbatch notebooks/mdaniol/hpc/probe_fusion_ares.sbatch
#   (adjust account/partition for your CPU grant, or: sbatch -A plgshl26-cpu -p plgrid ...)
#
#SBATCH --job-name=shl-probe
#SBATCH --array=0-10
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00         # fusion head (lgbm on emb⊕520, ~800k rows) — heavy + deterministic (~3-4x)
#SBATCH --output=notebooks/mdaniol/modeling/logs/probe_ares_%A_%a.out
#SBATCH --partition=plgrid                 # Ares CPU partition (no GPU); adjust if needed
#SBATCH --account=plgshl26-cpu             # CPU grant; CONFIRM your Ares account name

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# data on shared GROUP storage (Athena scratch is not visible from Ares/Helios)
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
# MLflow: same file store as the extraction (group env.sh server, if present, overrides via :-)
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios}"

# the cached embedding sets to probe (one per array task). Sets not present are SKIPPED.
EMBS=(mantisv2_V0 mantisv2_V1 mantisv2_V2 \
      mantis8m_V0 mantis8m_V1 mantis8m_V2 \
      utica_V0 utica_V1 utica_V2 \
      moment-small_V1 moment-small_V2)
E="${EMBS[$SLURM_ARRAY_TASK_ID]}"

SCHEME="temporal"; SPLIT_NAME="val_split_temporal.npy"
[ "${TEMPORAL:-1}" = "0" ] && { SCHEME="blocked"; SPLIT_NAME="val_split.npy"; }
SPLIT="$PROJECT_DIR/notebooks/mdaniol/modeling/artifacts/$SPLIT_NAME"
SPLIT_FLAG=""; [ "${USE_SPLIT:-1}" = "1" ] && SPLIT_FLAG="--use-split --split $SPLIT"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

# race-safe split generation (per-task temp .npy -> atomic mv -n; first wins)
if [ "${USE_SPLIT:-1}" = "1" ] && [ ! -f "$SPLIT" ]; then
  TMP="${SPLIT%.npy}.task${SLURM_ARRAY_TASK_ID:-0}.tmp.npy"
  echo "[$(date)] generating $SCHEME split (race-safe) -> $SPLIT"
  if python notebooks/mdaniol/modeling/split.py --scheme "$SCHEME" --embargo 100 \
        --feat-dir "$PROJECT_DIR/dataset_parquet_features" --out "$TMP"; then
    mv -n "$TMP" "$SPLIT"
    mv -n "${TMP%.npy}.json" "${SPLIT%.npy}.json" 2>/dev/null || true
  fi
  rm -f "$TMP" "${TMP%.npy}.json"
fi
if [ "${USE_SPLIT:-1}" = "1" ] && [ ! -f "$SPLIT" ]; then
  echo "[$(date)] FATAL: split $SPLIT unavailable" >&2; exit 1
fi

# skip (not fail) embedding sets not staged to group storage yet
EMB_DIR="$PROJECT_DIR/embeddings/$E"
for sp in train validation; do for loc in Bag Hips Torso Hand; do
  if [ ! -f "$EMB_DIR/${sp}__${loc}.npy" ]; then
    echo "[$(date)] SKIP $E — missing $EMB_DIR/${sp}__${loc}.npy (stage it to group storage first)"
    exit 0
  fi
done; done

echo "[$(date)] probe $E (split=$SPLIT_NAME) on ARES $(hostname)"
python notebooks/mdaniol/modeling/probe_fusion.py --emb "$E" $SPLIT_FLAG \
    --emb-root "$PROJECT_DIR/embeddings" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" || {
        echo "[$(date)] FAILED $E" >&2; exit 1; }
echo "[$(date)] done $E"
