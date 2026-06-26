#!/usr/bin/env bash
# Idempotently link raw + heavy-output data into the repo so $PROJECT_DIR-relative
# paths resolve. Safe to run repeatedly (every job calls it at startup); safe to
# run by hand once after a fresh clone or a $SCRATCH purge:
#
#   bash notebooks/mdaniol/hpc/link_data.sh [PROJECT_DIR]
#
# Canonical sources (override via env if your layout differs). Defaults point at GROUP
# storage — features/embeddings are persistent there (Helios pr3); $SCRATCH auto-purges
# (30 d) and is NOT where the staged data lives, so $SCRATCH defaults caused submit jobs
# to link an empty dir ("test__all.npy missing"). Group is the single source of truth.
#   RAW_SRC  = $PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet                 (read-only raw)
#   FEAT_SRC = $PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features (features)
#   EMB_SRC  = $PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings              (embeddings)
set -euo pipefail

ROOT="${1:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${PLG_GROUPS_STORAGE:?must be on the cluster}"
: "${SCRATCH:?must be set by the cluster at login}"
RAW_SRC="${RAW_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet}"
FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"

mkdir -p "$FEAT_SRC" "$EMB_SRC"   # outputs live on $SCRATCH; raw is read-only

link() {  # link <src> <dst> : (re)create symlink, but never clobber a real dir
  local src="$1" dst="$2"
  if [ -L "$dst" ] || [ ! -e "$dst" ]; then
    ln -sfn "$src" "$dst"; echo "  linked $dst -> $src"
  elif [ -d "$dst" ]; then
    echo "  kept   $dst (real dir, not a symlink — left as-is)"
  fi
}

link "$RAW_SRC"  "$ROOT/dataset_parquet"
link "$FEAT_SRC" "$ROOT/dataset_parquet_features"
link "$EMB_SRC"  "$ROOT/embeddings"
