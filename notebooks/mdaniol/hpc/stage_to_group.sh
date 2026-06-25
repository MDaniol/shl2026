#!/usr/bin/env bash
# Run on ATHENA. Stage cached features + ALL FM embeddings from Athena $SCRATCH to shared
# GROUP storage, so Ares (CPU) can run tier1 / the bake-off (Athena scratch is invisible to
# Ares). Idempotent (rsync -a) — re-run after extracting a new FM to publish it to Ares.
#
#   bash notebooks/mdaniol/hpc/stage_to_group.sh            # stage features + all embeddings
#   bash notebooks/mdaniol/hpc/stage_to_group.sh emb        # only embeddings
#   bash notebooks/mdaniol/hpc/stage_to_group.sh feat       # only features
set -euo pipefail
: "${SCRATCH:?must be on the cluster (Athena)}"
: "${PLG_GROUPS_STORAGE:?must be on the cluster}"

SRC="$SCRATCH/shl2026"
DST="$PLG_GROUPS_STORAGE/plggmhealth/shl2026"
WHAT="${1:-all}"
mkdir -p "$DST/embeddings"

if [ "$WHAT" = "all" ] || [ "$WHAT" = "feat" ]; then
  echo "[stage] features -> $DST/dataset_parquet_features"
  rsync -a --info=progress2 "$SRC/dataset_parquet_features" "$DST/"
fi
if [ "$WHAT" = "all" ] || [ "$WHAT" = "emb" ]; then
  echo "[stage] all cached embeddings -> $DST/embeddings/"
  rsync -a --info=progress2 "$SRC/embeddings/" "$DST/embeddings/"
fi
echo "[stage] done. On Ares:  sbatch notebooks/mdaniol/hpc/probe_fusion_ares.sbatch"
ls "$DST/embeddings" 2>/dev/null | sed 's/^/  staged: /'
