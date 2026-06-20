#!/bin/bash
# Upload the (gitignored) data dirs to Athena group storage via rsync.
# Code travels via git (push + clone); only the large data needs rsync.
#
# Usage:
#   ATHENA=plgYOURLOGIN@athena.cyfronet.pl \
#   DEST=$PLG_GROUPS_STORAGE_ON_ATHENA/shl2026 \
#   bash notebooks/mdaniol/hpc/stage_to_athena.sh
# (find DEST by running `echo $PLG_GROUPS_STORAGE/plggmhealth/shl2026/shl2026` on Athena)
set -euo pipefail

ATHENA="${ATHENA:?set ATHENA=plgLOGIN@athena.cyfronet.pl}"
DEST="${DEST:?set DEST=<PROJECT_DIR on Athena, e.g. \$PLG_GROUPS_STORAGE/plggmhealth/shl2026/shl2026>}"
LOCAL="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

ssh "$ATHENA" "mkdir -p '$DEST'"
echo "Uploading dataset_parquet (≈15 GB) + dataset_parquet_features (≈2.7 GB) → $ATHENA:$DEST"
rsync -avh --partial --progress "$LOCAL/dataset_parquet"          "$ATHENA:$DEST/"
rsync -avh --partial --progress "$LOCAL/dataset_parquet_features" "$ATHENA:$DEST/"
echo "Done. (raw dataset/ NOT uploaded — only needed if you rebuild parquet on Athena.)"
