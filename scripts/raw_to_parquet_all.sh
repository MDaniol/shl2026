#!/usr/bin/env bash
# Batch every unpacked SHL archive under a root into Parquet.
#
# Expects each archive already unpacked as <SRC_ROOT>/SHL-2026-<Split>_<Location>/
# (e.g. .../SHL-2026-Train_Bag/train/Bag/*.txt). Writes <OUT_ROOT>/<split>/<location>.parquet.
#
# Intended to run on Athena group storage (where the full ~95 GiB unpacked set
# fits), NOT on a laptop. Convert-then-delete-raw per archive if space is tight.
#
#   ./scripts/raw_to_parquet_all.sh dataset dataset_parquet
#
set -euo pipefail

SRC_ROOT="${1:?usage: raw_to_parquet_all.sh <SRC_ROOT> <OUT_ROOT>}"
OUT_ROOT="${2:?usage: raw_to_parquet_all.sh <SRC_ROOT> <OUT_ROOT>}"
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
# Prefer zips (read in place, no unpack); fall back to unpacked dirs.
archives=("$SRC_ROOT"/SHL-2026-*.zip)
[ ${#archives[@]} -eq 0 ] && archives=("$SRC_ROOT"/SHL-2026-*/)
if [ ${#archives[@]} -eq 0 ]; then
  echo "no SHL-2026-* archives (.zip or dir) under $SRC_ROOT" >&2
  exit 1
fi

for arc in "${archives[@]}"; do
  echo "==> $arc"
  "$PY" "$HERE/raw_to_parquet.py" --src "$arc" --out "$OUT_ROOT"
done
echo "done: $OUT_ROOT"
