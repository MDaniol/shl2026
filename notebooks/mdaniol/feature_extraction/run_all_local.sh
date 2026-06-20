#!/bin/bash
# Run feature extraction over all 9 files sequentially on the local machine
# (no SLURM). Uses the SHL-local venv. ~10 min total at ~3.3k windows/s.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/dataset_parquet}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/dataset_parquet_features}"
VENV="${VENV:-$PROJECT_DIR/SHL-local}"
cd "$PROJECT_DIR/notebooks/mdaniol/feature_extraction"

for REL in train/Bag train/Hips train/Torso train/Hand \
           validation/Bag validation/Hips validation/Torso validation/Hand \
           test/all ; do
    SPLIT="${REL%%/*}"; LOC="${REL##*/}"
    mkdir -p "$OUT_DIR/$SPLIT"
    "$VENV/bin/python" extract_features.py \
        --input  "$DATA_DIR/$REL.parquet" \
        --output "$OUT_DIR/$REL.parquet" \
        --split "$SPLIT" --location "$LOC" --chunk-size 20000
done
echo "All features written under: $OUT_DIR"
