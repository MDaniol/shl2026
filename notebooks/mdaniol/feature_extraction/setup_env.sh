#!/bin/bash
# Create the "SHL-local" uv venv for feature extraction.
# Run from anywhere; resolves the project root via git.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
VENV="$PROJECT_DIR/SHL-local"
REQ="$PROJECT_DIR/notebooks/mdaniol/feature_extraction/requirements.txt"

echo "Creating venv at: $VENV"
uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" -r "$REQ"

echo "Done. Activate with:  source $VENV/bin/activate"
"$VENV/bin/python" "$PROJECT_DIR/notebooks/mdaniol/feature_extraction/extract_features.py" --self-test
