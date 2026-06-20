#!/bin/bash
# Build the SHL-local venv on ACK Cyfronet Athena (x86 + CUDA A100).
# Run once on a login node (uv must be available; `pip install uv` if not).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026}"
VENV="${VENV:-$PROJECT_DIR/SHL-local}"
REQ="$(cd "$(dirname "$0")" && pwd)/requirements_hpc.txt"

echo "Project: $PROJECT_DIR"
echo "Venv:    $VENV"
command -v uv >/dev/null || { echo "installing uv"; pip install --user uv; }

uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" -r "$REQ"
# FM packages without their (broken) dependency pins:
uv pip install --python "$VENV/bin/python" --no-deps momentfm mantis-tsfm

echo "=== verify ==="
"$VENV/bin/python" - <<'PY'
import torch, momentfm, mantis, transformers, lightgbm
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__, "| lightgbm", lightgbm.__version__)
print("momentfm + mantis import OK")
PY
echo "Done. Activate with: source $VENV/bin/activate"
