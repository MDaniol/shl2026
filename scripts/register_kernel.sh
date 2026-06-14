#!/usr/bin/env bash
# Register a venv as a Jupyter kernel for THIS user, with the team env baked in.
#
# Why: JupyterHub kernels do NOT inherit your shell (no env.sh), and the default
# kernel isn't the team venv — so notebooks can't `import shl2026` and don't see
# MLFLOW_TRACKING_URI. This makes a kernel that runs the team venv's Python AND
# carries the shared cache + MLflow URI, so notebooks "just work".
#
# Run once, in a terminal:
#   ./scripts/register_kernel.sh                      # shared team venv -> "SHL 2026 (team)"
#   ./scripts/register_kernel.sh "$SCRATCH/venvs/mine" mine   # personal venv -> "SHL 2026 (mine)"
set -euo pipefail

ROOT="${PLG_GROUPS_STORAGE:?must be on the cluster}/plggmhealth/shl2026"

# JupyterHub loads LMOD modules that put system py3.13 site-packages on
# PYTHONPATH; they shadow this 3.12 venv and crash imports (e.g. pyzmq's Cython
# backend). Clear it so every python call below uses the venv alone.
unset PYTHONPATH

VENV="${1:-$ROOT/venv}"           # which venv to expose as a kernel
SUFFIX="${2:-team}"               # kernel label suffix: team | mine | ...
if [ "$SUFFIX" = "team" ]; then
  NAME="shl2026";          DISPLAY="SHL 2026 (team)"
else
  NAME="shl2026-$SUFFIX";  DISPLAY="SHL 2026 ($SUFFIX)"
fi

PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no python at $PY — is the venv path right?" >&2; exit 1; }
"$PY" -c "import ipykernel" 2>/dev/null || {
  echo "ipykernel is not installed in $VENV" >&2
  echo "  team venv : ask the lead (it's pinned in pyproject [dev])" >&2
  echo "  personal  : uv pip install -e '.[dev]'  (includes ipykernel)" >&2
  exit 1
}

# The env every kernel of this venv should carry (so the kernel is self-sufficient).
CACHE="$ROOT/data/embeddings"
URI_FILE="$ROOT/mlflow_uri"
if [ -f "$URI_FILE" ]; then read -r URI < "$URI_FILE"; else URI="http://172.23.30.9:5000"; fi

# 1) Install the kernelspec (its argv points at this venv's Python).
"$PY" -m ipykernel install --user --name "$NAME" --display-name "$DISPLAY"

# 2) Bake the team env into the kernelspec so notebooks need no source/env.sh.
KDIR="$("$PY" -c "from jupyter_client.kernelspec import KernelSpecManager as K; print(K().get_kernel_spec('$NAME').resource_dir)")"
"$PY" - "$KDIR/kernel.json" "$URI" "$CACHE" <<'PY'
import json, sys
path, uri, cache = sys.argv[1:4]
spec = json.load(open(path))
spec.setdefault("env", {}).update(
    {"PYTHONPATH": "", "MLFLOW_TRACKING_URI": uri, "SHL_EMB_CACHE": cache}
)
json.dump(spec, open(path, "w"), indent=1)
print("baked env into", path, "->", spec["env"])
PY

echo
echo "Registered '$DISPLAY'  (python: $PY)"
echo "In Jupyter: Kernel -> Change Kernel -> '$DISPLAY'."
