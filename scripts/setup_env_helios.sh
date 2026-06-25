#!/usr/bin/env bash
# Helios GH200 (aarch64 / Grace ARM) env builder — ONLY for the GH200 GPU partition.
#
# Helios is HYBRID: login + CPU nodes are x86_64 (use the normal scripts/setup_env.sh there — the
# x86 uv.lock works). ONLY the GH200 GPU nodes are aarch64. ARM wheels can't be built on the x86
# login node, so this script MUST be run ON a GH200 node:
#
#   srun -A plgshl26-gpu-gh200 -p plgrid-gpu-gh200 --gres=gpu:1 --time=1:00:00 --pty bash
#   cd ~/shl2026 && ./scripts/setup_env_helios.sh        # then: add_fm_deps_helios.sh for the FM stack
#
# We resolve FRESH from pyproject for aarch64 (the x86 lock won't sync on ARM); minor version drift
# vs Athena, acceptable for extraction. The aarch64 venv lives at a SEPARATE path so it never
# clobbers the x86 venv.
set -euo pipefail

: "${SCRATCH:?must be set by the cluster at login}"
[ "$(uname -m)" = "aarch64" ] || {
  echo "This must run on a GH200 (aarch64) node, but host is $(uname -m)." >&2
  echo "Get one:  srun -A plgshl26-gpu-gh200 -p plgrid-gpu-gh200 --gres=gpu:1 --time=1:00:00 --pty bash" >&2
  echo "(For the Helios CPU lane — tier1/bake-off — use scripts/setup_env.sh on the x86 login node.)" >&2
  exit 1; }

# LMOD modules can pollute these and break a clean venv; clear them (mirrors setup_env.sh).
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
VENV="${SHL_VENV_AARCH64:-$SCRATCH/venvs/shl2026-gh200}"   # separate from the x86 venv
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv-cache}"
REPO="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
cd "$REPO"

echo "Helios aarch64: building fresh env at $VENV (resolve from pyproject, NOT the x86 lock) ..."
uv python install 3.12                       # aarch64 CPython (not the x86 interpreter in the lock)
uv venv "$VENV" --python 3.12
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -e .                          # fresh aarch64 resolve of the core deps
uv pip install lightgbm pytest               # tier1 / bake-off deps (lightgbm isn't in the core spec)

python - <<'PY'
import platform, lightgbm
from shl2026 import track, evaluate_predictions   # noqa: F401  (import smoke test)
print(f"helios env OK | arch={platform.machine()} | python ok | lightgbm {lightgbm.__version__}")
PY

echo
echo "Env  : $VENV   (CPU lane ready: tier1 / bake-off)"
echo "Use  : source notebooks/mdaniol/hpc/env_mdaniol.sh   (activates this venv in shells/jobs)"
echo "GPU  : for FM extraction add the stack -> bash notebooks/mdaniol/hpc/add_fm_deps_helios.sh"
