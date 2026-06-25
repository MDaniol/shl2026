#!/usr/bin/env bash
# Helios (aarch64 / GH200) version of setup_env.sh.
#
# WHY a separate script: the shared uv.lock is resolved for x86 (Athena), so `uv sync` against it
# does NOT work on Helios's ARM CPUs. Here we resolve FRESH from pyproject.toml for aarch64 and
# fetch an aarch64 CPython. Trade-off: minor version drift vs the Athena lock (acceptable; note it
# in the log for traceability). Builds the CPU env (core + lightgbm) — enough for tier1 / bake-off.
# For GPU FM extraction, add the FM stack afterwards: notebooks/mdaniol/hpc/add_fm_deps_helios.sh
#
#   ./scripts/setup_env_helios.sh        # rebuild anytime (after a $SCRATCH purge, etc.)
set -euo pipefail

: "${SCRATCH:?must be set by the cluster at login}"
[ "$(uname -m)" = "aarch64" ] || {
  echo "Helios aarch64 setup, but host is $(uname -m) — use scripts/setup_env.sh on x86 (Athena/Ares)" >&2
  exit 1; }

# LMOD modules can pollute these and break a clean venv; clear them (mirrors setup_env.sh).
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
VENV="${SHL_VENV:-$SCRATCH/venvs/shl2026}"
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
