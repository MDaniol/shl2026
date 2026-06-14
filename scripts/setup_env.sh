#!/usr/bin/env bash
# One-time (re-runnable): build YOUR OWN environment from the team's locked spec
# (pyproject.toml + uv.lock) and register it as a Jupyter kernel, so you can
# start working in JupyterHub. Everyone gets identical versions (from the lock);
# your env is yours to install into without affecting anyone else.
#
# Re-run anytime to rebuild — after a $SCRATCH purge, or when the shared spec
# changes:  git pull && ./scripts/setup_env.sh
#
#   ./scripts/setup_env.sh
set -euo pipefail

: "${PLG_GROUPS_STORAGE:?must be on the cluster}"
: "${SCRATCH:?must be set by the cluster at login}"

# LMOD modules (loaded by JupyterHub) pollute these and would break the 3.12
# env; clear them so uv + the kernel use the venv alone.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP

export UV_PROJECT_ENVIRONMENT="${SHL_VENV:-$SCRATCH/venvs/shl2026}"  # where YOUR env lives
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv-cache}"             # same filesystem -> cheap hardlinks

REPO="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
cd "$REPO"
[ -f uv.lock ] || { echo "no uv.lock in $REPO — run 'git pull' (or ask the lead)" >&2; exit 1; }

echo "Building your env at $UV_PROJECT_ENVIRONMENT from the shared lock ..."
uv sync --extra dev --python 3.12      # identical versions for everyone (from uv.lock)

"$REPO/scripts/register_kernel.sh" "$UV_PROJECT_ENVIRONMENT"

echo
echo "Your env : $UV_PROJECT_ENVIRONMENT   (rebuild anytime: ./scripts/setup_env.sh)"
echo "Next     : restart your JupyterHub server (Hub Control Panel -> Stop/Start),"
echo "           then pick the 'SHL 2026' kernel. In terminals/batch: source env.sh."
