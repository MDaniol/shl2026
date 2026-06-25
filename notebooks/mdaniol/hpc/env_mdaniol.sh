#!/usr/bin/env bash
# SINGLE source of truth for my SHL-2026 environment on Athena.
# Source it once per interactive shell AND it's sourced by every sbatch:
#
#     cd ~/shl2026 && source notebooks/mdaniol/hpc/env_mdaniol.sh
#
# After this, the venv is active and HF_HOME / UV_CACHE_DIR / TMPDIR / PROJECT_DIR
# are all set — no more exporting things by hand. Layers my personal env on top of
# the shared group env.sh.

# 0. uv is arch-specific and $HOME is shared across x86/aarch64 nodes — on ARM (Helios GH200)
#    prefer the aarch64 uv from its own dir so it doesn't hit the x86 uv in ~/.local/bin.
[ "$(uname -m)" = "aarch64" ] && export PATH="$HOME/.local/bin/aarch64:$PATH"

# 1. group env: PLG paths, MLflow URI, (and UV_CACHE_DIR/TMPDIR if defined there)
ENVSH="$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"   # on Athena pr2; absent on Helios pr3
[ -f "$ENVSH" ] && source "$ENVSH" || echo "[env_mdaniol] $ENVSH absent — using defaults (MLflow URI may be unset)"

# 2. MY env — overrides env.sh's shared GROUP venv. Pick by CPU arch: Helios GH200 is aarch64
#    (Grace ARM); everything else (Athena/Ares/Helios-CPU/login) is x86_64. ARM and x86 venvs are
#    NOT interchangeable, so they live at separate paths.
if [ "$(uname -m)" = "aarch64" ]; then
  source "${SHL_VENV_AARCH64:-$SCRATCH/venvs/shl2026-gh200}/bin/activate"
else
  source "${SHL_VENV:-$SCRATCH/venvs/shl2026}/bin/activate"
fi

# 3. all caches off $HOME (10 GB quota) -> per-user $SCRATCH; HF runtime knobs
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv-cache}"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

# 4. repo root = where I submitted from (batch) or current dir (interactive)
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

echo "[env_mdaniol] venv=$VIRTUAL_ENV"
echo "[env_mdaniol] HF_HOME=$HF_HOME  UV_CACHE_DIR=$UV_CACHE_DIR  PROJECT_DIR=$PROJECT_DIR"
