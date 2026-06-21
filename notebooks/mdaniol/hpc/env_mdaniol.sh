#!/usr/bin/env bash
# SINGLE source of truth for my SHL-2026 environment on Athena.
# Source it once per interactive shell AND it's sourced by every sbatch:
#
#     cd ~/shl2026 && source notebooks/mdaniol/hpc/env_mdaniol.sh
#
# After this, the venv is active and HF_HOME / UV_CACHE_DIR / TMPDIR / PROJECT_DIR
# are all set — no more exporting things by hand. Layers my personal env on top of
# the shared group env.sh.

# 1. group env: PLG paths, MLflow URI, (and UV_CACHE_DIR/TMPDIR if defined there)
source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"

# 2. MY env (team lock + FM add-ons) — overrides env.sh's shared GROUP venv
source "${SHL_VENV:-$SCRATCH/venvs/shl2026}/bin/activate"

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
