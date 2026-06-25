#!/usr/bin/env bash
# Add the FM stack to YOUR personal env on HELIOS (aarch64 / Grace-Hopper GH200).
# Mirrors add_fm_deps.sh but installs aarch64+CUDA wheels. Run AFTER ./scripts/setup_env.sh.
#
# HONEST NOTE: Helios is aarch64 and docs/CLUSTER.md flags its GPU as "needs a separate image".
# This is the bare-venv attempt (simpler than a container). If torch/CUDA fights the GH200,
# fall back to Athena (x86, proven) for extraction, or build an Apptainer image
# (scripts/build_container.sh). The CPU lane (tier1/bake-off) does NOT need this script.
#
#   ./scripts/setup_env_helios.sh                          # aarch64 core env (fresh resolve, once)
#   bash notebooks/mdaniol/hpc/add_fm_deps_helios.sh       # then add the aarch64 FM stack
set -euo pipefail
arch="$(uname -m)"
[ "$arch" = "aarch64" ] || { echo "aarch64 installer; this host is $arch — use add_fm_deps.sh on x86 (Athena)" >&2; exit 1; }

source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"
source "${SHL_VENV:-$SCRATCH/venvs/shl2026}/bin/activate"
echo "installing into: ${VIRTUAL_ENV:?personal env not active} (arch=$arch)"
python -c "import shl2026" 2>/dev/null || { echo "build env first: ./scripts/setup_env.sh" >&2; exit 1; }

# Check the Helios CUDA driver so we pick a compatible torch (run on a node where nvidia-smi works,
# or just proceed — pip selects the aarch64 wheel for this platform automatically).
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || \
  echo "[note] nvidia-smi unavailable here (login node has no GPU) — proceeding; GPU is checked in the smoke job"

# torch for aarch64 + CUDA. PyTorch publishes linux_aarch64 cu124 wheels for recent versions;
# pip auto-selects the aarch64 build on this machine. If install/import fails on GH200, try a
# different CUDA index to match the driver:  cu126 (newer) or cu121 (older).
echo "[torch] installing aarch64 CUDA build (cu124 index) ..."
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124

# rest of the FM stack (same pins as Athena; all have aarch64 wheels)
uv pip install "transformers==4.44.2" "huggingface_hub==0.25.2" "tokenizers>=0.19,<0.20" \
               safetensors einops datasets lightgbm tsfel statsmodels pycatch22
uv pip install --no-deps momentfm mantis-tsfm

python - <<'PY'
import platform, torch, momentfm, mantis, lightgbm
print(f"arch={platform.machine()} | torch {torch.__version__} | cuda compiled={torch.version.cuda} | "
      f"cuda available={torch.cuda.is_available()} (False on a login node is expected)")
print("FM stack imports OK on aarch64")
PY
echo "Done. NEXT: smoke-test on a GPU node before the full run (see HELIOS_SETUP.md §GPU)."
