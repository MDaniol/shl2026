#!/usr/bin/env bash
# Add the FM-branch dependencies to YOUR personal env — does NOT touch the shared
# pyproject.toml / uv.lock. Run AFTER `./scripts/setup_env.sh` has built your env
# from the team lock. Per STUDENTS.md: "It's YOUR env: uv pip install anything you
# like into it without affecting anyone else."
#
#   ./scripts/setup_env.sh                       # team lock -> your env (once)
#   bash notebooks/mdaniol/hpc/add_fm_deps.sh    # then add the FM stack
set -euo pipefail

source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"   # group paths + UV_CACHE_DIR/TMPDIR on $SCRATCH
# env.sh activates the shared GROUP venv; override it with YOUR personal env so
# these deps land in $SCRATCH/venvs/shl2026, NOT the group env.
source "${SHL_VENV:-$SCRATCH/venvs/shl2026}/bin/activate"
echo "installing into: ${VIRTUAL_ENV:?personal env not active}"
python -c "import shl2026" 2>/dev/null || {
  echo "build your env first:  ./scripts/setup_env.sh" >&2; exit 1; }
echo "uv cache: ${UV_CACHE_DIR:?env.sh must export UV_CACHE_DIR (on \$SCRATCH)}"

# Pinned: newer transformers/huggingface_hub break mantis-tsfm's from_pretrained.
uv pip install torch "transformers==4.44.2" "huggingface_hub==0.25.2" "tokenizers>=0.19,<0.20" \
               safetensors einops datasets lightgbm tsfel statsmodels pycatch22
# momentfm + mantis-tsfm pin an unbuildable old transformers -> skip their deps.
uv pip install --no-deps momentfm mantis-tsfm

python - <<'PY'
import torch, momentfm, mantis, lightgbm, tsfel
print(f"FM deps OK in your env | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
PY
echo "Done. Your env now has the FM stack (the shared lock is untouched)."
