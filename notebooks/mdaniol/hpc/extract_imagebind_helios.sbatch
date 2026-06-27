#!/bin/bash
# Helios GH200: extract Meta ImageBind's NATIVE frozen IMU encoder (arXiv:2305.05665) -> 1024-d.
# A truly orthogonal voter (native multimodal FM, trained on a different IMU distribution). Caveat:
# Ego4D head-mounted Aria IMU -> domain gap to phone -> VALIDATE STANDALONE (per-class incl.
# Run/vehicles) before trusting it in the vote.
#
#   SMOKE: LIMIT=2000 EMBDIR=$SCRATCH/ib_smoke sbatch notebooks/mdaniol/hpc/extract_imagebind_helios.sbatch
#   FULL:  sbatch notebooks/mdaniol/hpc/extract_imagebind_helios.sbatch
#
#SBATCH --job-name=shl-extract-ib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/extractib_%j.out
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgshl26-gpu-gh200

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
GROUP="$PLG_GROUPS_STORAGE/plggmhealth/shl2026"
DATA="$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet"
EMB="${EMBDIR:-$GROUP/embeddings}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$GROUP/mlruns-helios}"
LIMIT_FLAG=""; [ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # aarch64 venv (torch)
cd "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs "$EMB"
# the 4.8 GB imagebind_huge.pth downloads to ./.checkpoints — keep it off the 10 GB $HOME quota
mkdir -p "$SCRATCH/imagebind_ckpt"; ln -sfn "$SCRATCH/imagebind_ckpt" "$PROJECT_DIR/.checkpoints"
# ImageBind is not on PyPI cleanly; install from the repo (aarch64 deps may be finicky -> smoke first)
python -c "import imagebind" 2>/dev/null || uv pip install "git+https://github.com/facebookresearch/ImageBind.git"
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }

echo "[$(date)] HELIOS GH200 ImageBind-IMU extract limit=${LIMIT:-all} -> $EMB/imagebind_V0"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python notebooks/mdaniol/modeling/extract_embeddings.py --model imagebind --variant V0 \
    --data-dir "$DATA" --emb-dir "$EMB" --device cuda --chunk-size "${CHUNK:-4000}" --tf-batch "${TFB:-256}" \
    $LIMIT_FLAG
echo "[$(date)] done imagebind_V0. Next (CPU): probe_fusion_ares (add imagebind_V0 to EMBS) -> validate standalone -> vote."
