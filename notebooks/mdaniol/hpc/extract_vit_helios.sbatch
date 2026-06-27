#!/bin/bash
# Helios GH200: extract frozen VISION-ViT (DINOv2) embeddings from per-channel IMU spectrogram-IMAGES
# -> a diversity voter with the STRONGEST published evidence (TiViT, arXiv:2506.08641: frozen ViT +
# linear probe beats temporal FMs and ensembles to +1.7-3 pts). Uses an INTERMEDIATE layer.
# A big ViT applied per channel is SLOW -> ALWAYS smoke first.
#
#   SMOKE: LIMIT=2000 EMBDIR=$SCRATCH/vit_smoke MODEL=dinov2 sbatch notebooks/mdaniol/hpc/extract_vit_helios.sbatch
#   FULL:  MODEL=dinov2 sbatch notebooks/mdaniol/hpc/extract_vit_helios.sbatch   (or MODEL=dinov2-large)
#
#SBATCH --job-name=shl-extract-vit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/extractvit_%j.out
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgshl26-gpu-gh200

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL="${MODEL:-dinov2}"; VARIANT="${VARIANT:-V2}"; LFRAC="${LFRAC:-0.65}"
GROUP="$PLG_GROUPS_STORAGE/plggmhealth/shl2026"
DATA="$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet"
EMB="${EMBDIR:-$GROUP/embeddings}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$GROUP/mlruns-helios}"
LIMIT_FLAG=""; [ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # aarch64 venv (torch)
cd "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs "$EMB"
python -c "import transformers, scipy" 2>/dev/null || uv pip install "transformers==4.44.2" scipy
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }

echo "[$(date)] HELIOS GH200 ViT extract $MODEL variant=$VARIANT layer-frac=$LFRAC limit=${LIMIT:-all} -> $EMB/${MODEL}_${VARIANT}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python notebooks/mdaniol/modeling/extract_embeddings.py --model "$MODEL" --variant "$VARIANT" \
    --vit-layer-frac "$LFRAC" --data-dir "$DATA" --emb-dir "$EMB" --device cuda \
    --chunk-size "${CHUNK:-2000}" --tf-batch "${TFB:-64}" $LIMIT_FLAG
echo "[$(date)] done ${MODEL}_${VARIANT}. Next (CPU): probe_fusion_ares (add to EMBS) -> add to vote."
