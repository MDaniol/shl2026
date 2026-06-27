#!/bin/bash
# Helios GH200: extract frozen AST (Audio Spectrogram Transformer) embeddings from per-channel
# IMU log-spectrograms -> a representation ORTHOGONAL to the temporal FMs (diversity voter, E-AST-01).
# AST is a big ViT applied per channel, so it is SLOWER than the temporal FMs — ALWAYS smoke first.
#
#   SMOKE (2000 win/file, throwaway dir) -> read the win/s before committing to the full run:
#     LIMIT=2000 EMBDIR=$SCRATCH/ast_smoke sbatch notebooks/mdaniol/hpc/extract_ast_helios.sbatch
#   FULL (after the smoke rate looks acceptable):
#     sbatch notebooks/mdaniol/hpc/extract_ast_helios.sbatch
#
#SBATCH --job-name=shl-extract-ast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/extractast_%j.out
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgshl26-gpu-gh200

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VARIANT="${VARIANT:-V2}"                       # V2 = 5 orientation-invariant magnitudes (full accuracy)
PC="${PC:---per-channel}"   # keep per-channel (n,C,d) -> cross-channel SE head; set PC="" to mean-pool
GROUP="$PLG_GROUPS_STORAGE/plggmhealth/shl2026"
DATA="$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet"
EMB="${EMBDIR:-$GROUP/embeddings}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$GROUP/mlruns-helios}"
LIMIT_FLAG=""; [ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # aarch64 venv (torch)
cd "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs "$EMB"
# transformers (ASTModel) + scipy must be present in the aarch64 env
python -c "import transformers, scipy" 2>/dev/null || uv pip install "transformers==4.44.2" scipy
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }

echo "[$(date)] HELIOS GH200 AST extract variant=$VARIANT limit=${LIMIT:-all} -> $EMB/ast_${VARIANT}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python notebooks/mdaniol/modeling/extract_embeddings.py --model ast --variant "$VARIANT" $PC \
    --data-dir "$DATA" --emb-dir "$EMB" --device cuda --chunk-size "${CHUNK:-2000}" \
    --tf-batch "${TFB:-128}" $LIMIT_FLAG
echo "[$(date)] done ast_${VARIANT}. Next (CPU): probe_fusion_ares (add ast_${VARIANT} to EMBS) -> vote."
