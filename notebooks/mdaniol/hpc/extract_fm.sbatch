#!/bin/bash
# Athena GPU: extract a frozen FM's embeddings across input variants (prefetch then extract).
# Reusable for any missing FM (mantis8m, and later unimts/normwear once integrated).
#
#   MODEL=mantis8m VARIANTS="V0 V1 V2" sbatch notebooks/mdaniol/hpc/extract_fm.sbatch
#   MODEL=mantisv2  VARIANTS="V1 V2"    sbatch notebooks/mdaniol/hpc/extract_fm.sbatch
# After it finishes, publish to Ares:  bash notebooks/mdaniol/hpc/stage_to_group.sh emb
#
#SBATCH --job-name=shl-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/extract_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
: "${MODEL:?set MODEL=mantis8m (or another FM)}"

echo "[$(date)] prefetch $MODEL weights (serial, avoids the array download race)"
python notebooks/mdaniol/modeling/extract_embeddings.py --model "$MODEL" --prefetch
for V in ${VARIANTS:-V0 V1 V2}; do
  echo "[$(date)] extract $MODEL $V"
  python notebooks/mdaniol/modeling/extract_embeddings.py --model "$MODEL" --variant "$V" \
      --data-dir "$PROJECT_DIR/dataset_parquet" --emb-dir "$PROJECT_DIR/embeddings" \
      --device cuda --chunk-size "${CHUNK:-8000}" --tf-batch "${TFB:-512}"
done
echo "[$(date)] done. Publish to Ares:  bash notebooks/mdaniol/hpc/stage_to_group.sh emb"
