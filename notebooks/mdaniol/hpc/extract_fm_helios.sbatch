#!/bin/bash
# Helios GH200 (aarch64 GPU): extract a frozen FM's embeddings across input variants.
# Helios variant of extract_fm.sbatch — same driver, GH200 headers + aarch64 env.
#
# PREREQ (see HELIOS_SETUP.md §GPU):
#   1. aarch64 FM env:  ./scripts/setup_env.sh  &&  bash notebooks/mdaniol/hpc/add_fm_deps_helios.sh
#   2. RAW data on Helios group storage (pr3): rsync dataset_parquet there (raw is ~15 GB).
#   3. SMOKE-TEST first (cheap):  MODEL=mantis8m VARIANTS=V0 LIMIT=2000 sbatch this file
#      -> confirm the FM runs on GH200 + measure speed BEFORE the full extraction.
#
#   MODEL=mantis8m VARIANTS="V0 V1 V2" sbatch notebooks/mdaniol/hpc/extract_fm_helios.sbatch
# After it finishes:  bash notebooks/mdaniol/hpc/stage_to_group.sh emb   (publish for the bake-off)
#
#SBATCH --job-name=shl-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/extract_helios_%j.out
#SBATCH --partition=plgrid-gpu-gh200       # Helios GH200 partition (48h max); confirm with `sinfo`
#SBATCH --account=plgshl26-gpu-gh200       # Helios GPU grant; CONFIRM with `hpc-grants`

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # activates the aarch64 FM venv
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
: "${MODEL:?set MODEL=mantis8m (or mantisv2 / utica / moment-small)}"

echo "[$(date)] HELIOS GH200 extract: $MODEL  on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
LIM=""; [ -n "${LIMIT:-}" ] && LIM="--limit $LIMIT"   # smoke-test cap

echo "[$(date)] prefetch $MODEL weights (serial)"
python notebooks/mdaniol/modeling/extract_embeddings.py --model "$MODEL" --prefetch
for V in ${VARIANTS:-V0 V1 V2}; do
  echo "[$(date)] extract $MODEL $V ${LIMIT:+(LIMIT=$LIMIT smoke)}"
  python notebooks/mdaniol/modeling/extract_embeddings.py --model "$MODEL" --variant "$V" $LIM \
      --data-dir "$PROJECT_DIR/dataset_parquet" --emb-dir "$PROJECT_DIR/embeddings" \
      --device cuda --chunk-size "${CHUNK:-8000}" --tf-batch "${TFB:-512}"
done
echo "[$(date)] done. If full run: bash notebooks/mdaniol/hpc/stage_to_group.sh emb"
