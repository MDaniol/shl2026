#!/bin/bash
# Helios/Ares CPU job: mean-pool a PER-CHANNEL embedding dir (<tag>_pc, files (n,C,d)) into a
# mean-pooled VOTER dir (<tag>, files (n,d)) on GROUP storage. The channel-mean is NUMBER-IDENTICAL
# to a non-per-channel extraction (extract_embeddings: `emb = pooled if per_channel else pooled.mean(1)`),
# so this reuses dinov2_V2_pc (extracted for the cross-channel head) as a fusion VOTER without burning
# GPU hours. Gated by the guardrail suite; logged; writes onto persistent group storage.
#
#   SRC_TAG=dinov2_V2_pc DST_TAG=dinov2_V2 sbatch notebooks/mdaniol/hpc/pool_emb_helios.sbatch
#   (defaults below already do dinov2_V2_pc -> dinov2_V2 for the E-FMDIV N-way vote)
#
#SBATCH --job-name=shl-pool
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/pool_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
SRC_TAG="${SRC_TAG:-dinov2_V2_pc}"
DST_TAG="${DST_TAG:-dinov2_V2}"

# pre-HPC gate (locks the channel-mean number-identity + the rest of the suite)
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }

[ -d "$EMB_SRC/$SRC_TAG" ] || { echo "[$(date)] FATAL: $EMB_SRC/$SRC_TAG missing" >&2; exit 1; }
echo "[$(date)] pooling $SRC_TAG -> $DST_TAG under $EMB_SRC on $(hostname)"
python notebooks/mdaniol/modeling/pool_per_channel.py \
    --src "$EMB_SRC/$SRC_TAG" --dst "$EMB_SRC/$DST_TAG"
echo "[$(date)] done — $DST_TAG ready as a fusion voter (number-identical to non-pc extraction)."
