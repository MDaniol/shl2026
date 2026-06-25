#!/bin/bash
# Helios (CPU, aarch64) version of tier1.sbatch. Tier-1 is pure CPU (LightGBM + NumPy on
# CACHED features/embeddings) — no GPU, no torch — so it runs on Helios's Grace (ARM) CPUs.
#
# HELIOS DIFFERENCES vs Athena/Ares (see docs/CLUSTER.md):
#   * aarch64 (ARM): build a FRESH venv ON Helios — the x86 Athena/Ares venv will NOT work.
#   * storage is separate (pr3): rsync features + moment-small_V1 embeddings to Helios group
#     storage first. $PLG_GROUPS_STORAGE on Helios resolves to the pr3 path.
#   * CPU grant: plgshl26-cpu / plgrid (5000 CPU-h, 72h walltime). Confirm with `hpc-grants`.
#   Full setup: notebooks/mdaniol/hpc/HELIOS_SETUP.md
#
#   sbatch notebooks/mdaniol/hpc/tier1_helios.sbatch
#
#SBATCH --job-name=shl-tier1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/tier1_helios_%j.out
#SBATCH --partition=plgrid                 # Helios CPU partition (72h); confirm with `sinfo`
#SBATCH --account=plgshl26-cpu             # Helios CPU grant; CONFIRM with `hpc-grants`

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# data on Helios GROUP storage (pr3) — populated by the one-time rsync (see HELIOS_SETUP.md)
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # activates the aarch64 venv built on Helios
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling
SPLIT="$M/artifacts/val_split_temporal.npy"

python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
[ -f "$SPLIT" ] || python $M/split.py --scheme temporal --embargo 100 \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --out "$SPLIT"

echo "[$(date)] Tier-1 post-hoc on HELIOS (CPU, aarch64) (emb=${EMB:-moment-small_V1})"
python $M/tier1_experiment.py --emb "${EMB:-moment-small_V1}" --split "$SPLIT" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --emb-root "$PROJECT_DIR/embeddings"
echo "[$(date)] done — see TIER1_RESULTS.md + MLflow"
