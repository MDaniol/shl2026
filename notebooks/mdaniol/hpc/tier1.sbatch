#!/bin/bash
# Athena job: Tier-1 macro-F1 post-hoc ablation (calibration / logit-adjust / MLLS prior
# adaptation) on the MOMENT-fusion base, honest temporal split. CPU-bound; reuses cached
# features + embeddings. Writes TIER1_RESULTS.md + MLflow (tier1_*).
#
#   EMB=moment-small_V1 sbatch notebooks/mdaniol/hpc/tier1.sbatch
#
#SBATCH --job-name=shl-tier1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/tier1_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1            # Athena is GPU-only (job is CPU-bound)
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling
SPLIT="$M/artifacts/val_split_temporal.npy"

python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
[ -f "$SPLIT" ] || python $M/split.py --scheme temporal --embargo 100 \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --out "$SPLIT"

echo "[$(date)] Tier-1 post-hoc ablation (emb=${EMB:-moment-small_V1})"
python $M/tier1_experiment.py --emb "${EMB:-moment-small_V1}" --split "$SPLIT" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --emb-root "$PROJECT_DIR/embeddings"
echo "[$(date)] done — see TIER1_RESULTS.md + MLflow"
