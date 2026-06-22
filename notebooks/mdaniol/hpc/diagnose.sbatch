#!/bin/bash
# Athena job: per-class + confusion diagnostic for MOMENT-fusion under the temporal
# (conservative) split. Traceable: logs to logs/diagnose_%j.out. Answers "which classes
# drag macro-F1 and what is Bus/vehicle confused with -> is a V4 corrector worth it?"
# CPU; reuses cached features + MOMENT-small_V1 embeddings.
#
#   EMB=moment-small_V1 sbatch notebooks/mdaniol/hpc/diagnose.sbatch
#
#SBATCH --job-name=shl-diagnose
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/diagnose_%j.out
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
FEAT="$PROJECT_DIR/dataset_parquet_features"
SPLIT="$M/artifacts/val_split_temporal.npy"

# ensure the conservative temporal split exists (rebuild if missing)
[ -f "$SPLIT" ] || python $M/split.py --scheme temporal --embargo 100 --feat-dir "$FEAT" --out "$SPLIT"

echo "[$(date)] per-class + confusion diagnostic (emb=${EMB:-moment-small_V1}, temporal split)"
python $M/diagnose_perclass.py --emb "${EMB:-moment-small_V1}" \
    --feat-dir "$FEAT" --emb-root "$PROJECT_DIR/embeddings" --split "$SPLIT"
echo "[$(date)] done — see this log for per-class F1 + confusion."
