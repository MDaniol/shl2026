#!/bin/bash
# Athena job: produce the FINAL submission (MOMENT-fusion, the first >=0.80 recipe).
# Phase B — trains on User-1 + ALL validation, predicts the unlabelled test, writes a
# validated 92726x500 matrix + logs SUBMISSIONS.md. CPU; reuses cached features + the
# cached MOMENT-small_V1 test embeddings.
#
#   VERSION=v1 EMB=moment-small_V1 sbatch notebooks/mdaniol/hpc/submit.sbatch
#
#SBATCH --job-name=shl-submit
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/submit_%j.out
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

# guardrail gate first (don't ship from broken code)
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }

# the submission calibrates on val[TUNE] of the temporal split — ensure it exists
[ -f "$SPLIT" ] || python $M/split.py --scheme temporal --embargo 100 --feat-dir "$FEAT" --out "$SPLIT"

echo "[$(date)] building submission (${VERSION:-v1}, emb=${EMB:-moment-small_V1})"
python $M/submit_fusion.py --version "${VERSION:-v1}" --emb "${EMB:-moment-small_V1}" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --emb-root "$PROJECT_DIR/embeddings"
echo "[$(date)] done. Submission + SUBMISSIONS.md written; archive the file + git add/commit/push it."
