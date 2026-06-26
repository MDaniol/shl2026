#!/bin/bash
# Helios CPU job: produce the FINAL submission from cached embeddings on Helios group storage.
# Phase B — fit on User-1 + val[FIT], per-class calibrate on val[TUNE], predict the unlabelled
# test, write a validated 92726x500 matrix + SUBMISSIONS.md + MLflow (rule §8). CPU (LightGBM).
# Helios variant of submit.sbatch (Athena queue dead; utica_V2 embeddings live on Helios pr3).
#
#   VERSION=v2 EMB=utica_V2 sbatch notebooks/mdaniol/hpc/submit_helios.sbatch
#   (defaults below already build the new best base: utica_V2 -> v2)
#
#SBATCH --job-name=shl-submit
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/submit_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
# stage from GROUP storage (persistent); $SCRATCH purges -> link_data must NOT default to it
export RAW_SRC="${RAW_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet}"
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling
FEAT="$PROJECT_DIR/dataset_parquet_features"
SPLIT="$M/artifacts/val_split_temporal.npy"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios}"

EMB="${EMB:-utica_V2}"
VERSION="${VERSION:-v2}"
HELDOUT="${HELDOUT:-0.8157 (temporal lock; bake-off v2)}"   # utica_V2 default; override per EMB

# guardrail gate first (don't ship from broken code)
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }

# the submission calibrates on val[TUNE] of the temporal split — ensure it exists
[ -f "$SPLIT" ] || python $M/split.py --scheme temporal --embargo 100 --feat-dir "$FEAT" --out "$SPLIT"

# require the chosen embedding's TEST split (the submission predicts test/all)
[ -f "$PROJECT_DIR/embeddings/$EMB/test__all.npy" ] || {
  echo "[$(date)] FATAL: embeddings/$EMB/test__all.npy missing — stage $EMB (incl test) to group storage" >&2
  exit 1; }

echo "[$(date)] building submission ($VERSION, emb=$EMB, heldout=$HELDOUT) on $(hostname)"
python $M/submit_fusion.py --version "$VERSION" --emb "$EMB" --heldout "$HELDOUT" \
    --feat-dir "$FEAT" --emb-root "$PROJECT_DIR/embeddings"
echo "[$(date)] done. Submission + SUBMISSIONS.md + MLflow written; archive the .txt + git add/commit/push it."
