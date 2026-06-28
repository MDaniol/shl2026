#!/bin/bash
# Helios CPU job: FINAL submission = calibrated soft-VOTE of the top-2 FMs (E-VOTE-01, 0.8342).
# Phase B over cached embeddings on Helios group storage. Writes a validated 92726x500 matrix
# + SUBMISSIONS.md row + MLflow (rule §8). CPU (2 LightGBM fusion fits + NumPy vote).
#
#   VERSION=v3 EMBS=utica_V2,mantisv2_V1 sbatch notebooks/mdaniol/hpc/submit_vote_helios.sbatch
#   (defaults below already ship the kept vote)
#
#SBATCH --job-name=shl-submit-vote
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/submitvote_%j.out
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

EMBS="${EMBS:-utica_V2,mantisv2_V1}"
VERSION="${VERSION:-v3}"
HELDOUT="${HELDOUT:-0.8342 (temporal lock; E-VOTE-01 weighted+recal)}"

# guardrail gate first
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
[ -f "$SPLIT" ] || python $M/split.py --scheme temporal --embargo 100 --feat-dir "$FEAT" --out "$SPLIT"

# require each FM's TEST split (the vote predicts test/all)
for e in ${EMBS//,/ }; do
  [ -f "$PROJECT_DIR/embeddings/$e/test__all.npy" ] || {
    echo "[$(date)] FATAL: embeddings/$e/test__all.npy missing — stage $e (incl test) to group storage" >&2
    exit 1; }
done

echo "[$(date)] building VOTE submission ($VERSION, embs=$EMBS) on $(hostname)"
python $M/submit_vote.py --version "$VERSION" --embs "$EMBS" --heldout "$HELDOUT" \
    --decision "${DECISION:-mult}" \
    --feat-dir "$FEAT" --emb-root "$PROJECT_DIR/embeddings" \
    --save-proba "notebooks/mdaniol/preds_test_${VERSION}_vote.npy"   # (92726,8) for cross-team combine
echo "[$(date)] done. Vote submission + SUBMISSIONS.md + MLflow written; archive the .txt + git add/commit/push."
