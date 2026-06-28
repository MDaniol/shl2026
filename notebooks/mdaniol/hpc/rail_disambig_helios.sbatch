#!/bin/bash
# E-RAIL gated Train↔Subway disambiguator (CPU): binary head re-splits the champion's within-pair
# {Train,Subway} mass; β on TUNE, TEST locked once, paired-significance vs champion. Data-justified by
# ROBUSTNESS_DIAG (Subway↔Train errors 80–99% top-2-recoverable). Reuses cached embeddings + 520 HC.
#
#   EMBS=utica_V2,mantisv2_V1 BIN_EMB=utica_V2 sbatch notebooks/mdaniol/hpc/rail_disambig_helios.sbatch
#
#SBATCH --job-name=shl-rail
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/rail_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
EMBS="${EMBS:-utica_V2,mantisv2_V1}"
BIN_EMB="${BIN_EMB:-utica_V2}"
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios}"
SPLIT="$PROJECT_DIR/notebooks/mdaniol/modeling/artifacts/val_split_temporal.npy"
M="$PROJECT_DIR/notebooks/mdaniol/modeling"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
[ -f "$SPLIT" ] || python "$M/split.py" --scheme temporal --embargo 100 \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --out "$SPLIT"

echo "[$(date)] E-RAIL disambiguator EMBS=$EMBS BIN_EMB=$BIN_EMB on $(hostname)"
python "$M/rail_disambig.py" --embs "$EMBS" --bin-emb "$BIN_EMB" --split "$SPLIT" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --emb-root "$PROJECT_DIR/embeddings"
echo "[$(date)] done — see RAIL_DISAMBIG_RESULTS.md + MLflow run rail_${EMBS}."
