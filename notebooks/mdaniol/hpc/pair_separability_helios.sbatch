#!/bin/bash
# A1 diagnostic (CPU): can a linear probe on the submission rep separate Train↔Subway / Car↔Bus on
# held-out TEST? Decides whether the hard-pair program (B1 Fisher feature / B2 OvO specialist) is worth
# building. Pure CPU over cached embeddings + 520 handcrafted; reuses env_mdaniol.sh + group storage.
#
#   EMB=utica_V2 REP=fusion sbatch notebooks/mdaniol/hpc/pair_separability_helios.sbatch
#
#SBATCH --job-name=shl-pairsep
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/pairsep_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
EMB="${EMB:-utica_V2}"
REP="${REP:-fusion}"
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios}"
SPLIT="$PROJECT_DIR/notebooks/mdaniol/modeling/artifacts/val_split_temporal.npy"
M="$PROJECT_DIR/notebooks/mdaniol/modeling"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

# pre-HPC gate
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
[ -f "$SPLIT" ] || { echo "[$(date)] FATAL: split $SPLIT missing" >&2; exit 1; }

echo "[$(date)] A1 pair-separability EMB=$EMB REP=$REP on $(hostname)"
python "$M/pair_separability.py" --emb "$EMB" --rep "$REP" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" \
    --emb-root "$PROJECT_DIR/embeddings" --split "$SPLIT" \
    --out "notebooks/mdaniol/PAIR_SEPARABILITY_${EMB}_${REP}.md"   # rep-specific (parallel-safe)
echo "[$(date)] done — see PAIR_SEPARABILITY.md + MLflow run pairsep_${EMB}_${REP}."
