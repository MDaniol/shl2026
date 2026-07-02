#!/bin/bash
# FINAL all-class v5 confusion matrix (CPU). Re-fits Lane A holding out the collaborator's target_holdout
# (clean for Lane B, all 8 classes), blends with the pre-locked w, and renders the all-class v5 CM.
# No vision-lane re-run, no Athena — reuses cached embeddings/features + env_mdaniol.sh.
#
#   sbatch notebooks/mdaniol/hpc/final_cm_helios.sbatch
#
#SBATCH --job-name=shl-finalcm
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/finalcm_%j.out
#SBATCH --partition=plgrid
#SBATCH --account=plgshl26-cpu

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
EMBS="${EMBS:-utica_V2,mantisv2_V1}"
W="${W:-0.575}"
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios}"
SPLIT="$PROJECT_DIR/notebooks/mdaniol/modeling/artifacts/val_split_temporal.npy"
PDUSZA_DIR="${PDUSZA_DIR:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/pdusza_results}"
HIS_HOLDOUT="$PDUSZA_DIR/02_adaptation_selection/selected_untouched_target_holdout/predictions.npz"
M="$PROJECT_DIR/notebooks/mdaniol/modeling"
FIGDIR="$PROJECT_DIR/notebooks/mdaniol/figures"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs "$FIGDIR"

python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
[ -f "$SPLIT" ] || python "$M/split.py" --scheme temporal --embargo 100 \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --out "$SPLIT"
[ -f "$HIS_HOLDOUT" ] || { echo "FATAL: his holdout npz missing: $HIS_HOLDOUT" >&2; exit 1; }

echo "[$(date)] FINAL CM: re-fit Lane A holding out his target_holdout; blend w=$W; all-class v5 CM"
python "$M/final_cm.py" --embs "$EMBS" --split "$SPLIT" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --emb-root "$PROJECT_DIR/embeddings" \
    --his-holdout "$HIS_HOLDOUT" --w "$W" --out-dir "$FIGDIR" --mlflow \
    || { echo "final_cm failed" >&2; exit 1; }

echo "[$(date)] done — see $FIGDIR/cm_v5_final.png + FINAL_CM_RESULTS.md + MLflow final_cm_v5:"
ls -la "$FIGDIR"/cm_*_final.png "$FIGDIR"/FINAL_CM_RESULTS.md
