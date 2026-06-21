#!/bin/bash
# Athena job: Step 7 — branch-level probability fusion (global_hc + freqmag [+ fm]).
# CPU; reuses cached features/embeddings. Leakage-safe stacking (base on FIT,
# stacker on TUNE, confirm on TEST). Writes FUSION_RESULTS.md + MLflow.
#   add the FM branch:  EMB=mantisv2_V1 sbatch ... fusion.sbatch
#
#SBATCH --job-name=shl-fusion
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/fusion_%j.out
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
[ -f "$M/artifacts/val_split.npy" ] || python $M/split.py --feat-dir "$FEAT"

echo "[$(date)] branch fusion (emb=${EMB:-none})"
python $M/fusion_experiment.py --feat-dir "$FEAT" --emb-root "$PROJECT_DIR/embeddings" \
    ${EMB:+--emb "$EMB"}
echo "[$(date)] done — see notebooks/mdaniol/FUSION_RESULTS.md + MLflow"
