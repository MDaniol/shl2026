#!/bin/bash
# Athena job: VehicleExpert subset corrector (default V4 {Car,Bus,Train,Subway}) on the
# calibrated MOMENT-fusion base, temporal split. Gated KEEP/DISABLE. CPU; cached data.
#   SUBSET=V4|SV5|R2 sbatch notebooks/mdaniol/hpc/vehicle_expert.sbatch
#SBATCH --job-name=shl-vexpert
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/vexpert_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account=plgshl26-gpu-a100
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"; bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling; FEAT="$PROJECT_DIR/dataset_parquet_features"
SPLIT="$M/artifacts/val_split_temporal.npy"
[ -f "$SPLIT" ] || python $M/split.py --scheme temporal --embargo 100 --feat-dir "$FEAT" --out "$SPLIT"
echo "[$(date)] VehicleExpert ${SUBSET:-V4}"
python $M/vehicle_expert.py --subset "${SUBSET:-V4}" --feat-dir "$FEAT" --emb-root "$PROJECT_DIR/embeddings" --split "$SPLIT"
echo "[$(date)] done -> notebooks/mdaniol/VEHICLE_EXPERT_RESULTS.md"
