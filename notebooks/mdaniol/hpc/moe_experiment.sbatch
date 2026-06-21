#!/bin/bash
# Athena job: location mixture-of-experts experiment (plan Steps 4-6).
# Global + Bag/Hips/Torso experts + router -> oracle/uniform/hard/soft/β configs,
# scored on TUNE+TEST -> MOE_RESULTS.md + MLflow (shl2026.track). CPU/LightGBM
# (reuses cached features/embeddings; Athena forces a GPU we don't use).
#
#   REP=handcrafted (default) | emb | fusion ; EMB=<model>_<variant> for emb/fusion
#
#SBATCH --job-name=shl-moe
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/moe_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1            # Athena is GPU-only (job is CPU-bound)
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # single env source: group env + my venv + caches
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"   # ensure data symlinks
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling
FEAT="$PROJECT_DIR/dataset_parquet_features"

# the MoE reuses the SAME split as train_split; build it if missing
[ -f "$M/artifacts/val_split.npy" ] || python $M/split.py --feat-dir "$FEAT"

REP="${REP:-handcrafted}"
echo "[$(date)] MoE experiment rep=$REP emb=${EMB:-none}"
python $M/moe_experiment.py --rep "$REP" ${EMB:+--emb "$EMB"} \
    --feat-dir "$FEAT" --emb-root "$PROJECT_DIR/embeddings"
echo "[$(date)] done — see notebooks/mdaniol/MOE_RESULTS.md + MLflow (experiment 'mdaniol')"
