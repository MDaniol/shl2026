#!/bin/bash
# Helios GH200 (aarch64) array job: extract ALL FMs for the temporal bake-off, fully logged.
# One model per array task (its variants run serially -> no same-model prefetch race). Writes
# embeddings DIRECTLY to group storage (persistent; $SCRATCH auto-purges). Each (model,variant)
# extraction logs to MLflow (params + windows/sec + a manifest) for reproducibility.
#
#   sbatch notebooks/mdaniol/hpc/extract_all_helios.sbatch
#   then bake-off (x86 CPU): sbatch notebooks/mdaniol/hpc/probe_fusion_ares.sbatch
#
# Prereq: aarch64 FM env built (setup_env_helios.sh + add_fm_deps_helios.sh) and raw data on
# Helios group storage. See HELIOS_SETUP.md §GPU.
#
#SBATCH --job-name=shl-extract-all
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/extractall_%A_%a.out
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgshl26-gpu-gh200

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# one model per task; variants serial within the task (V0/V1/V2; moment-small has no V0)
MODELS=(mantis8m mantisv2 utica moment-small)
VARS=("V0 V1 V2" "V0 V1 V2" "V0 V1 V2" "V1 V2")
M="${MODELS[$SLURM_ARRAY_TASK_ID]}"
VARIANTS="${VARS[$SLURM_ARRAY_TASK_ID]}"

GROUP="$PLG_GROUPS_STORAGE/plggmhealth/shl2026"
DATA="$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet"   # raw (rsync'd to Helios pr3)
EMB="$GROUP/embeddings"                                   # persistent embedding cache
# MLflow file store on group storage (the group MLflow server is Athena-side / unreachable here)
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$GROUP/mlruns-helios}"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # aarch64 venv + aarch64 uv on PATH
cd "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs "$EMB"

echo "[$(date)] HELIOS GH200 extract $M [$VARIANTS] -> $EMB (MLflow: $MLFLOW_TRACKING_URI)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

python notebooks/mdaniol/modeling/extract_embeddings.py --model "$M" --prefetch   # serial, this task only
for V in $VARIANTS; do
  echo "[$(date)] extract $M $V"
  python notebooks/mdaniol/modeling/extract_embeddings.py --model "$M" --variant "$V" \
      --data-dir "$DATA" --emb-dir "$EMB" --device cuda --chunk-size "${CHUNK:-8000}" --tf-batch "${TFB:-512}"
done
echo "[$(date)] done $M. Bake-off (x86 CPU): sbatch notebooks/mdaniol/hpc/probe_fusion_ares.sbatch"
