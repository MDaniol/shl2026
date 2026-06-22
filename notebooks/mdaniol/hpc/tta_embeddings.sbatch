#!/bin/bash
# Athena GPU job: rotation-TTA embeddings (rotation-marginalized FM representation).
# Pushes K randomly-reoriented copies of each window through the FROZEN FM and stores the
# MEAN embedding -> a drop-in alternative embedding set the existing probe/submit consume
# via --emb (no downstream changes). Targets the unknown test orientation; helps the
# axis-based FM input (V1), no-op for magnitude handcrafted features.
#
#   MODEL=moment-small VARIANT=V1 ROT=gravity_aware TTAK=4 SPLITS=train,validation \
#     sbatch notebooks/mdaniol/hpc/tta_embeddings.sbatch
# Output dir: embeddings/<model>_<variant>_tta<K><rot>/  (e.g. moment-small_V1_tta4gravity_aware)
#
#SBATCH --job-name=shl-tta
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00          # K x the base extraction — train is ~784k windows
#SBATCH --output=notebooks/mdaniol/modeling/logs/tta_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs

echo "[$(date)] TTA embeddings: ${MODEL:-moment-small}/${VARIANT:-V1} rot=${ROT:-gravity_aware} K=${TTAK:-4} splits=${SPLITS:-train,validation}"
python notebooks/mdaniol/modeling/extract_embeddings.py \
    --model "${MODEL:-moment-small}" --variant "${VARIANT:-V1}" \
    --rotation "${ROT:-gravity_aware}" --tta-k "${TTAK:-4}" --max-tilt-deg "${TILT:-30}" \
    --splits "${SPLITS:-train,validation}" \
    --data-dir "$PROJECT_DIR/dataset_parquet" --emb-dir "$PROJECT_DIR/embeddings" \
    --device cuda --chunk-size "${CHUNK:-4000}" --tf-batch "${TFB:-512}"
echo "[$(date)] done. Probe it: point probe_fusion/submit --emb at the _tta dir vs the original."
