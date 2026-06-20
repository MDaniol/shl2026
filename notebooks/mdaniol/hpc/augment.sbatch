#!/bin/bash
# Athena job: train-time augmentation experiment (Branch A).
#   augment_features.py x4 -> Branch-A augmented User-1 TRAIN features
#                             (scale/jitter/time-warp; NO rotation — invariant)
#   train_split.py --aug-train-dir -> Phase A reports the +AUG held-out delta,
#                             KEEP/DROP per AUGMENTATION_STRATEGY.md §8.
# Run AFTER train_split.sbatch (reuses artifacts/val_split.npy).
# To ship an augmented submission, re-run train_split.py --use-aug-for-submission
# only if the +AUG row improved held-out macro-F1.
#
#SBATCH --job-name=shl-aug
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/aug_%j.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1            # Athena is GPU-only
#SBATCH --account=plgshl26-gpu-a100

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"   # group paths + UV_CACHE_DIR
source "${SHL_VENV:-$SCRATCH/venvs/shl2026}/bin/activate" # YOUR env (team lock + FM add-ons)
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"   # ensure data symlinks
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling
RAW="$PROJECT_DIR/dataset_parquet"
FEAT="$PROJECT_DIR/dataset_parquet_features"
# heavy augmented features -> $SCRATCH (keep $HOME/group clean)
AUG="${AUG_DIR:-$SCRATCH/shl2026/dataset_parquet_features_aug}/train"
COPIES="${COPIES:-1}"

[ -f "$M/artifacts/val_split.npy" ] || python $M/split.py --feat-dir "$FEAT"

for LOC in Bag Hips Torso Hand; do
  echo "[$(date)] augment train/$LOC (Branch A, $COPIES copies)"
  python $M/augment_features.py \
      --input  "$RAW/train/$LOC.parquet" \
      --output "$AUG/$LOC.parquet" \
      --branch A --copies "$COPIES" --seed 0 --chunk-size 8000
done

echo "[$(date)] train_split with +AUG held-out comparison"
python $M/train_split.py --feat-dir "$FEAT" --aug-train-dir "$AUG"

echo "[$(date)] done — read the '+AUG' row + 'augmentation effect' line in the log;"
echo "          if positive, re-run: python $M/train_split.py --feat-dir $FEAT \\"
echo "                         --aug-train-dir $AUG --use-aug-for-submission"
