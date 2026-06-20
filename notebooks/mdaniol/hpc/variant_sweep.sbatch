#!/bin/bash
# Athena GPU array: input-VARIANT sweep for the FM branch.
# V0 (raw axes) underperformed locally (MantisV2 internal z-score discards the
# gravity/magnitude amplitude). V1 (body-acc) and V2 (orientation-invariant
# magnitudes) test whether a better input representation lets the FM branch beat
# the handcrafted baseline. One (model:variant) per array task -> extract + probe.
#
#SBATCH --job-name=shl-vsweep
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/vsweep_%A_%a.out
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgshl26-gpu-a100

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026}"
VENV="${VENV:-$PROJECT_DIR/SHL-local}"

# (model:variant) combos. V3 (16 ch) needs a channel adapter (>10) — excluded.
COMBOS=(mantisv2:V1 mantisv2:V2 utica:V1 utica:V2 moment-small:V1 moment-small:V2)
C="${COMBOS[$SLURM_ARRAY_TASK_ID]}"; M="${C%:*}"; V="${C#*:}"

source "$VENV/bin/activate"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_DIR"; mkdir -p notebooks/mdaniol/modeling/logs

echo "[$(date)] $M variant $V"
python notebooks/mdaniol/modeling/extract_embeddings.py --model "$M" --variant "$V" \
    --data-dir "$PROJECT_DIR/dataset_parquet" --emb-dir "$PROJECT_DIR/embeddings" \
    --device cuda --chunk-size 8000 --tf-batch 512
python notebooks/mdaniol/modeling/probe_fusion.py --emb "${M}_${V}" \
    --emb-root "$PROJECT_DIR/embeddings" --feat-dir "$PROJECT_DIR/dataset_parquet_features"
echo "[$(date)] done $M:$V"
