#!/bin/bash
# Athena job: Step 3 — embedding L2/PCA + linear heads over cached FM sets.
# Cheap (linear heads on PCA-reduced dims); reuses cached embeddings, no FM
# re-inference. Writes EMB_PCA_RESULTS.md + MLflow (experiment 'mdaniol').
#
#SBATCH --job-name=shl-embpca
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/embpca_%j.out
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

# cached embedding sets to evaluate (override via EMBS="...")
EMBS="${EMBS:-mantisv2_V0 mantisv2_V1 mantisv2_V2 utica_V1 utica_V2 moment-small_V1 moment-small_V2}"
for E in $EMBS; do
  if [ -d "$PROJECT_DIR/embeddings/$E" ]; then
    echo "[$(date)] emb-pca $E"
    python $M/emb_pca_heads.py --emb "$E" --pcas "${PCAS:-64,128}" \
        --feat-dir "$FEAT" --emb-root "$PROJECT_DIR/embeddings"
  else
    echo "[$(date)] SKIP $E (no cached embeddings)"
  fi
done
echo "[$(date)] done — see notebooks/mdaniol/EMB_PCA_RESULTS.md + MLflow"
