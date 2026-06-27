#!/bin/bash
# Helios GH200: train the cross-channel lightweight heads (E-HEAD-01) on per-channel embeddings.
# Small PyTorch heads over (n,C,d) _pc embeddings — torch is in the aarch64 env, so run on GH200
# (--device cuda; the heads are tiny, the GPU just makes the seed sweep fast). Reads _pc embeddings
# + handcrafted features from group storage. MLflow-tracked (rule §8).
#
#   sbatch notebooks/mdaniol/hpc/head_xchannel_helios.sbatch              # utica_V1_pc, 4 heads, 3 seeds
#   EMB_PC=mantisv2_V1_pc sbatch ...                                      # other FM
#
# PREREQ: extract_per_channel_helios.sbatch has produced embeddings/<EMB_PC>/ on group storage.
#
#SBATCH --job-name=shl-head
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/head_%j.out
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgshl26-gpu-gh200

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # aarch64 venv (has torch)
cd "$PROJECT_DIR"
# stage from GROUP storage (persistent); $SCRATCH purges -> link_data must NOT default to it
export RAW_SRC="${RAW_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet}"
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling
FEAT="$PROJECT_DIR/dataset_parquet_features"
SPLIT="$M/artifacts/val_split_temporal.npy"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://$PLG_GROUPS_STORAGE/plggmhealth/shl2026/mlruns-helios}"

EMB_PC="${EMB_PC:-utica_V1_pc}"
HEADS="${HEADS:-mean,concat,multistat,se}"
SEEDS="${SEEDS:-0,1,2}"
BAR="${BAR:-0.8342}"

# sklearn is needed (StandardScaler); ensure present in the aarch64 env (no-op if already there)
python -c "import sklearn" 2>/dev/null || uv pip install scikit-learn
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR" || true
[ -f "$SPLIT" ] || python $M/split.py --scheme temporal --embargo 100 --feat-dir "$FEAT" --out "$SPLIT"

# require the per-channel embeddings
for sp in train validation; do for loc in Bag Hips Torso Hand; do
  [ -f "$PROJECT_DIR/embeddings/$EMB_PC/${sp}__${loc}.npy" ] || {
    echo "[$(date)] FATAL: embeddings/$EMB_PC/${sp}__${loc}.npy missing — run extract_per_channel_helios first" >&2
    exit 1; }
done; done

echo "[$(date)] cross-channel head: emb=$EMB_PC heads=$HEADS seeds=$SEEDS bar=$BAR on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python $M/head_xchannel.py --emb-pc "$EMB_PC" --heads "$HEADS" --seeds "$SEEDS" --bar "$BAR" \
    --emb-root "$PROJECT_DIR/embeddings" --feat-dir "$FEAT" --split "$SPLIT" --device cuda \
    --out "notebooks/mdaniol/HEAD_RESULTS_${EMB_PC}.md"   # per-FM file -> no clobber across EMB_PC runs
echo "[$(date)] done. Read HEAD_RESULTS_${EMB_PC}.md + MLflow; if SE KEEP, it's a paper method."
