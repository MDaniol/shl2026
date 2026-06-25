#!/bin/bash
# Ares (CPU-only) version of tier1.sbatch. Tier-1 is pure CPU — LightGBM + NumPy on CACHED
# features + embeddings; it never touches a GPU or the FM — so it runs natively on Ares.
#
# ========================= ONE-TIME ARES SETUP (do once) =========================
# Ares $SCRATCH is separate from Athena's, and your features/embeddings live on Athena
# scratch — so make them reachable on Ares, and create a CPU venv:
#
#   # 1. repo + venv on Ares (slim — Tier-1 needs no torch/FM):
#   git clone <repo> ~/shl2026 && cd ~/shl2026
#   module load python                       # or the Ares Python module you use
#   python -m venv $SCRATCH/venvs/shl2026 && source $SCRATCH/venvs/shl2026/bin/activate
#   pip install -e . lightgbm scikit-learn pandas pyarrow mlflow pytest   # CPU deps only
#   #   (or recreate the full env from the lock: uv sync  — installs torch-cpu too, heavier)
#
#   # 2. make features + the MOMENT-small_V1 embeddings reachable on Ares. Easiest: copy
#   #    them to GROUP storage (shared across clusters), then point FEAT_SRC/EMB_SRC at it:
#   #      (run from Athena)  rsync -a $SCRATCH/shl2026/dataset_parquet_features \
#   #            $PLG_GROUPS_STORAGE/plggmhealth/shl2026/
#   #      (run from Athena)  rsync -a $SCRATCH/shl2026/embeddings/moment-small_V1 \
#   #            $PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings/
#   #    then export before sbatch (or set in env):  see FEAT_SRC/EMB_SRC below.
# =================================================================================
#
# Run:  sbatch notebooks/mdaniol/hpc/tier1_ares.sbatch
#   adjust the account/partition for your CPU grant, or override on the CLI:
#   sbatch -A plgshl26-cpu -p plgrid notebooks/mdaniol/hpc/tier1_ares.sbatch
#
#SBATCH --job-name=shl-tier1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/tier1_ares_%j.out
#SBATCH --partition=plgrid                 # Ares CPU partition (no GPU); adjust if needed
#SBATCH --account=plgshl26-cpu             # CPU grant; CONFIRM your Ares account name

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Point the data symlinks at GROUP storage (shared with Ares) instead of Athena scratch.
# Override these on the command line if you copied the data somewhere else.
export FEAT_SRC="${FEAT_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet_features}"
export EMB_SRC="${EMB_SRC:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings}"

source "$PROJECT_DIR/notebooks/mdaniol/hpc/env_mdaniol.sh"   # parameterized; venv = $SCRATCH/venvs/shl2026
cd "$PROJECT_DIR"
bash "$PROJECT_DIR/notebooks/mdaniol/hpc/link_data.sh" "$PROJECT_DIR"
mkdir -p notebooks/mdaniol/modeling/logs
M=notebooks/mdaniol/modeling
SPLIT="$M/artifacts/val_split_temporal.npy"

python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }
[ -f "$SPLIT" ] || python $M/split.py --scheme temporal --embargo 100 \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --out "$SPLIT"

echo "[$(date)] Tier-1 post-hoc ablation on ARES (CPU) (emb=${EMB:-moment-small_V1})"
python $M/tier1_experiment.py --emb "${EMB:-moment-small_V1}" --split "$SPLIT" \
    --feat-dir "$PROJECT_DIR/dataset_parquet_features" --emb-root "$PROJECT_DIR/embeddings"
echo "[$(date)] done — see TIER1_RESULTS.md + MLflow"
