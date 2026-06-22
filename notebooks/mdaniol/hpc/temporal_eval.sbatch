#!/bin/bash
# Athena job: CONSERVATIVE (trustworthy) re-evaluation — the over-claim check.
# blocked split (FIT/TEST share sessions) gives an optimistic ~0.867; this rebuilds
# everything under a temporal+embargo split (held-out period, no shared journey) and
# the cross-user lower bound, so we trust the number we submit on. CPU/LightGBM;
# reuses cached features/embeddings.
#
#SBATCH --job-name=shl-temporal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=notebooks/mdaniol/modeling/logs/temporal_%j.out
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
SPLIT="$M/artifacts/val_split_temporal.npy"
EMB="${EMB:-moment-small_V1}"   # best FM from the bake-off; override if needed

# --- pre-HPC guardrail gate (fail fast; don't spend compute on broken code) ---
echo "[$(date)] guardrail gate"
python -m pytest tests/test_guardrails.py -q || { echo "GUARDRAIL GATE FAILED" >&2; exit 1; }

# --- 1. conservative temporal + embargo split --------------------------------
echo "[$(date)] build temporal+embargo split"
python $M/split.py --scheme temporal --embargo 100 --feat-dir "$FEAT" --out "$SPLIT"

# --- 2. cross-user LOWER BOUND (train User1 -> test Users 2&3) ----------------
echo "[$(date)] cross-user baseline (User1 -> Users 2&3)"
python $M/train_baseline.py --feat-dir "$FEAT"

# --- 3. conservative global + soft-MoE (handcrafted) --------------------------
echo "[$(date)] temporal MoE (handcrafted)"
python $M/moe_experiment.py --rep handcrafted --split "$SPLIT" \
    --feat-dir "$FEAT" --emb-root "$PROJECT_DIR/embeddings" \
    --out "$PROJECT_DIR/notebooks/mdaniol/MOE_RESULTS_temporal_hc.md"

# --- 4. conservative MOMENT-fusion + soft-MoE --------------------------------
echo "[$(date)] temporal MoE (handcrafted ⊕ $EMB)"
python $M/moe_experiment.py --rep fusion --emb "$EMB" --split "$SPLIT" \
    --feat-dir "$FEAT" --emb-root "$PROJECT_DIR/embeddings" \
    --out "$PROJECT_DIR/notebooks/mdaniol/MOE_RESULTS_temporal_moment.md"

echo "[$(date)] DONE. Compare:"
echo "  optimistic (blocked): notebooks/mdaniol/MOE_RESULTS.md"
echo "  conservative (temporal): notebooks/mdaniol/MOE_RESULTS_temporal_{hc,moment}.md"
echo "  cross-user lower bound : notebooks/mdaniol/modeling/artifacts/baseline_results.json"
echo "  A big optimistic->temporal drop = the 0.90->0.71 over-claim; trust the temporal number."
