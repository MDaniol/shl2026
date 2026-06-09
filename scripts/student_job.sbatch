#!/usr/bin/env bash
# Run a student experiment script on an Athena compute node.
#
# Usage (from the repo root, after Part 1 of STUDENTS.md):
#   sbatch scripts/student_job.sbatch notebooks/<your-name>/exp.py [args...]
#
# Output lands in slurm-<jobid>.out next to where you submitted (gitignored).
# Need more time/memory? Override at submit time, e.g.:
#   sbatch --time=02:00:00 --mem=32G scripts/student_job.sbatch ...
#
#SBATCH --job-name=shl2026-exp
#SBATCH --account=plgshl26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

# sbatch starts in the directory you submitted from and inherits your
# environment, so ~/.bashrc exports (SHL_EMB_CACHE, MLFLOW_TRACKING_URI)
# carry over. The cache path is re-derived here as a safety net.
export SHL_EMB_CACHE="${SHL_EMB_CACHE:-$PLG_GROUPS_STORAGE/plggmhealth/shl2026/data/embeddings}"

if [[ ! -f .venv/bin/activate ]]; then
  echo "error: no .venv here — submit from the repo root (cd ~/shl2026)" >&2
  exit 1
fi
source .venv/bin/activate

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch scripts/student_job.sbatch <your-script.py> [args...]" >&2
  exit 64
fi

python "$@"
