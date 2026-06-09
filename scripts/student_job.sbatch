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

# The team env.sh provides SHL_EMB_CACHE, MLFLOW_TRACKING_URI and the shared
# venv (STUDENTS.md Part 1, step 5). A personal .venv in the submit directory
# (see "Need an extra library?") is activated on top, so it wins on Python
# packages while the team env vars stay.
ENV_FOUND=0
ENV_SH="$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"
if [[ -f "$ENV_SH" ]]; then
  source "$ENV_SH"
  ENV_FOUND=1
fi
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
  ENV_FOUND=1
fi
if [[ "$ENV_FOUND" -eq 0 ]]; then
  echo "error: team env not found at $ENV_SH and no local .venv — see STUDENTS.md" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch scripts/student_job.sbatch <your-script.py> [args...]" >&2
  exit 64
fi

python "$@"
