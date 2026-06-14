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

# env.sh sets SHL_EMB_CACHE + MLFLOW_TRACKING_URI, clears the LMOD pollution, and
# activates YOUR env (built once by ./scripts/setup_env.sh). See STUDENTS.md.
ENV_SH="$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"
if [[ ! -f "$ENV_SH" ]]; then
  echo "error: team env.sh not found at $ENV_SH — see STUDENTS.md" >&2
  exit 1
fi
source "$ENV_SH"
if ! python -c "import shl2026" 2>/dev/null; then
  echo "error: shl2026 not importable — build your env: ./scripts/setup_env.sh" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch scripts/student_job.sbatch <your-script.py> [args...]" >&2
  exit 64
fi

python "$@"
