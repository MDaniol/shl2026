#!/usr/bin/env bash
# Submit a Slurm job that runs the SHL 2026 pipeline inside the hermetic
# Apptainer image. Refuses to submit on a dirty tree or a missing/mismatched
# image, so every Slurm job is linkable back to a single (git SHA, image SHA).
#
# Usage:
#   ./scripts/submit.sh <experiment_name> [hydra_overrides...]
#
# Required env (typically exported by the user or by ~/.bashrc on Helios):
#   PLGRID_GRANT          PLGrid grant ID (used in storage paths and MLflow tags)
#   PLG_GROUPS_STORAGE    set by Helios login
#   SCRATCH               set by Helios login

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <experiment_name> [hydra_overrides...]" >&2
  exit 64
fi
EXPERIMENT="$1"; shift

: "${PLGRID_GRANT:?must be set (e.g. plgshl2026)}"
: "${PLG_GROUPS_STORAGE:?must be set on Helios}"
: "${SCRATCH:?must be set on Helios}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# 1. Refuse to submit on a dirty tree.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is dirty; commit or stash before submitting." >&2
  git status --short >&2
  exit 65
fi
GITSHA="$(git rev-parse --short HEAD)"
GITSHA_FULL="$(git rev-parse HEAD)"

# 2. Locate the hermetic image for this commit.
SIF="${REPO_ROOT}/containers/shl2026_${GITSHA}.sif"
if [[ ! -f "$SIF" ]]; then
  echo "error: container ${SIF} not found." >&2
  echo "       build it with: ./scripts/build_container.sh" >&2
  exit 66
fi

# 3. Verify the image SHA-256 matches the recorded value, if present.
SIF_SHA256="$(sha256sum "$SIF" | awk '{print $1}')"
RECORDED="${REPO_ROOT}/containers/shl2026_${GITSHA}.sif.sha256"
if [[ -f "$RECORDED" ]]; then
  EXPECTED="$(awk '{print $1}' "$RECORDED")"
  if [[ "$SIF_SHA256" != "$EXPECTED" ]]; then
    echo "error: image SHA-256 mismatch for ${SIF}" >&2
    echo "       expected ${EXPECTED}" >&2
    echo "       actual   ${SIF_SHA256}" >&2
    exit 67
  fi
fi

# 4. Compose the Slurm script.
JOBDIR="${SCRATCH}/shl2026/runs/${EXPERIMENT}_${GITSHA}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$JOBDIR"
SCRIPT="${JOBDIR}/job.sbatch"

cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=shl2026-${EXPERIMENT}
#SBATCH --account=${PLGRID_GRANT}
#SBATCH --partition=${SHL_PARTITION:-plgrid-gpu-h100}
#SBATCH --gres=gpu:${SHL_GPUS:-1}
#SBATCH --cpus-per-task=${SHL_CPUS:-8}
#SBATCH --mem=${SHL_MEM:-64G}
#SBATCH --time=${SHL_TIME:-04:00:00}
#SBATCH --output=${JOBDIR}/slurm-%j.out
#SBATCH --error=${JOBDIR}/slurm-%j.err

set -euo pipefail

module load apptainer

export SHL_GIT_SHA="${GITSHA_FULL}"
export SHL_SIF_SHA256="${SIF_SHA256}"
export PLGRID_GRANT="${PLGRID_GRANT}"
export MLFLOW_TRACKING_URI="\${MLFLOW_TRACKING_URI:-file://\${PLG_GROUPS_STORAGE}/${PLGRID_GRANT}/shl2026/mlflow}"
export MLFLOW_EXPERIMENT_NAME="${EXPERIMENT}"

apptainer exec --nv \\
  --bind "\${PLG_GROUPS_STORAGE}/${PLGRID_GRANT}/shl2026/data:/data" \\
  --bind "\${PLG_GROUPS_STORAGE}/${PLGRID_GRANT}/shl2026/models:/models:ro" \\
  --bind "\${PLG_GROUPS_STORAGE}/${PLGRID_GRANT}/shl2026/mlflow:/mlflow" \\
  --bind "\${PLG_GROUPS_STORAGE}/${PLGRID_GRANT}/shl2026/dvc-cache:/dvc-cache" \\
  --bind "${JOBDIR}:/scratch" \\
  --bind "${REPO_ROOT}:/opt/shl2026:ro" \\
  "${SIF}" \\
  dvc repro $*
EOF

chmod +x "$SCRIPT"

# 5. Record the run-context manifest before queueing.
cat > "${JOBDIR}/run.json" <<EOF
{
  "experiment": "${EXPERIMENT}",
  "git_sha": "${GITSHA_FULL}",
  "container_sha256": "${SIF_SHA256}",
  "plgrid_grant": "${PLGRID_GRANT}",
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hydra_overrides": $(printf '%s\n' "$@" | python3 -c 'import json,sys; print(json.dumps([l.rstrip() for l in sys.stdin if l.strip()]))' 2>/dev/null || echo "[]")
}
EOF

echo "scaffold: job script written to ${SCRIPT}"
echo "scaffold: would run: sbatch ${SCRIPT}"
# Uncomment when ready:
# sbatch "$SCRIPT"
