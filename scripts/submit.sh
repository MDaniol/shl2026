#!/usr/bin/env bash
# Submit a Slurm job that runs the SHL 2026 pipeline inside the hermetic
# Apptainer image. Refuses to submit on a dirty tree or a missing/mismatched
# image, so every Slurm job is linkable back to a single (git SHA, image SHA).
#
# Usage:
#   ./scripts/submit.sh <experiment_name> [dvc_stage ...]
#
# The remaining args are passed to `dvc repro` inside the container (stage
# names from dvc.yaml, e.g. frozen_embeddings or format_submission).
#
# Required env (typically exported by the user or by ~/.bashrc on the cluster):
#   PLGRID_GRANT          PLGrid grant ID, e.g. plgshl26 (MLflow tag + account base)
#   PLG_GROUPS_STORAGE    set by the cluster at login (pr2 on Athena)
#   SCRATCH               set by the cluster at login
# Optional env (have sensible defaults for the Athena A100 extraction job):
#   SHL_GROUP             group name owning the shared storage (default: plggmhealth)
#   SHL_ACCOUNT           Slurm account (default: ${PLGRID_GRANT}-gpu-a100, Athena GPU)
#   SHL_PARTITION         Slurm partition (default: plgrid-gpu-a100)
# For the Helios CPU fallback: SHL_ACCOUNT=${PLGRID_GRANT}-cpu SHL_PARTITION=plgrid
#                              SHL_GPUS=0 (omits --gres/--nv). GH200 is aarch64 —
#                              needs a separate image; not used.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <experiment_name> [dvc_stage ...]" >&2
  exit 64
fi
EXPERIMENT="$1"; shift

: "${PLGRID_GRANT:?must be set (e.g. plgshl26)}"
: "${PLG_GROUPS_STORAGE:?must be set by the cluster at login}"
: "${SCRATCH:?must be set by the cluster at login}"

# Storage is owned by the GROUP, not the grant; the account is resource-specific.
SHL_GROUP="${SHL_GROUP:-plggmhealth}"
SHL_ACCOUNT="${SHL_ACCOUNT:-${PLGRID_GRANT}-gpu-a100}"

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

# SHL_GPUS=0 (Helios CPU fallback) drops the GPU request and --nv entirely.
SHL_GPUS="${SHL_GPUS:-1}"
GRES_LINE="#SBATCH --gres=gpu:${SHL_GPUS}"
NV_FLAG="--nv"
if [[ "$SHL_GPUS" == "0" ]]; then
  GRES_LINE=""
  NV_FLAG=""
fi

cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=shl2026-${EXPERIMENT}
#SBATCH --account=${SHL_ACCOUNT}
#SBATCH --partition=${SHL_PARTITION:-plgrid-gpu-a100}
${GRES_LINE}
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
# Prefer the team MLflow server (exported in ~/.bashrc and inherited by the
# job); the sqlite fallback covers server-down runs (file:// is deprecated).
export MLFLOW_TRACKING_URI="\${MLFLOW_TRACKING_URI:-sqlite:///\${PLG_GROUPS_STORAGE}/${SHL_GROUP}/shl2026/mlflow/backend.db}"
export MLFLOW_EXPERIMENT_NAME="${EXPERIMENT}"

apptainer exec ${NV_FLAG} \\
  --bind "\${PLG_GROUPS_STORAGE}/${SHL_GROUP}/shl2026/data:/data" \\
  --bind "\${PLG_GROUPS_STORAGE}/${SHL_GROUP}/shl2026/models:/models:ro" \\
  --bind "\${PLG_GROUPS_STORAGE}/${SHL_GROUP}/shl2026/mlflow:/mlflow" \\
  --bind "\${PLG_GROUPS_STORAGE}/${SHL_GROUP}/shl2026/dvc-cache:/dvc-cache" \\
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
  "dvc_targets": $(printf '%s\n' "$@" | python3 -c 'import json,sys; print(json.dumps([l.rstrip() for l in sys.stdin if l.strip()]))' 2>/dev/null || echo "[]")
}
EOF

echo "scaffold: job script written to ${SCRIPT}"
echo "scaffold: would run: sbatch ${SCRIPT}"
# Uncomment when ready:
# sbatch "$SCRIPT"
