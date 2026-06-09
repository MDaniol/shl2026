#!/usr/bin/env bash
# Build the hermetic Apptainer image for the current git SHA, tagged with it.
#
# On Athena: needs apptainer in PATH (module load apptainer).
# Locally: needs docker (or podman) and apptainer.
#
# Writes:
#   containers/shl2026_<gitsha>.sif
#   containers/shl2026_<gitsha>.sif.sha256

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is dirty; commit before building a reproducible image." >&2
  exit 65
fi
GITSHA="$(git rev-parse --short HEAD)"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

DOCKER_TAG="shl2026:${GITSHA}"
SIF="containers/shl2026_${GITSHA}.sif"

echo "==> docker build ${DOCKER_TAG}"
docker build -f containers/Dockerfile -t "${DOCKER_TAG}" .

echo "==> apptainer build ${SIF}"
DEF="containers/apptainer.def"
TMPDEF="$(mktemp)"
trap 'rm -f "$TMPDEF"' EXIT
sed \
  -e "s|{{ GITSHA }}|${GITSHA}|g" \
  -e "s|{{ BUILD_DATE }}|${BUILD_DATE}|g" \
  "$DEF" > "$TMPDEF"

apptainer build --force "$SIF" "$TMPDEF"

sha256sum "$SIF" | tee "${SIF}.sha256"
echo "==> built ${SIF}"
