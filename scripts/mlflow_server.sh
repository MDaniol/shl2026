#!/usr/bin/env bash
# Team MLflow tracking server. Run inside tmux on the PINNED Athena login node
# (the hostname is baked into every student's MLFLOW_TRACKING_URI).
#
#   tmux new -s mlflow
#   source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/venv/bin/activate"
#   ./scripts/mlflow_server.sh
#   # detach: Ctrl-b d
#
# Artifacts are PROXIED through the server (MLflow 2.x default), so students
# need no write access to the artifact directory — only HTTP to this port.
set -euo pipefail

ROOT="${PLG_GROUPS_STORAGE}/plggmhealth/shl2026"
PORT="${MLFLOW_PORT:-5000}"

mkdir -p "${ROOT}/mlflow/artifacts"

echo "MLFLOW_TRACKING_URI for students:  http://$(hostname -f):${PORT}"

# --workers 1 is load-bearing: the default (4 gunicorn workers) means four
# processes writing one SQLite file on Lustre, whose POSIX-lock support is
# mount-dependent -> intermittent "database is locked" or worse. One process
# handles 12 light users fine and sidesteps cross-process locking entirely.
exec mlflow server \
  --backend-store-uri "sqlite:///${ROOT}/mlflow/backend.db" \
  --artifacts-destination "${ROOT}/mlflow/artifacts" \
  --workers 1 \
  --host 0.0.0.0 --port "${PORT}"
