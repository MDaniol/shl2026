#!/usr/bin/env bash
# Register a venv as a Jupyter kernel for THIS user, via a launcher wrapper that
# sanitizes the environment.
#
# Why a wrapper (not just a kernelspec "env" block): JupyterHub loads LMOD
# modules that export PYTHONPATH/PYTHONHOME pointing at a system Python; those
# shadow the team's 3.12 venv and crash the kernel at startup. A kernelspec
# "env" block can only SET vars (so it can't remove PYTHONHOME); a wrapper
# script CAN unset them, then exec the real kernel in a clean environment.
#
# Run once, in a terminal (a login node, or a Jupyter terminal):
#   ./scripts/register_kernel.sh                              # shared team venv -> "SHL 2026 (team)"
#   ./scripts/register_kernel.sh "$SCRATCH/venvs/mine" mine   # personal venv    -> "SHL 2026 (mine)"
#
# If you run it INSIDE a running JupyterHub session, restart the server after
# (File -> Hub Control Panel -> Stop My Server, then Start) so it sees the new
# kernel. Registering it before you spawn a session needs no restart.
set -euo pipefail

ROOT="${PLG_GROUPS_STORAGE:?must be on the cluster}/plggmhealth/shl2026"

# Clear the LMOD-polluted vars so the python calls below use the venv alone.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP

VENV="${1:-$ROOT/venv}"           # which venv to expose as a kernel
SUFFIX="${2:-team}"               # kernel label suffix: team | mine | ...
if [ "$SUFFIX" = "team" ]; then
  NAME="shl2026";          DISPLAY="SHL 2026 (team)"
else
  NAME="shl2026-$SUFFIX";  DISPLAY="SHL 2026 ($SUFFIX)"
fi

PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no python at $PY — is the venv path right?" >&2; exit 1; }
"$PY" -c "import ipykernel" 2>/dev/null || {
  echo "ipykernel is not installed in $VENV" >&2
  echo "  team venv : ask the lead (it's pinned in pyproject [dev])" >&2
  echo "  personal  : uv pip install -e '.[dev]'  (includes ipykernel)" >&2
  exit 1
}

# The env every kernel of this venv should carry (so notebooks need no env.sh).
CACHE="$ROOT/data/embeddings"
URI_FILE="$ROOT/mlflow_uri"
if [ -f "$URI_FILE" ]; then read -r URI < "$URI_FILE"; else URI="http://172.23.30.9:5000"; fi

# 1) Install the kernelspec, then find where it landed.
"$PY" -m ipykernel install --user --name "$NAME" --display-name "$DISPLAY"
KDIR="$("$PY" -c "from jupyter_client.kernelspec import KernelSpecManager as K; print(K().get_kernel_spec('$NAME').resource_dir)")"

# 2) Write a launcher that strips the polluting env, sets the team env, then
#    execs the kernel. (A wrapper can unset PYTHONHOME; a kernelspec env can't.)
cat > "$KDIR/launch.sh" <<EOF
#!/usr/bin/env bash
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
export MLFLOW_TRACKING_URI="$URI"
export SHL_EMB_CACHE="$CACHE"
exec "$PY" -m ipykernel_launcher "\$@"
EOF
chmod +x "$KDIR/launch.sh"

# 3) Point the kernelspec at the launcher.
"$PY" - "$KDIR/kernel.json" "$KDIR/launch.sh" <<'PY'
import json, sys
kj, launch = sys.argv[1:3]
spec = json.load(open(kj))
spec["argv"] = [launch, "-f", "{connection_file}"]
json.dump(spec, open(kj, "w"), indent=1)
print("argv ->", spec["argv"])
PY

echo
echo "Registered '$DISPLAY'  (python: $PY)"
echo "If you ran this inside a running JupyterHub session, restart it"
echo "  (File -> Hub Control Panel -> Stop My Server, then Start)."
echo "Then: Kernel -> Change Kernel -> '$DISPLAY'."
