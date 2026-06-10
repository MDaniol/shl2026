# Setup — one-time bootstrap (lead)

Manual steps taken **once** to bring a fresh environment online. Students never
do any of this — their path is `STUDENTS.md`. Cluster reference:
`docs/CLUSTER.md`.

## A. Local laptop

```bash
# Clone (after the GitHub repo exists)
git clone git@github.com:MDaniol/shl2026.git
cd shl2026

# Dev env
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"     # in case ~/.local/bin isn't on PATH yet
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
pytest -q                                # CPU smoke
```

## B. Athena (one-time, per team)

1. **PLGrid grant.** Grant **`plgshl26`** (active 2026-05-29 → 2027-05-28):
   Athena `plgshl26-gpu-a100` (15,000 GPU-h, partition `plgrid-gpu-a100`), plus
   Helios fallback allocations. Storage = 6 TB via group **`plggmhealth`** —
   the shared area is keyed by *group name*, not grant name. Put in `~/.bashrc`:

   ```bash
   export PLGRID_GRANT=plgshl26
   ```

2. **Layout.** From an Athena login node:

   ```bash
   ROOT="${PLG_GROUPS_STORAGE}/plggmhealth/shl2026"
   mkdir -p "$ROOT"/{data/{raw,embeddings},models/foundation,mlflow/artifacts,dvc-cache,containers}
   ```

3. **Raw data.** Download the SHL Challenge 2026 train / validation / test
   archives (links from <http://www.shl-dataset.org/challenge-2026/> after team
   registration; SHL terms — see `DATA_LICENSE.md`, never redistribute) into
   `$ROOT/data/raw/`, preserving the `train/{Bag,Hips,Torso,Hand}/` layout and
   the test split with no Hand directory. Then:

   ```bash
   python scripts/verify_data.py generate \
     --root "$ROOT/data/raw" \
     --out  "$ROOT/data/raw/MANIFEST.sha256"

   chmod -R a-w "$ROOT/data/raw"     # make immutable
   ```

4. **Clone code on Athena:**

   ```bash
   cd "$HOME"
   git clone git@github.com:MDaniol/shl2026.git
   cd shl2026
   ```

5. **Shared Python environment + `env.sh`** — built once, used by all 12
   students. `$HOME` is only 10 GB per person, so the venv, the uv cache, and
   the uv-managed interpreter all live **off `$HOME`** (group storage /
   scratch). Students never run `uv`; they `source` one file.

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   export UV_CACHE_DIR="$SCRATCH/uv-cache"           # build cache (regenerable)
   export UV_PYTHON_INSTALL_DIR="$ROOT/uv-python"    # interpreter the venv links to

   # Canonical clone the venv installs from (editable): updating the team SDK
   # is `git pull` here (+ re-run the install line if dependencies changed).
   git clone https://github.com/MDaniol/shl2026.git "$ROOT/repo"

   uv venv --python 3.12 "$ROOT/venv"
   source "$ROOT/venv/bin/activate"
   uv pip install -e "$ROOT/repo[dev]"
   deactivate

   cat > "$ROOT/env.sh" <<'EOF'
   # SHL 2026 team environment — students just `source` this (STUDENTS.md).
   export SHL_EMB_CACHE="$PLG_GROUPS_STORAGE/plggmhealth/shl2026/data/embeddings"
   export MLFLOW_TRACKING_URI="<MLFLOW_URI>"   # lead fills in after step 8
   source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/venv/bin/activate"
   EOF

   chmod -R g+rX "$ROOT"/{venv,repo,uv-python} "$ROOT/env.sh"

   # Seed the synthetic embedding cache — STUDENTS.md promises that
   # embeddings("synthetic", ...) always works, but only tests generate it
   # (into tmp dirs); the shared cache needs this once:
   source "$ROOT/env.sh"
   python -c "from shl2026.data.synthetic import make_synthetic_embedding_cache as m; \
              import os; print(m(os.environ['SHL_EMB_CACHE']))"
   chmod -R g+rX "$SHL_EMB_CACHE"
   ```

   **Adding a package later** (student request — aim for same-day):

   ```bash
   source "$ROOT/venv/bin/activate"
   uv pip install <pkg>
   # pin it: add to pyproject.toml dependencies in $ROOT/repo, commit + push
   chmod -R g+rX "$ROOT/venv"
   ```

   Rebuild the container before the next *certified* run (`dvc repro` /
   submission), not on every install. **Submission gate:** before regenerating
   any winning run, check its MLflow `python_env` tag — if it isn't
   `$ROOT/venv`, the student used a personal venv: promote the packages, have
   them re-run in the team env, and only then certify.

6. **DVC remote:**

   ```bash
   dvc init
   dvc remote add -d athena "$ROOT/dvc-cache"
   git add .dvc/config && git commit -m "dvc: add athena remote"
   ```

7. **Build the container** (once per dependency change):

   ```bash
   module load apptainer
   ./scripts/build_container.sh
   mv containers/shl2026_*.sif "$ROOT/containers/"
   ln -sf "$ROOT/containers/shl2026_$(git rev-parse --short HEAD).sif" \
          containers/
   ```

   (If on-cluster builds are blocked — policy is undocumented, ask Helpdesk —
   build off-cluster and `rsync` the `.sif` in.)

8. **MLflow server** (long-running on a login node via tmux). It must be
   reachable from **compute nodes** (JupyterHub sessions). Three cluster
   gotchas are baked into `scripts/mlflow_server.sh`:

   - **Use the INTERNAL address, not the hostname.** Login nodes are
     multi-homed: the public name (`login01.athena.cyfronet.pl` → `ext`
     interface) is **unreachable from compute nodes** (`errno 113, no route
     to host`). Compute traffic rides the InfiniBand network (`ib0`,
     verified 2026-06: login01 = `172.23.30.9`). The script derives the
     right address automatically (`ip route get <a-compute-node-ip>` →
     `src` field) and prints the URI to publish.
   - **Pin the login node.** `athena.cyfronet.pl` can land you on any login
     node, and the pinned node's *IP* is baked into every student's tracking
     URI. Note which node you're on (`hostname`, e.g. `login01`) and always
     run/restart the server **on that same node** (`ssh login01` from the
     other one if needed).
   - **Artifacts are proxied through the server** (`--artifacts-destination`,
     MLflow 2.x default) instead of `--default-artifact-root`. With the old
     flag each client writes artifacts straight to the filesystem path —
     which requires every student to have group-write + the right umask
     there. Proxied, students only need HTTP to the port.

   ```bash
   tmux new -s mlflow
   source "$ROOT/venv/bin/activate"
   ./scripts/mlflow_server.sh      # prints the student-facing URI
   # detach with Ctrl-b d
   ```

   Record the printed `http://<internal-ip>:5000` — that is the
   **`<MLFLOW_URI>`**: write it into **`$ROOT/env.sh`** (step 5), which sets
   `MLFLOW_TRACKING_URI` for every student. Anyone on the cluster can reach
   the port, which is acceptable for this sprint. From a laptop, view the UI
   via `ssh -L 5000:localhost:5000 <login>@athena.cyfronet.pl` (tunnel
   terminates on the login node itself, so `localhost` works there) →
   <http://localhost:5000>.

   tmux survives logout but **not a login-node reboot** (maintenance). If
   students report connection errors: `ssh` to the pinned node,
   `tmux attach -t mlflow` (or re-run the block above). The SQLite file and
   artifacts live on group storage, so nothing is lost across restarts.

   **Verify before announcing the URI** (each check kills one assumption):

   ```bash
   # 1. SQLite locking works on this Lustre mount (run on the login node):
   python -c "import sqlite3; c=sqlite3.connect('$ROOT/mlflow/_lock_test.db'); \
              c.execute('create table t(x)'); c.execute('insert into t values(1)'); \
              c.commit(); print('sqlite OK')"
   rm "$ROOT/mlflow/_lock_test.db"
   # If this throws "database is locked"/"disk I/O error": move the backend
   # to NFS instead — backend-store-uri "sqlite:////$HOME/mlflow-backend.db"
   # (tiny file, fits $HOME quota easily; artifacts stay on group storage).

   # 2. Server is reachable from a COMPUTE node (where students actually run)
   #    — use the INTERNAL IP the script printed, never the public hostname:
   srun -A plgshl26-gpu-a100 -p plgrid-gpu-a100 --gres=gpu:1 --time=0:05:00 \
        curl -s http://<internal-ip>:5000/health        # expect: OK

   # 3. An end-to-end run lands in the DB (from that same compute node or a
   #    JupyterHub session): run the STUDENTS.md "first run" snippet and
   #    confirm it appears in leaderboard() and in the UI.
   ```

   If the login node enforces CPU-time limits that eventually kill the
   server (policy undocumented — symptom: tmux pane shows the process was
   killed), ask Helpdesk for a blessed place to run it; restarts lose
   nothing meanwhile.

   Do **not** fall back to a `file://` store or per-user databases — 12
   concurrent writers need the server, and per-directory SQLite silently
   fragments the leaderboard.

9. **First smoke:**

   ```bash
   dvc repro verify_raw         # SHA-256 sanity over raw data
   pytest -q                    # CPU tests
   ./scripts/submit.sh smoke    # dry-render of a Slurm job
   ```

## C. GitHub side

- Repo: **public** `MDaniol/shl2026` (<https://github.com/MDaniol/shl2026>).
  Add every student as a collaborator with write access.
- Branch protection on `main` (applied 2026-06-10 via `gh api`): **PR required,
  0 approvals** — students self-merge instantly, but no direct pushes; admins
  (lead) exempt (`enforce_admins: false`); force-pushes and branch deletion
  blocked. Students' flow: branch → push → PR → self-merge (`STUDENTS.md`
  house rule 4). CI stays on push.
- Reserve a Zenodo concept DOI (linked GitHub release at submission time).
- Add the eventual repo URL + DOI placeholder back into `CITATION.cff`,
  `codemeta.json`, and `README.md`.
- Placeholders (filled 2026-06-10): lead contact in `STUDENTS.md` =
  Mateusz Daniol — <daniol@agh.edu.pl>; `<MLFLOW_URI>` in `$ROOT/env.sh`
  (step B.8) = `http://172.23.30.9:5000` (login01 ib0).

## D. Verification (the six checks)

These are the gating checks from `REPRODUCIBILITY.md`. Run them before any
modelling work starts:

1. `pytest -q` green (smoke + validator + manifest-corruption).
2. `dvc repro` from a clean clone produces the synthetic-data submission.
3. Second-person reproduction (a teammate, same SHA, same image) matches.
4. Manifest corruption → `verify_data.py verify` exits non-zero.
5. Dirty-tree `submit.sh` aborts with code 65.
6. Rebuild image from same SHA on a second machine → record SHA-256 delta in
   `REPRODUCIBILITY.md` if non-zero.
