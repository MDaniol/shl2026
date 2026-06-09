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

5. **DVC remote:**

   ```bash
   dvc init
   dvc remote add -d athena "$ROOT/dvc-cache"
   git add .dvc/config && git commit -m "dvc: add athena remote"
   ```

6. **Build the container** (once per dependency change):

   ```bash
   module load apptainer
   ./scripts/build_container.sh
   mv containers/shl2026_*.sif "$ROOT/containers/"
   ln -sf "$ROOT/containers/shl2026_$(git rev-parse --short HEAD).sif" \
          containers/
   ```

   (If on-cluster builds are blocked — policy is undocumented, ask Helpdesk —
   build off-cluster and `rsync` the `.sif` in.)

7. **MLflow server** (long-running on a login node via tmux). It must be
   reachable from **compute nodes** (JupyterHub sessions), so bind to the login
   node's hostname, not 127.0.0.1:

   ```bash
   tmux new -s mlflow
   ROOT="${PLG_GROUPS_STORAGE}/plggmhealth/shl2026"
   mlflow server \
     --backend-store-uri "sqlite:///${ROOT}/mlflow/backend.db" \
     --default-artifact-root "${ROOT}/mlflow/artifacts" \
     --host "$(hostname)" --port 5000
   # detach with Ctrl-b d
   ```

   Record `http://$(hostname):5000` — that is the **`<MLFLOW_URI>`** every
   student exports as `MLFLOW_TRACKING_URI` (fill it into `STUDENTS.md`).
   Anyone on the cluster can reach the port, which is acceptable for this
   sprint. From a laptop, view the UI via
   `ssh -L 5000:<that-hostname>:5000 athena` → <http://localhost:5000>.

   Do **not** fall back to a `file://` store or per-user databases — 12
   concurrent writers need the server, and per-directory SQLite silently
   fragments the leaderboard.

8. **First smoke:**

   ```bash
   dvc repro verify_raw         # SHA-256 sanity over raw data
   pytest -q                    # CPU tests
   ./scripts/submit.sh smoke    # dry-render of a Slurm job
   ```

## C. GitHub side

- Repo: **public** `MDaniol/shl2026` (<https://github.com/MDaniol/shl2026>).
  Add every student as a collaborator with write access (they push to `main`
  directly this sprint — see `STUDENTS.md`, so **no required reviews**; just
  block force-pushes and keep CI on push).
- Reserve a Zenodo concept DOI (linked GitHub release at submission time).
- Add the eventual repo URL + DOI placeholder back into `CITATION.cff`,
  `codemeta.json`, and `README.md`.
- Fill the `STUDENTS.md` placeholders: `<MLFLOW_URI>`, `<NAME>`,
  `<EMAIL>`.

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
