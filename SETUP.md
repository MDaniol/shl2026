# Setup — one-time bootstrap

This document describes the manual steps the user (or a teammate) takes once
to bring a fresh environment online. The repo itself stays minimal.

## A. Local laptop

```bash
# Clone (after the GitHub repo exists)
git clone git@github.com:<org>/shl2026.git
cd shl2026

# Dev env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
pytest -q                                # CPU smoke
```

## B. Helios (one-time, per team)

1. **PLGrid grant**. Confirm an active grant; record its ID as `PLGRID_GRANT`
   (e.g. `plgshl2026`). Put it in `~/.bashrc`:

   ```bash
   export PLGRID_GRANT=plgshl2026
   ```

2. **Layout**. From a Helios login node:

   ```bash
   ROOT="${PLG_GROUPS_STORAGE}/${PLGRID_GRANT}/shl2026"
   mkdir -p "$ROOT"/{data/{raw,processed},models/foundation,mlflow/artifacts,dvc-cache,containers}
   ```

3. **Raw data**. Download SHL train / validation / test archives into
   `$ROOT/data/raw/` (preserving the `train/{Bag,Hips,Torso,Hand}/` layout
   and the test split with no Hand directory). Then:

   ```bash
   python scripts/verify_data.py generate \
     --root "$ROOT/data/raw" \
     --out  "$ROOT/data/raw/MANIFEST.sha256"

   chmod -R a-w "$ROOT/data/raw"     # make immutable
   ```

4. **Clone code on Helios**:

   ```bash
   cd "$HOME"
   git clone git@github.com:<org>/shl2026.git
   cd shl2026
   ```

5. **DVC remote**:

   ```bash
   dvc init
   dvc remote add -d helios "$ROOT/dvc-cache"
   git add .dvc/config && git commit -m "dvc: add helios remote"
   ```

6. **Build the container** (once per code-relevant change):

   ```bash
   module load apptainer
   ./scripts/build_container.sh
   mv containers/shl2026_*.sif "$ROOT/containers/"
   ln -sf "$ROOT/containers/shl2026_$(git rev-parse --short HEAD).sif" \
          containers/
   ```

7. **MLflow** (option A: long-running on login node via tmux):

   ```bash
   tmux new -s mlflow
   export ROOT="${PLG_GROUPS_STORAGE}/${PLGRID_GRANT}/shl2026"
   mlflow server \
     --backend-store-uri "sqlite:///${ROOT}/mlflow/backend.db" \
     --default-artifact-root "${ROOT}/mlflow/artifacts" \
     --host 127.0.0.1 --port 5000
   # detach with Ctrl-b d
   ```

   From the laptop:

   ```bash
   ssh -L 5000:127.0.0.1:5000 helios
   # open http://localhost:5000
   ```

   **Option B fallback** (no server allowed): everywhere export

   ```bash
   export MLFLOW_TRACKING_URI="file://${ROOT}/mlflow"
   ```

8. **First smoke**:

   ```bash
   dvc repro verify_raw         # SHA-256 sanity over raw data
   pytest -q                    # CPU tests
   ./scripts/submit.sh smoke    # dry-render of a Slurm job
   ```

## C. GitHub side

- Create private repo `<org>/shl2026`.
- Branch protection on `main`: require PR review (one approver), require CI
  green, no force-push.
- Reserve a Zenodo concept DOI (linked GitHub release at submission time).
- Add the eventual repo URL + DOI placeholder back into `CITATION.cff`,
  `codemeta.json`, and `README.md`.

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
