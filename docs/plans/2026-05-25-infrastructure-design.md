# SHL 2026 — Reproducible Research Infrastructure (Helios HPC)

## Context

The Sussex–Huawei Locomotion Challenge 2026 (HASCA @ UbiComp 2026, Shanghai) asks teams to build a user-independent transportation-mode classifier over 5 s × 100 Hz inertial windows, using **frozen** foundation models with only lightweight trainable heads. Training is from user 1 (4 phone locations: Bag/Hips/Torso/Hand); validation and test mix users 2 & 3; the test set drops the Hand location and ships shuffled frames. Submission is a 92 726 × 500 plain-text per-sample prediction matrix, accompanied by a 3–6 page HASCA paper (ACM SIGCHI template). Challenge window: 10.05–30.06.2026; paper due 04.07.2026.

Work will run on the ACK Cyfronet **Helios** HPC under PLGrid. The goal of this plan is to put a reproducible, traceable research scaffold in place **before any modelling code is written**, so every later artifact — features, embeddings, trained heads, predictions, figures, the submitted matrix, the paper — is linkable back to (git SHA, container hash, data hash, config hash, Slurm job ID). The scaffold must satisfy FAIR-for-software (Findable, Accessible, Interoperable, Reusable) and survive past the workshop as a citable artifact.

Confirmed decisions from this session:
- Hosting: **small team, private GitHub** (flip to public at submission).
- Experiment tracking: **self-hosted MLflow**.
- Data/model versioning: **DVC with remote on Helios group storage**.
- Environment: **Apptainer image** built from a versioned Dockerfile.
- Openness target: code public + arXiv preprint + Zenodo DOI at submission.

Assumptions to verify with the team before execution (listed once here, not repeated below):
- An active PLGrid grant exists with quota on `$PLG_GROUPS_STORAGE` and access to a GPU partition on Helios (H100). Grant ID is referred to as `<GRANT>` throughout.
- Stack is Python 3.11 + PyTorch (CUDA 12.x) + Hugging Face; alternatives (JAX/TF) not in scope.
- Code license: **MIT**; data license: **SHL terms** (no raw data committed, ever).
- MLflow may run as a long-lived process on a Helios login node or via a dedicated, periodically renewed Slurm service job; if neither holds, we fall back to a file-scheme tracking URI on shared storage (described below).

---

## Repository layout (local working tree, `SHL_2026/`)

```
SHL_2026/
├── README.md                  # pitch + 10-line quickstart
├── METHODS.md                 # dataset, splits, eval, frozen-FM rule
├── REPRODUCIBILITY.md         # the contract (see below)
├── LICENSE                    # MIT (code)
├── DATA_LICENSE.md            # SHL terms, what is/isn’t redistributable
├── CITATION.cff               # machine-readable citation
├── codemeta.json              # FAIR software metadata
├── pyproject.toml             # uv-managed, locked
├── uv.lock
├── dvc.yaml                   # pipeline DAG (see Data versioning)
├── params.yaml                # tunables DVC tracks for cache keys
├── .pre-commit-config.yaml
├── .gitignore / .dvcignore
├── conf/                      # Hydra configs
│   ├── data/                  # per-location, per-split
│   ├── model/                 # per foundation model + per head
│   ├── train/                 # head training schedules
│   └── eval/                  # validation + submission formatting
├── src/shl2026/               # installable package
│   ├── data/                  # readers for Acc/Gyr/Magn/Label files
│   ├── features/              # windowing, normalisation, sanity checks
│   ├── foundation/            # adapters per FM (MOMENT, Chronos, BERT, …)
│   ├── heads/                 # lightweight trainable classifiers
│   ├── eval/                  # metrics, per-user / per-mode breakdowns
│   ├── submission/            # 92726×500 writer + validator
│   └── tracking/              # MLflow helpers, run-context recorder
├── scripts/
│   ├── submit.sh              # Slurm wrapper (records SHA + image hash)
│   ├── build_container.sh
│   └── verify_data.py         # SHA-256 of all raw .txt files
├── containers/
│   ├── Dockerfile
│   └── apptainer.def
├── tests/                     # pytest; CPU-only smoke
├── notebooks/                 # nbstripout-enforced, exploratory only
├── paper/                     # acmart template, figures auto-built
├── docs/
│   └── plans/                 # design docs (incl. this one, committed)
└── .github/workflows/         # CI
```

Rules:
- `notebooks/` outputs are stripped by pre-commit; analysis that matters moves into `src/` + a DVC stage.
- Nothing under `data/` is ever committed; the directory exists only as a bind-mount target on Helios.
- Every script that produces a file the paper depends on must be invokable by `dvc repro <stage>` — never ad-hoc.

---

## Helios layout and bind plan

```
$HOME/shl2026/                                      # git clone (code only)
$SCRATCH/shl2026/runs/<jobid>/                      # ephemeral run scratch
$PLG_GROUPS_STORAGE/<GRANT>/shl2026/
├── data/raw/                                       # immutable SHL files (chmod a-w)
│   ├── train/{Bag,Hips,Torso,Hand}/Acc_*.txt, Gyr_*.txt, Magn_*.txt, Label.txt
│   ├── validation/{Bag,Hips,Torso,Hand}/...
│   ├── test/{Bag,Hips,Torso}/...                   # no Hand, frames shuffled
│   └── MANIFEST.sha256                             # checked at every run start
├── data/processed/                                 # DVC-tracked features, embeddings
├── models/foundation/                              # frozen FM weight cache (HF, etc.)
├── mlflow/{backend.db, artifacts/}                 # tracking store
├── containers/shl2026_<gitsha>.sif                 # built images, immutable
└── dvc-cache/                                      # DVC remote
```

Permissions: `data/raw` and `containers/` are world-read, owner-only-write, and recursively chmodded read-only after population. `MANIFEST.sha256` is generated once by `scripts/verify_data.py` and is the authority for what "the dataset" is.

---

## Sync workflow (laptop ↔ Helios)

Single source of truth for code is GitHub. Helios pulls; it never originates code.

- **Laptop**: develop, run unit tests, push branches, open PRs.
- **Helios**: `git pull` inside `$HOME/shl2026`; `dvc pull` to hydrate the cache; submit Slurm jobs.
- **Raw data ingest**: one-time download on Helios (or rsync from a staging machine) into `$PLG_GROUPS_STORAGE/.../data/raw`, then SHA-256 manifest is generated and the tree is made read-only.
- Editing on Helios via VS Code Remote-SSH is allowed for emergencies, but any change must go through a PR before being merged.

---

## Containerization (the hermetic boundary)

- `containers/Dockerfile` pins: base CUDA image, Python 3.11, PyTorch, Hugging Face, MLflow client, DVC, project deps via `uv pip compile`.
- `containers/apptainer.def` wraps the Docker image; build produces `shl2026_<gitsha>.sif`.
- The Slurm submit wrapper refuses to run if (a) the working tree is dirty, (b) the `.sif` for the current git SHA is missing, or (c) the `.sif` SHA-256 differs from the recorded one.
- Bind plan inside the container: code RO, `data/raw` RO, `models/foundation` RO, `$SCRATCH/.../runs` RW, `mlflow/` RW, `dvc-cache/` RW.
- Image rebuilds are themselves CI artifacts (see Quality gates).

---

## Experiment tracking — MLflow

- Backend store: SQLite file on `$PLG_GROUPS_STORAGE/.../mlflow/backend.db` (Postgres optional later; SQLite is enough for one team).
- Artifact store: `$PLG_GROUPS_STORAGE/.../mlflow/artifacts/`.
- Tracking server: long-running process on a Helios login node (`mlflow server --backend-store-uri ... --default-artifact-root ...`), wrapped by `tmux` or a small systemd-style user script. Accessed from the laptop via SSH port-forward (`ssh -L 5000:localhost:5000 helios`).
- **Fallback** if persistent processes are disallowed: set `MLFLOW_TRACKING_URI=file:///.../mlflow/` — all clients (Slurm jobs, laptop) write to the same shared filesystem; UI is launched ad hoc by any user pointing at the same path. No server, no port-forward.
- Every run logs:
  - params: full Hydra-resolved YAML, `params.yaml` hash, foundation-model identifier + revision, seed
  - tags: git SHA, dirty-flag, container SHA-256, Slurm job ID, Helios node, PLGrid grant
  - metrics: macro-F1 (challenge metric), per-class F1, per-user F1, per-location F1, latency
  - artifacts: confusion matrices, learning curves, the run’s resolved config, a copy of the submission validator’s report
- Run naming convention: `<experiment>/<git-short-sha>/<slurm-jobid>`.

---

## Data + pipeline versioning — DVC

DVC turns the project into a DAG whose nodes have content-addressed inputs and outputs. Pipeline stages (all in `dvc.yaml`):

1. `verify_raw` — recompute SHA-256 of every raw `.txt`, compare to `MANIFEST.sha256`. Refuses to continue on mismatch.
2. `window_features` — per (split, location): load Acc/Gyr/Magn, sanity-check shape, normalise, write Parquet shards. Depends on `params.yaml:features.*`.
3. `frozen_embeddings` — per (split, location, foundation_model): run the frozen FM over windows, write embeddings. Depends on `params.yaml:foundation.<name>` (model id, revision, pooling).
4. `train_head` — per (foundation_model, head_arch): train lightweight head on train embeddings, evaluate on validation. Logs to MLflow.
5. `predict_test` — run the chosen (FM, head) on test embeddings (Bag/Hips/Torso only).
6. `format_submission` — emit `teamName_predictions.txt` (92 726 × 500), run the strict validator, copy to MLflow artifacts.

Notes:
- Foundation-model weights are pulled from Hugging Face once into `models/foundation/<name>@<revision>` and pinned by revision SHA in `params.yaml`; the DVC stage takes this directory as a dependency so a revision bump invalidates downstream caches.
- DVC remote = `$PLG_GROUPS_STORAGE/.../dvc-cache`. Pushed on every merge to `main`; pulled by collaborators and CI.
- The Hand-location data is intentionally excluded from the test-time pipeline (`predict_test`) but kept available in `train_head` per a config flag, since participants may choose to leverage it during training or ignore it to better match test conditions.

---

## Configuration management

- **Hydra** composes configs from `conf/`; the fully resolved config is written into the run dir and logged as an MLflow artifact.
- **`params.yaml`** is the DVC-visible subset (anything that should bust DVC cache keys when changed): seeds, foundation-model id/revision, head architecture, window normalisation parameters.
- All seeds (NumPy, PyTorch, Python `random`, CUDA) are set centrally in `src/shl2026/tracking/run_context.py` and logged. PyTorch deterministic mode is enabled where it doesn’t cripple a frozen FM forward pass.

---

## Slurm submission

`scripts/submit.sh <experiment> <hydra_overrides...>`:

1. Refuse if `git status` is dirty.
2. Compute `gitsha=$(git rev-parse --short HEAD)`.
3. Assert `containers/shl2026_${gitsha}.sif` exists and matches recorded SHA-256; otherwise print the `build_container.sh` command and exit.
4. Render a Slurm script (partition, time, GPU count from the experiment’s `conf/`).
5. Inside the job: `apptainer exec --bind ...` the image; export `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `SHL_GIT_SHA`, `SHL_SIF_SHA256`, `SLURM_JOB_ID`; invoke `python -m shl2026.<entrypoint> ...`.
6. On exit, write a small `run.json` next to the run dir containing all of the above for post-hoc forensics.

Job arrays are the natural unit for per-location and per-foundation-model sweeps.

---

## Quality gates / CI

- **Pre-commit**: `ruff format`, `ruff check`, `mypy` (lenient), `nbstripout`, `gitleaks`, `dvc-pre-commit`.
- **GitHub Actions on PR**: install with `uv`; run `ruff`, `mypy`, `pytest` (CPU-only smoke with tiny synthetic frames); build the Apptainer image in a self-hosted runner if available, else skip image build and only validate the Dockerfile.
- **On merge to `main`**: rebuild image, push the new `.sif` to Helios via a deploy key (or open a Helios PR with the new image hash), tag the commit with the image SHA-256.
- **Branch protection**: PR required, one review, CI green.

---

## FAIR / open-science artifacts

- `README.md`: project pitch, 10-line quickstart, link to Zenodo DOI placeholder.
- `METHODS.md`: dataset description, the user-independent split, the frozen-FM rule (quoted from the challenge), evaluation metric, known caveats (Hand absent at test, shuffled test frames).
- `REPRODUCIBILITY.md`: the **contract** — "Given (git SHA, container SHA-256, raw-data MANIFEST.sha256, DVC pipeline hashes, `params.yaml` hash), `dvc repro` must produce bit-identical `teamName_predictions.txt`, modulo declared fp-nondeterminism." Lists exactly which environment variables, seeds, and CUDA settings are part of the contract.
- `CITATION.cff` + `codemeta.json`: kept in sync via pre-commit hook.
- `LICENSE` (MIT) + `DATA_LICENSE.md` (cites SHL terms, makes explicit that raw data is not redistributed by this repo).
- Zenodo: concept DOI reserved at project start (manual, one-time); GitHub release at submission time triggers a versioned DOI via Zenodo–GitHub integration.
- arXiv: preprint posted at submission (same content as HASCA paper, plus optional extended appendix); arXiv ID added to `CITATION.cff`.

---

## Submission pipeline

- `dvc repro format_submission` is the only sanctioned way to produce the matrix.
- The validator (`src/shl2026/submission/validate.py`) hard-asserts shape `(92726, 500)`, integer dtype, label set ⊆ the eight challenge classes, no NaNs.
- The final file is logged as an MLflow artifact under a `submission/` tag, alongside the resolved config, the image SHA-256, and the git SHA. The email to `shldataset.challenge@gmail.com` will reference that run.

---

## Paper (HASCA-SHL)

- `paper/` holds the ACM SIGCHI Master template (acmart, sigconf, 2-column), BibTeX, and a `figures/` dir populated by `python -m shl2026.eval.make_paper_figures --run <mlflow-run-id>`. Figures are not hand-edited; rerunning the script regenerates them deterministically.
- `paper/README.md` lists the exact MLflow run IDs that back each figure and table.
- Camera-ready triggers a Zenodo release whose DOI is added back into the paper (final pre-camera-ready edit).

---

## Verification (how we know the scaffold works before any modelling starts)

1. **Smoke end-to-end**: a synthetic-data fixture (a few hundred fake 5 s frames) is checked into `tests/fixtures/`. `dvc repro` from a clean clone, inside the container, produces a `(N, 500)` prediction matrix and passes the validator.
2. **Second-person repro**: a teammate clones the repo on Helios, `dvc pull`s, runs the smoke pipeline, and gets bit-identical output (or within a documented fp-noise tolerance).
3. **Manifest integrity**: deliberately corrupt one byte in a copy of a raw file; `verify_raw` must fail and refuse to continue.
4. **Dirty-tree refusal**: edit a tracked file without committing, run `scripts/submit.sh` — must abort with a clear message.
5. **MLflow contract**: a smoke run’s tags include git SHA, container SHA-256, Slurm job ID, and the resolved Hydra config is present as an artifact.
6. **Container determinism**: rebuild the image from the same git SHA on a second machine; the resulting `.sif` SHA-256 should match (modulo timestamp metadata — if it doesn’t, document the irreducible nondeterminism and pin it in `REPRODUCIBILITY.md`).

Only after these six checks pass do we start writing modelling code.

---

## Out of scope for this plan

- Model selection (which foundation models to try, which head architectures): tracked separately in `docs/plans/` once the scaffold is in place.
- Paper outline and narrative.
- Long-term post-workshop maintenance (will be revisited at camera-ready time).
