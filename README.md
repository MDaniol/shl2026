# SHL 2026 — AGH submission

Reproducible pipeline for the **Sussex–Huawei Locomotion Challenge 2026**
(HASCA workshop, UbiComp 2026, Shanghai). User-independent transportation-mode
recognition using **frozen foundation models** with lightweight trainable
heads, on 5 s × 100 Hz inertial windows.

> Code: MIT licence. Data: SHL terms (see `DATA_LICENSE.md`). DOI: reserved on
> Zenodo at submission time. **Predictions deadline: 30.06.2026.**

> **New to the team?** Everything you need is in [`STUDENTS.md`](STUDENTS.md)
> (PLGrid account → first run → daily work).

## Quickstart (Athena, post-setup)

```bash
# Hydrate code + data on Athena
git pull
dvc pull

# Build the hermetic container (only when deps change)
./scripts/build_container.sh

# Submit the embedding-extraction job (refuses on dirty tree or hash mismatch)
./scripts/submit.sh embed frozen_embeddings

# Reproduce end-to-end inside the container
apptainer exec containers/shl2026_$(git rev-parse --short HEAD).sif \
    dvc repro format_submission
```

See `SETUP.md` for the one-time Athena bootstrap, `docs/CLUSTER.md` for the
cluster reference, `METHODS.md` for the scientific contract, and
`REPRODUCIBILITY.md` for the formal reproducibility contract that every
release must honour.

## Layout

```
src/shl2026/        installable Python package
conf/               Hydra configs
dvc.yaml            pipeline DAG
params.yaml         DVC-tracked tunables
containers/         Dockerfile + Apptainer recipe
scripts/            Slurm + bootstrap helpers
tests/              pytest (CPU-only smoke + synthetic fixtures)
paper/              ACM SIGCHI (acmart) HASCA paper
docs/plans/         design documents (committed)
```

## Status

Scaffolding stage — no modelling code yet. The infrastructure must pass the
six verification checks in `REPRODUCIBILITY.md` before any modelling work
begins.
