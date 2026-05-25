# SHL 2026 — AGH submission

Reproducible pipeline for the **Sussex–Huawei Locomotion Challenge 2026**
(HASCA workshop, UbiComp 2026, Shanghai). User-independent transportation-mode
recognition using **frozen foundation models** with lightweight trainable
heads, on 5 s × 100 Hz inertial windows.

> Code: MIT licence. Data: SHL terms (see `DATA_LICENSE.md`). DOI: reserved on
> Zenodo at submission time.

## Quickstart (Helios, post-setup)

```bash
# Hydrate code + data on Helios
git pull
dvc pull

# Build the hermetic container (only when deps change)
./scripts/build_container.sh

# Submit a run (refuses on dirty tree or hash mismatch)
./scripts/submit.sh baseline_moment foundation=moment head=mlp

# Reproduce end-to-end inside the container
apptainer exec containers/shl2026_$(git rev-parse --short HEAD).sif \
    dvc repro format_submission
```

See `SETUP.md` for the one-time Helios bootstrap, `METHODS.md` for the
scientific contract, and `REPRODUCIBILITY.md` for the formal reproducibility
contract that every release must honour.

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
