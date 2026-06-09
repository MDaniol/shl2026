# Reproducibility contract

This project commits to the following contract. Every release (tag, Zenodo
record, paper submission) MUST honour it; pull requests that break it must be
either fixed or accompanied by an explicit, documented amendment to this file.

## The contract

> Given the tuple
> **(git SHA × container SHA-256 × `MANIFEST.sha256` × DVC pipeline hashes ×
> `params.yaml` hash)**,
> running `dvc repro format_submission` inside the named Apptainer image on
> Athena MUST produce a `teamName_predictions.txt` whose SHA-256 matches the
> one logged in the corresponding MLflow run, modulo the documented
> floating-point-nondeterminism allowances below.

## Components of the tuple

| Component                   | Where it lives                                                      | Who writes it                |
|-----------------------------|---------------------------------------------------------------------|------------------------------|
| git SHA                     | `.git`                                                              | every commit                 |
| container SHA-256           | `containers/shl2026_<gitsha>.sif` + tag on the GitHub release       | `scripts/build_container.sh` |
| `MANIFEST.sha256`           | `$PLG_GROUPS_STORAGE/<GRANT>/shl2026/data/raw/MANIFEST.sha256`      | `scripts/verify_data.py`     |
| DVC pipeline hashes         | `dvc.lock`                                                          | `dvc repro`                  |
| `params.yaml` hash          | `params.yaml` (SHA-256 logged per run)                              | author of the change         |

## Declared nondeterminism allowances

- **CUDA-level nondeterminism**: some PyTorch kernels (e.g., certain
  reductions, attention kernels in cuDNN) are not bitwise reproducible across
  runs even with `torch.use_deterministic_algorithms(True)`. We:
  - set `PYTHONHASHSEED`, `torch.manual_seed`, `numpy.random.seed`,
    `random.seed`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
  - enable `torch.use_deterministic_algorithms(True)` where the frozen FM's
    forward pass supports it;
  - log per-run whether full determinism was achieved.
- When full bitwise determinism is unattainable for a given FM, we record the
  tolerance (max abs diff on logits, max symmetric-difference in argmax) in
  the MLflow run and in the release notes.

## What is logged per run (MLflow)

- params: full Hydra-resolved YAML, `params.yaml` hash, foundation-model id +
  HF revision SHA, all seeds.
- tags: `git_sha`, `git_dirty` (must be `false` for any release run),
  `container_sha256`, `slurm_job_id`, `helios_node`, `plgrid_grant`.
- metrics: macro-F1, per-class F1, per-user F1, per-location F1, latency
  (ms / window), peak GPU memory.
- artifacts: resolved config YAML, confusion matrices, learning curves, the
  submission validator's report, the submission file itself (small enough).

## What submission requires before send

1. `git status` clean.
2. Container SHA-256 matches the recorded build for the current git SHA.
3. `dvc repro` produces no changes (i.e., the cache is up to date).
4. `pytest -q` green (CPU smoke).
5. Submission validator passes (shape, dtype, label range, no NaN).
6. A GitHub release tag points at the same commit; Zenodo DOI minted.

## When this contract changes

Open a PR titled `repro:` that touches both this file and the changelog. Two
reviewers required. No silent reproducibility drift.
