# SHL 2026 — Running on Athena (reuses the team setup)

Uses the **existing** team workflow (`scripts/setup_env.sh` + `env.sh` +
`scripts/student_job.sbatch` conventions). It does **not** touch the shared
`pyproject.toml` / `uv.lock`: you build *your own* env from the team lock, then
add the FM deps to *your* env only.

## 0. Build your env (once)
```bash
cd $REPO                                   # your clone (per STUDENTS.md, $HOME)
./scripts/setup_env.sh                      # team lock -> your env at $SCRATCH/venvs/shl2026 + kernel
bash notebooks/mdaniol/hpc/add_fm_deps.sh   # add torch + momentfm/mantis/lightgbm/... to YOUR env
```

## 1. Data
`dataset_parquet` already lives in group storage. Either symlink it into the repo
so default paths resolve, or pass `--data-dir`:
```bash
ln -s $PLG_GROUPS_STORAGE/plggmhealth/shl2026/dataset_parquet "$REPO/dataset_parquet"
```
Heavy **outputs** (features, embeddings) should live on `$SCRATCH` (keep `$HOME`
free) — symlink them in, or pass `--feat-dir/--emb-dir`:
```bash
mkdir -p $SCRATCH/shl2026/{dataset_parquet_features,embeddings}
ln -s $SCRATCH/shl2026/dataset_parquet_features "$REPO/dataset_parquet_features"
ln -s $SCRATCH/shl2026/embeddings               "$REPO/embeddings"
```

## 2. Submit (all `.sbatch` source `env.sh` → your env; Athena GPU settings baked in)
```bash
cd "$REPO"; export PROJECT_DIR="$REPO"
sbatch notebooks/mdaniol/feature_extraction/extract_features.sbatch   # CPU work (Athena needs a GPU anyway)
sbatch notebooks/mdaniol/hpc/variant_sweep.sbatch                    # FM V1/V2 sweep (extract+probe)
sbatch notebooks/mdaniol/hpc/extract_embeddings.sbatch              # V0 embeddings
sbatch notebooks/mdaniol/hpc/probe_fusion.sbatch                    # after embeddings
sbatch notebooks/mdaniol/hpc/train_baseline.sbatch                  # handcrafted baseline + submission
```
For a single ad-hoc script you can also use the team template directly:
`sbatch scripts/student_job.sbatch notebooks/mdaniol/modeling/<script>.py [args]`.

## 3. Monitor
```bash
squeue --me ; tail -f notebooks/mdaniol/modeling/logs/*.out ; cat notebooks/mdaniol/BAKEOFF_RESULTS.md
```

## Notes
- **Athena is GPU-only** → every `.sbatch` here uses `plgshl26-gpu-a100` /
  `plgrid-gpu-a100` / `--gres=gpu:1` (matches `scripts/student_job.sbatch`).
- The shared `pyproject.toml`/`uv.lock` are untouched; FM deps live only in your
  personal env (`uv pip install`, pinned so mantis-tsfm loads).
- Local Mac dev still uses the `SHL-local` venv (same packages); on the cluster
  use the team env above.
