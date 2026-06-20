# SHL 2026 — Running the pipeline on Athena (HPC)

Every experiment has both a **local** runner (`notebooks/mdaniol/{feature_extraction,modeling}/`)
and an **Athena SLURM** version here. Athena (A100/CUDA) is recommended for the
embedding sweep and heavier models — far faster and with robust GPU memory
management (the MacBook MPS path is memory-fragile; see `extract_embeddings.py`
notes). Account/partition placeholders below must be edited to your grant.

## 0. One-time setup on Athena
```bash
# place repo + data under group storage, e.g.:
export PROJECT_DIR=$PLG_GROUPS_STORAGE/plggmhealth/shl2026
# ensure $PROJECT_DIR/dataset_parquet exists (run scripts/raw_to_parquet.py there,
# or rsync the local dataset_parquet/ up).
bash notebooks/mdaniol/hpc/setup_athena_env.sh        # builds $PROJECT_DIR/SHL-local
```

## 1. Features (CPU) — already have `feature_extraction/extract_features.sbatch`
```bash
sbatch notebooks/mdaniol/feature_extraction/extract_features.sbatch   # 9-task array -> dataset_parquet_features/
```

## 2. Track-A baseline + calibration + submission (CPU)
```bash
sbatch notebooks/mdaniol/hpc/train_baseline.sbatch
```

## 3. Frozen-FM embeddings (GPU array, one model per task)
```bash
sbatch notebooks/mdaniol/hpc/extract_embeddings.sbatch          # MODELS=(mantisv2 utica mantis8m moment-small)
# more models: edit the MODELS=() list + --array range; e.g. add moment-base, moment-large
```
A100 settings: `--device cuda --chunk-size 8000 --tf-batch 512` (≈5–10× the laptop).

## 4. Probe + fusion per embedding set (CPU array, big RAM)
```bash
sbatch notebooks/mdaniol/hpc/probe_fusion.sbatch               # EMBS=(mantisv2_V0 utica_V0 …)
```
Appends val test-loc macro-F1 (logreg / lgbm / lgbm+handcrafted fusion) to
`notebooks/mdaniol/BAKEOFF_RESULTS.md`.

## Dependency notes (match the working local pins)
- `transformers==4.44.2`, `huggingface_hub==0.25.2`, `tokenizers<0.20` — newer
  versions break `mantis-tsfm`'s `PyTorchModelHubMixin.from_pretrained`.
- `momentfm` / `mantis-tsfm` installed `--no-deps` (their pins drag in an
  unbuildable old transformers). See `setup_athena_env.sh`.
- Mantis `from_pretrained` is an **instance** method: `MantisV2(device=…).from_pretrained(repo)`.
- UTICA = `Mantis8M` arch + `load_state_dict(strict=False)` of `fegounna/Utica`.

## Convention going forward
Each new experiment ships a local script **and** an Athena `*.sbatch` here, sharing
the same `PROJECT_DIR`/`SHL-local`/`dataset_parquet*` layout so local↔HPC are
drop-in. MLflow logging (server `172.23.30.9:5000`) to be wired into the probe/
fusion runs for traceability.
