# Running on Helios (CPU lane) — setup runbook

Helios is **aarch64 (ARM / GH200)** and has **separate storage (pr3)** from Athena (pr2). Use it
for the **CPU lane** (Tier 1, bake-off). Keep **FM extraction on Athena** — Helios GPU needs an
aarch64 container (`docs/CLUSTER.md` flags it "avoid"). Grant: `plgshl26-cpu` / `plgrid` (72h).

## One-time setup

### 1. Repo on Helios
```bash
ssh plgmdaniol@helios.cyfronet.pl
git clone git@github.com:MDaniol/shl2026.git ~/shl2026   # or HTTPS+PAT
cd ~/shl2026 && git checkout mdaniol/shl-fm-pipeline && git pull
```

### 2. Build the aarch64 venv (must be built ON Helios — x86 venv won't work)
Use the Helios setup script — it resolves FRESH from pyproject for aarch64 (the shared uv.lock is
x86, so `uv sync`/`setup_env.sh` does NOT work here). Tier 1 needs no torch, so this is light.
```bash
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv missing
./scripts/setup_env_helios.sh                                    # aarch64 fresh resolve + lightgbm
source notebooks/mdaniol/hpc/env_mdaniol.sh
```

### 3. Copy the data to Helios group storage (pr3)
Helios can't see Athena's scratch. Find the Helios group path, then rsync from Athena.
```bash
# on HELIOS: note the path + make the target
echo $PLG_GROUPS_STORAGE            # e.g. /net/pr3/projects/plgrid/plggmhealth
mkdir -p $PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings

# on ATHENA: push features + the moment-small_V1 embeddings over the network
rsync -a $SCRATCH/shl2026/dataset_parquet_features \
      plgmdaniol@helios.cyfronet.pl:$PLG_GROUPS_STORAGE/plggmhealth/shl2026/
rsync -a $SCRATCH/shl2026/embeddings/moment-small_V1 \
      plgmdaniol@helios.cyfronet.pl:$PLG_GROUPS_STORAGE/plggmhealth/shl2026/embeddings/
```
(`$PLG_GROUPS_STORAGE` in the remote target expands on Athena — if the Helios path differs, paste
it literally. raw data isn't needed for Tier 1.)

### 4. Confirm the grant
```bash
hpc-grants           # confirm account=plgshl26-cpu, partition=plgrid on Helios
```

## Run
```bash
cd ~/shl2026 && git pull
sbatch notebooks/mdaniol/hpc/tier1_helios.sbatch     # Tier-1 on Helios CPU
# bake-off uses the same grant — submit probe_fusion_ares.sbatch on Helios too:
#   sbatch -J shl-probe notebooks/mdaniol/hpc/probe_fusion_ares.sbatch
squeue --me
cat notebooks/mdaniol/TIER1_RESULTS.md
```

## §GPU — FM extraction on Helios GH200 (aarch64) — the harder path
Use this to extract FM embeddings (MOMENT/Mantis/UTICA/Mantis8M) on the GH200. It's fast but
aarch64-risky (the torch-on-GH200 install is the part that may need iteration). **Smoke-test before
committing to a full run.** Grant `plgshl26-gpu-gh200` / `plgrid-gpu-gh200` (48h).

### 1. Build the aarch64 FM env (on Helios)
```bash
cd ~/shl2026 && ./scripts/setup_env_helios.sh      # aarch64 core env (fresh resolve, once)
bash notebooks/mdaniol/hpc/add_fm_deps_helios.sh   # aarch64 torch+CUDA + FM stack
# -> prints torch version + "FM stack imports OK on aarch64" (cuda=False on login is fine)
```
If the torch install/import fails on GH200: check `nvidia-smi` for the CUDA driver and edit the
index in `add_fm_deps_helios.sh` (try `cu126` or `cu121`). If it keeps fighting, fall back to
**Athena** for extraction — the CPU bake-off can still run on Helios afterwards.

### 2. Put the RAW data on Helios group storage (pr3)
Extraction needs the raw windows (not just features). From **Athena**:
```bash
rsync -a $SCRATCH/../dataset_parquet \
      plgmdaniol@helios.cyfronet.pl:$PLG_GROUPS_STORAGE/plggmhealth/   # raw, ~15 GB
#   (raw lives at $PLG_GROUPS_STORAGE/plggmhealth/dataset_parquet — link_data.sh points there)
```

### 3. SMOKE-TEST (cheap — validates the GH200 + measures speed)
```bash
MODEL=mantis8m VARIANTS=V0 LIMIT=2000 sbatch notebooks/mdaniol/hpc/extract_fm_helios.sbatch
tail -f notebooks/mdaniol/modeling/logs/extract_helios_*.out   # expect win/s + no CUDA errors
```
Only if the smoke run succeeds and the speed is acceptable, do the full extraction.

### 4. Full extraction, then publish for the bake-off
```bash
MODEL=mantis8m VARIANTS="V0 V1 V2" sbatch notebooks/mdaniol/hpc/extract_fm_helios.sbatch
# (repeat for mantisv2 / utica / moment-small as needed)
bash notebooks/mdaniol/hpc/stage_to_group.sh emb     # publish embeddings -> group storage
sbatch notebooks/mdaniol/hpc/probe_fusion_ares.sbatch  # CPU bake-off (runs on Helios CPU too)
```

**Reminder:** the *experiment* is unchanged — swapping FMs is just `--model X`. This whole section
is only about making the aarch64 GPU usable; the framework itself is already FM-agnostic.
