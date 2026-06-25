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
Same `setup_env.sh` as Athena/Ares; `uv` selects aarch64 wheels automatically. Tier 1 needs no
torch, so this is light.
```bash
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv missing
./scripts/setup_env.sh                                            # core+dev env from the lock (aarch64, NO torch)
source notebooks/mdaniol/hpc/env_mdaniol.sh
uv pip install lightgbm                                           # the one Tier-1 dep not in the lock
python -c "from shl2026 import track, evaluate_predictions; import lightgbm; print('helios env OK')"
```
If `uv sync` ever balks on aarch64, fall back to a plain venv:
`python -m venv $SCRATCH/venvs/shl2026 && source .../activate && pip install -e . lightgbm pytest`.

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

## Later (optional, harder): Helios GH200 for FM extraction
Only if you want the GH200's speed for embedding extraction. Requires an **aarch64 Apptainer
image** with torch (aarch64+CUDA/sbsa build) + the FM stack (`scripts/build_container.sh` is the
starting point), grant `plgshl26-gpu-gh200` / `plgrid-gpu-gh200`. Non-trivial — keep extraction on
Athena unless we invest in this.
