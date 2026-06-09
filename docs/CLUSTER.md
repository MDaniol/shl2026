# Cluster guide — Athena (ACK Cyfronet / PLGrid)

Students' daily work happens in JupyterHub and is fully covered by
[`STUDENTS.md`](../STUDENTS.md). This file is the layer underneath: what the
cluster is, how batch jobs run, and the commands for when something is stuck.
Infrastructure facts verified against Cyfronet/PLGrid docs and `hpc-grants`
(2026-05-31 / 2026-06-02).

---

## Our setup at a glance

Everything lives on **Athena** (x86_64, A100 GPUs): embedding extraction, the
student inner loop (JupyterHub), the shared cache, MLflow, and the container —
one cluster, one filesystem, no syncing.

| | `--account` | `--partition` | Budget | Role |
|---|---|---|---|---|
| **Athena** (primary) | `plgshl26-gpu-a100` | `plgrid-gpu-a100` (48 h) | 15,000 GPU-h | everything |
| Helios (fallback) | `plgshl26-cpu` | `plgrid` (72 h) | 5,000 CPU-h | CPU jobs if Athena is exhausted/busy |
| Helios GH200 (avoid) | `plgshl26-gpu-gh200` | `plgrid-gpu-gh200` (48 h) | 5,000 GPU-h | aarch64 — needs a separate image |

> **Athena is GPU-only.** Every job and JupyterHub session must request a GPU
> (`--gres=gpu:1`), and Cyfronet may suspend accounts running non-GPU workloads
> there. Billing is un-normalised — cost = `max(cpu, mem, gpu) × walltime` — so
> an idle session burns A100-hours. Spawn → work → stop.

> **Fallback caveat:** group storage is **per-cluster** (`$PLG_GROUPS_STORAGE`
> = `pr2` on Athena, `pr3` on Helios — separate filesystems). Activating the
> Helios fallback means one `rsync` of the embedding cache + `.sif` (tens of
> GB) and an `MLFLOW_TRACKING_URI` reachable from there. Lead's call.

### Storage

| Area | Path | Size | Notes |
|---|---|---|---|
| Home | `$HOME` | 10 GB | backed up; **code only** |
| Scratch | `$SCRATCH` | large | **auto-purged** (files >30 d, job dirs >7 d); never keepables |
| **Project root** | `$PLG_GROUPS_STORAGE/plggmhealth/shl2026` | 6 TB (group quota) | everything durable |

```
$PLG_GROUPS_STORAGE/plggmhealth/shl2026/
├── env.sh               # ← students source this: env vars + shared venv
├── venv/                # the one shared Python env (keeps $HOME empty)
├── repo/                # canonical clone the venv installs from (lead-managed)
├── data/raw/            # immutable SHL Challenge data + MANIFEST.sha256 (chmod a-w)
├── data/embeddings/     # the shared cache  ← $SHL_EMB_CACHE
├── models/foundation/   # frozen FM weights (pinned by HF revision SHA)
├── mlflow/              # tracking DB + artifacts
├── dvc-cache/           # DVC remote
└── containers/          # the x86_64 .sif image(s)
```

Check quotas/usage with `hpc-fs`; grants and Slurm account names with
`hpc-grants`; your jobs with `hpc-jobs` / `hpc-jobs-history`.

---

## The one rule: login node ≠ compute node

`ssh <login>@athena.cyfronet.pl` lands you on a **login node** — for editing,
submitting jobs, and light file work only. Heavy processes there get killed.
Real work runs on **compute nodes**, reached only through Slurm: a batch job
(`sbatch`), an interactive shell (`srun --pty`), or JupyterHub (which spawns a
Slurm job for you).

## Slurm in one breath

You describe a job — command, resources, walltime, which grant pays — and the
scheduler queues it and runs it when resources free up.

```bash
sbatch job.sbatch            # submit → "Submitted batch job 1234567"
sbatch --test-only job.sbatch  # validate + estimate start; consumes NOTHING
squeue --me                  # my jobs: PD=pending, R=running
squeue --me --start          # estimated start of pending jobs
scancel 1234567              # cancel
sacct -j 1234567 --format=JobID,State,Elapsed,MaxRSS,ReqMem,ExitCode  # post-mortem
scontrol show job 1234567    # full detail incl. why it's waiting
```

Interactive shell on a compute node (Athena → must include a GPU):

```bash
srun -A plgshl26-gpu-a100 -p plgrid-gpu-a100 --gres=gpu:1 \
     -N 1 -n 1 --cpus-per-task=4 --time=1:00:00 --pty /bin/bash -l
```

## A real batch script (embedding extraction)

`scripts/submit.sh` generates and submits this shape with provenance baked in
(git SHA, container hash) — shown here so it isn't magic:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=shl2026-embed
#SBATCH --account=plgshl26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1               # 1 A100 (a node has 8)
#SBATCH --cpus-per-task=16         # ~node_cores/8 per GPU
#SBATCH --mem=120G                 # ~node_mem/8 per GPU
#SBATCH --time=04:00:00            # honest estimate — shorter queues faster
#SBATCH --output=logs/%x-%j.out    # %x=job name, %j=job id
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load apptainer              # load modules INSIDE the job, not at login

GROUP=$PLG_GROUPS_STORAGE/plggmhealth/shl2026
apptainer exec --nv \
  --bind "$GROUP/data:/data" \
  --bind "$GROUP/models:/models:ro" \
  "$GROUP/containers/shl2026.sif" \
  dvc repro frozen_embeddings
```

**Job arrays** for sweeps (one submission, N independent tasks):

```bash
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
HEADS=(linear logreg mlp mlp-256 mlp-512)
apptainer exec image.sif python -m shl2026 train-head --head "${HEADS[$SLURM_ARRAY_TASK_ID]}"
```

## Apptainer in one minute

One versioned x86_64 image = byte-identical software for every certified run.
Built by the lead (`scripts/build_container.sh`), stored in group storage.

```bash
module load apptainer
apptainer exec --nv image.sif <command>   # --nv exposes the GPU; forget it = no CUDA
apptainer shell image.sif                 # interactive shell inside
# --bind src:dst[:ro] mounts host paths into the container
```

## Why is my job pending? (squeue REASON)

| Reason | Meaning | Do |
|---|---|---|
| `Priority` / `Resources` | normal queue load | wait; shorter `--time` backfills sooner |
| `QOSMaxWallDurationPerJobLimit` | `--time` over the partition cap (48 h) | lower it |
| `AssocGrpBillingMinutes` | grant hours exhausted | `hpc-grants`; tell the lead |
| `InvalidAccount` / `PartitionConfig` | account↔partition mismatch | Athena = `plgshl26-gpu-a100` + `plgrid-gpu-a100` |
| `ReqNodeNotAvail` | maintenance / no node | `sinfo -p plgrid-gpu-a100`; retry later |

Other classics: job died at exactly `--time` → `TIMEOUT` (raise it honestly);
`OUT_OF_MEMORY` → raise `--mem` (check real use: `sacct … MaxRSS`); GPU job sees
no GPU → missing `--gres=gpu:1` or apptainer `--nv`.

## Shell survival kit

```bash
# ~/.ssh/config on your laptop, then just `ssh athena`
Host athena
    HostName athena.cyfronet.pl
    User <your-plgrid-login>
```

```bash
# transfer (run on your LAPTOP; rsync resumes and skips unchanged files)
rsync -avh --progress dir/ athena:~/dest/
rsync -avh athena:~/results/ ./results/

# MLflow UI from your laptop (server runs on an Athena login node)
ssh -L 5000:localhost:5000 athena      # → open http://localhost:5000

# keep transfers / the MLflow server alive after logout (login node only)
tmux new -s work   # detach: Ctrl-b d ; back: tmux attach -t work

# essentials
ls -lhrt           # newest last — "what did my job just write?"
tail -f logs/embed-1234567.out
du -sh * ; hpc-fs  # who is eating the quota
module avail / module load apptainer / module purge
```

Gotchas: `rm -r` has no undo; `nvidia-smi` works only on GPU compute nodes;
don't run computation in `tmux` on a login node — that's what Slurm is for.

---

### Sources
- PLGrid: guide.plgrid.pl/en/computing/slurm (accounts, partitions, `srun --pty`)
- Cyfronet: docs.hpc.cyfronet.pl — supercomputers/athena, environment/batch-system
  (script structure, modules-in-script, `--test-only`, billing), software/jupyterhubhpc
- Grant facts: `hpc-grants` / `hpc-fs` on Athena + Helios, 2026-06-02 (grant
  `plgshl26`, group `plggmhealth`)
- Open items: on-cluster Apptainer build policy (ask Helpdesk), A100 40 GB
  headroom for the largest FMs (check at first extraction), pr2↔pr3 transfer
  route (only if the Helios fallback is ever activated)
