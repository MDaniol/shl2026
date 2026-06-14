# SHL 2026 — student guide

Everything you need is in this one file: one-time setup → first run → how we
work. One month, 12 of us, one shared skeleton on **Athena** (ACK Cyfronet).

## The big idea (read this once)

The challenge forbids fine-tuning the foundation models — they stay **frozen**.
That's a gift: the only expensive, GPU-heavy step (running a FM over the ~1M
SHL Challenge windows to get **embeddings**) is done **once by the team and
shared**. Everything *you* prototype — the classifier head, feature transforms,
fusion tricks — runs on those cached embeddings in **seconds**. You never touch
the raw SHL data, the foundation models, or Docker.

---

## Part 1 — One-time setup (~30 min of clicking + a wait for approvals)

Approvals need a human, so **start today** even if you won't code until next week.

1. **Register on PLGrid:** <https://portal.plgrid.pl> → *Register*, signing in
   through your **home institution (AGH)** — not a private Google/email account.
   Your PLGrid login is usually `name.surname`.
2. **Send the lead your PLGrid login** (contact at the bottom). The lead adds you
   to grant **`plgshl26`** / group **`plggmhealth`**; accept the invitation in
   the portal when it appears.
3. **In the portal, request access to:** **Athena** and **Jupyter** (JupyterHub).
   Each is approved by a human — request both now. Also add an **SSH key**
   (Profile → SSH keys; `ssh-keygen -t ed25519`, paste `~/.ssh/id_ed25519.pub`):
   it's how you work while Jupyter approval is pending (see *the SSH + batch
   path* below), and your fallback after.
4. **Open JupyterHub** (link on the PLGrid Jupyter service page) and spawn a
   session on **Athena**:
   - account **`plgshl26-gpu-a100`**, partition **`plgrid-gpu-a100`**
   - the **smallest profile**: 1 GPU, a few cores, a short walltime.

   > Athena is a GPU cluster — every session reserves an A100 and **bills the
   > team's GPU-hours the whole time it is open**. Spawn → work → **Stop the
   > session** (File → Hub Control Panel → Stop) when you take a break.
5. **In a Jupyter terminal, build your environment** (one command — it creates
   *your own* copy of the team environment from the shared lockfile, plus a
   Jupyter kernel from it; the lock means everyone gets identical versions):

   ```bash
   # Clone the team skeleton (your code + the shared spec live here)
   git clone https://github.com/MDaniol/shl2026.git && cd shl2026

   # Build YOUR env (on $SCRATCH) from the shared lock + register a kernel:
   ./scripts/setup_env.sh

   # For terminals & batch jobs, auto-activate it (notebooks use the kernel):
   echo 'source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"' >> ~/.bashrc
   ```

   > **First time only:** restart your server so JupyterHub sees the new kernel —
   > **File → Hub Control Panel → Stop My Server**, then **Start**. Then pick
   > **"SHL 2026"** as the notebook kernel.

   > It's **your** env: `uv pip install` anything you like into it without
   > affecting anyone else. It lives on `$SCRATCH` (keeps `$HOME` free) — if it's
   > ever auto-purged, rebuild with `./scripts/setup_env.sh`. Want a package in
   > the *shared* spec? See *Need an extra library?* in Part 2.
6. **First run** — in a notebook (copy `notebooks/template_experiment.ipynb`
   into `notebooks/<your-name>/` first, then pick the kernel **Kernel → Change
   Kernel → "SHL 2026"**):

   ```python
   from shl2026 import embeddings, make_head, evaluate, track, leaderboard

   train = embeddings("synthetic", "train")      # instant, from the shared cache
   val   = embeddings("synthetic", "validation")

   with track("YOUR_NAME", run_name="first-run", seed=0) as run:
       head = make_head("logreg", seed=0).fit(train.X, train.y)
       run.log_eval(evaluate(head, val))          # challenge metric = macro-F1

   leaderboard()                                  # you should see your run here
   ```

   If your run shows up in `leaderboard()`, **setup is done.**

`"synthetic"` is a built-in fake cache so you can start before (or alongside)
the real ones. Real SHL embeddings appear in the same cache as the team extracts
them — `list_available()` shows what's there (e.g. `"moment"`); just swap the
name, nothing else changes.

### No JupyterHub (yet)? The SSH + batch path

Everything works over plain SSH — you run scripts instead of notebooks. You
need Athena access + the SSH key from step 3, nothing else.

1. `ssh <your-plgrid-login>@athena.cyfronet.pl` — you land on a **login node**:
   fine for setup, editing, and submitting jobs, **never for computation**.
2. Do step 5 above exactly as written (clone + `source …/env.sh` → `~/.bashrc`).
3. Put your experiment in a plain script — `notebooks/<your-name>/exp.py` with
   the code from Part 2 below (the `print(...)` lines are what you'll see in
   the job log).
4. Run it on a compute node, from the repo root:

   ```bash
   sbatch scripts/student_job.sbatch notebooks/<your-name>/exp.py
   squeue --me                 # PD = queued, R = running, gone = finished
   cat slurm-<jobid>.out       # your script's output (incl. result.summary())
   ```

   For fast trial-and-error, take an interactive shell instead — but remember
   it bills until you `exit`:

   ```bash
   srun -A plgshl26-gpu-a100 -p plgrid-gpu-a100 --gres=gpu:1 \
        --cpus-per-task=4 --time=1:00:00 --pty /bin/bash -l
   source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"
   python notebooks/<your-name>/exp.py   # …iterate, then exit!
   ```

5. Checking the team leaderboard is light enough for the login node:

   ```bash
   python -c "from shl2026 import leaderboard; print(leaderboard().head(15))"
   ```

A batch job bills only while it runs (the inner loop = minutes); details and
debugging in [`docs/CLUSTER.md`](docs/CLUSTER.md).

---

## Part 2 — The inner loop (this is the whole job)

```python
from shl2026 import embeddings, make_head, evaluate, track, leaderboard

train = embeddings("moment", "train")        # shared cache; None = all locations
val   = embeddings("moment", "validation")

with track("YOUR_NAME", run_name="moment|allloc|mlp", seed=0,
           params={"fm": "moment", "head": "mlp"}) as run:
    head = make_head("mlp", seed=0).fit(train.X, train.y)
    result = evaluate(head, val)             # challenge metric = macro-F1
    run.log_eval(result)                     # auto-logged with full provenance

print(result.summary())                      # macro-F1, accuracy, weakest class
leaderboard()                                # how everyone's runs compare
```

Your idea is the `make_head(...)` line (`list_heads()` shows the menu) and
anything you do to `train.X` before `.fit` (PCA, normalisation, concatenating
FMs, …). Everything else is plumbing.

> Every function's signature and return type — plus how to log from a
> hand-written training loop (PyTorch etc.) — is in
> [`docs/API.md`](docs/API.md).

**Notebook or script — your choice.** A plain `my_experiment.py` with the same
code is just as good: keep it in `notebooks/<your-name>/` and run it from the
JupyterHub terminal (`python my_experiment.py`). Never run computation on a
login node.

### What's yours to vary

| Knob | How |
|---|---|
| Foundation model | `embeddings("moment", ...)` vs others — `list_available()` |
| Locations | `embeddings(fm, split, "Hips")`; `None` pools the **samples** of all locations into one set (same feature width) |
| Head + hyperparams | `make_head("mlp", hidden_layer_sizes=(256,), alpha=1e-3)` |
| Feature transform | anything applied to `X` before `.fit` |

Not yours: the embeddings themselves (pooling, FM revision, window size) — those
are baked into the cache. Want a new FM or pooling? Ping the lead; it gets
extracted once for everyone.

### Five rules that keep your numbers honest

1. **Score on `validation`, never `test`** (test has no labels — by design).
2. **The metric is macro-F1, not accuracy.** `result.summary()` shows your
   weakest class — that's usually where the points are. (Classes 1–8 = Still,
   Walk, Run, Bike, Car, Bus, Train, Subway; see `METHODS.md`.)
3. **Expect validation ≪ train.** The split is user-independent (train = user 1,
   validation = users 2&3). The gap is the task, not a bug.
4. **The Hand trap.** `Hand` exists in train/validation but is **dropped at
   test**. Numbers including Hand are optimistic — for an honest comparison use
   `Bag, Hips, Torso`, or say `hand` in your run name.
5. **Fix `seed=0`; re-run close calls with 2–3 seeds** (MLP is seed-sensitive).
6. **Commit & push before runs you care about.** Uncommitted runs show as
   `git_dirty=true` on the leaderboard and can't be certified for submission.
   `track()` auto-attaches a code snapshot (`code_snapshot/` in the run's
   artifacts: diff + new files + executed notebook cells) so even a dirty run
   stays recreatable — but that's the safety net, not the workflow.

**Change one thing per run** and encode it in the run name
(`moment|3loc|mlp-256` beats `test3`) — then `leaderboard()` reads as a clean
ablation. **Claim a lane in [`IDEAS.md`](IDEAS.md) before sinking a day in**, so
12 of us don't re-run the same idea.

### House rules

1. **Your work lives in `notebooks/<your-name>/`** (notebooks *and* scripts).
   Never edit someone else's.
2. **`src/` is the shared skeleton — don't change it.** New head / SDK change?
   Ping the lead; it gets added for everyone.
3. **Never touch `params.yaml`, `data/raw/`, or other people's MLflow runs.**
4. **Push often — via a quick PR.** `main` only takes pull requests, but
   **no approval is needed**: you merge your own PR right away. The loop is:

   ```bash
   git checkout -b <your-name>/<anything> && git add -A && git commit -m "..."
   git push -u origin HEAD
   gh pr create --fill && gh pr merge --squash --delete-branch
   ```

   (~30 s; or click "Merge" on github.com. `git checkout main && git pull`
   before starting the next one.)
5. **`track(...)` every run** — it's one line, and it's how the team sees your
   results and reproduces the winner.
6. **Close idle JupyterHub sessions.** Every open session holds an A100. Big
   sweeps don't belong in a notebook → ask the lead about `sbatch`
   (see [`docs/CLUSTER.md`](docs/CLUSTER.md)).

### Need an extra library?

Your env is your own — just install into it:

```bash
source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"   # activates your env
uv pip install <package>                                   # into YOUR env only
```

Restart the notebook kernel to pick it up. Nobody else is affected.

**The promotion rule.** That install is local to you, so a run that needs it is
*provisional* — the submission container, built from the shared lock, can't
reproduce it (every run is tagged with its `python_env`, so nobody has to
remember). To make it count: tell the lead the package; they run `uv add
<package>` and commit the new `uv.lock`; then `git pull && ./scripts/setup_env.sh`
rebuilds your env on the locked versions and you re-run. Now it's reproducible
and submittable.

## When your result is good enough to submit

Don't email anything yourself. Tell the lead the **MLflow run** (name + git
SHA). The team picks the best validation macro-F1 from the leaderboard,
regenerates its predictions in the container, and submits one
`AGH_predictions.txt` per the SHL Challenge rules.

## Troubleshooting

- **Portal won't let me register** → use the AGH/institutional login option.
- **"Permission denied" pushing** → the repo is public to read, but you need
  write access to push: give the lead your GitHub username.
- **`leaderboard()` shows only my runs, or `ModuleNotFoundError: shl2026`** →
  in a notebook: you're not on the **SHL 2026** kernel — pick it (Kernel →
  Change Kernel). In a terminal/batch: you haven't `source`d `env.sh` this
  session (it's in `~/.bashrc` after step 5).
- **`FileNotFoundError: no cached embeddings for fm=...`** → that FM isn't
  extracted yet; `list_available()` shows what is, `"synthetic"` always works.
- **Batch job says `can't open file: No such file or directory`** for a file
  that's right there → you put it on `/tmp` (or another node-local path).
  Your job runs on a *different machine* that has its own empty `/tmp`. Keep
  scripts where STUDENTS.md puts them — `notebooks/<your-name>/` in the repo
  (it's on `$HOME`, which every node sees).
- **Notebook: `ImportError: … zmq Cython backend … not compiled`** (or other
  ipykernel/jupyter import errors) → cluster modules put a system Python on
  `PYTHONPATH`, shadowing your env. Rebuild + re-register with
  `./scripts/setup_env.sh`, restart the server, and pick the **SHL 2026** kernel.
- **`SHL 2026` isn't in the kernel list, or it says "connecting" then
  disconnects** → JupyterHub only reads kernels when the server *starts*, so it
  hasn't picked up the one you built. **File → Hub Control Panel → Stop My
  Server → Start**, then reselect the kernel.
- **Disk quota exceeded in `$HOME`** → something heavy landed in your 10 GB
  home (a venv, caches, data). Keep `$HOME` to code; the team env and cache
  live in group storage (`hpc-fs` shows your usage).
- **Everything cluster-side** (SSH, Slurm, storage, batch jobs) →
  [`docs/CLUSTER.md`](docs/CLUSTER.md).

## Who to ask

- **Lead / grant manager:** Mateusz Daniol — <daniol@agh.edu.pl> (PLGrid
  grant, access, repo).
- **PLGrid Helpdesk:** `helpdesk@plgrid.pl` — account/portal problems only.
