# Onboarding demo — my run-through (lead notes)

~60 min, live, students follow along. **One goal:** by the end everyone has run
an experiment and seen their own name on the shared leaderboard. That one act
exercises the whole stack (team env → SDK → `track()` → MLflow → `leaderboard()`),
so if it works for them live, everything works.

This is *my* script — every command spelled out the way I'd actually type it, so
I never have to flip to another doc mid-session. Real values are baked in
(`172.23.30.9:5000`, `plgshl26-gpu-a100`, group paths); where a student types
their own thing it's written `<name>`. Pick **one** path per session:
**A — JupyterHub** (default) or **B — SSH + Slurm** (also teaches the cost model).

Shorthand I use below:
- **env.sh** = `$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh` (one `source`
  gives Python + all libs + the shared embedding cache + the MLflow URI).
- **URI** = `http://172.23.30.9:5000` (login01 ib0 — what compute nodes can reach).

---

## Pre-flight — the day before (don't skip)

Each maps to something that actually bit us. Run ~24 h ahead so there's time to fix.

**1. ⚠ The demo-killer (Path A): the team kernel works.** In a notebook, pick the
**SHL 2026 (team)** kernel, then run:
```python
import sys, os
print(sys.executable)                          # want: .../shl2026/venv/bin/python
print(os.environ.get("MLFLOW_TRACKING_URI"))   # want: http://172.23.30.9:5000
```
If `sys.executable` is a `/net/software/...` Python, the kernel isn't
registered/selected → run `./scripts/register_kernel.sh` once and pick the
kernel. If the URI is `None`, the kernelspec didn't carry the env → re-run the
script (the notebook's config cell is the backstop). Either failure means runs
log to a local folder and the leaderboard payoff dies for the whole room.
**Nothing else matters until both print correctly.** Also confirm the shared
venv is group read-only, so no student can break it for everyone:
```bash
ls -ld "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/venv"   # want drwxr-xr-x (no group 'w')
```

**2. Server is up and reachable from where they run.** From a Jupyter terminal
or an `srun` shell:
```bash
curl -s http://172.23.30.9:5000/health    # expect: OK
```

**3. Synthetic cache is seeded** (so `embeddings("synthetic", …)` works for
everyone):
```bash
python -c "from shl2026.data.embeddings import list_available; print(list_available())"
# expect a list containing ('synthetic', 'v1')
```
If missing, seed it once (also in SETUP.md §B.5):
```bash
source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"
python -c "from shl2026.data.synthetic import make_synthetic_embedding_cache as m; \
           import os; print(m(os.environ['SHL_EMB_CACHE']))"
chmod -R g+rX "$SHL_EMB_CACHE"
```

**4. Leaderboard starts clean.** In the UI delete `ZZ_SMOKE` and any test runs,
so the board is empty and fills with their names live.

**5. (Path B only) ready the script + dry-run it myself:**
```bash
cd ~/shl2026 && git pull        # make sure notebooks/_demo/exp.py is present
sbatch scripts/student_job.sbatch notebooks/_demo/exp.py
squeue --me                     # watch PD → R → gone
cat slurm-<jobid>.out           # expect result.summary(), no "MLflow unavailable"
```

**6. Everyone pre-spawns once** before we start — first spawn is slow and surfaces
access surprises while there's still time.

**7. My screen: leaderboard on the projector.** From my laptop:
```bash
ssh -L 5000:localhost:5000 plgmdaniol@athena.cyfronet.pl
# then open http://localhost:5000 and leave it up
```

---

## Open (both paths) — 90 seconds, just say it

The FMs stay **frozen** — no fine-tuning allowed. So the expensive GPU step
(running an FM over ~1M windows → **embeddings**) is done **once by the team and
shared**. Everything *you* do — the head, feature tricks — runs on those cached
embeddings in **seconds**. Today we use a built-in **synthetic** cache so nobody
waits on data. The loop you learn now is the loop you'll use all month. Goal for
the next hour: your name on that board (point at projector).

---

## Path A — JupyterHub

**A1. Spawn a session.** JupyterHub → spawn on Athena: account
`plgshl26-gpu-a100`, partition `plgrid-gpu-a100`, **smallest** profile (1 GPU,
few cores, short walltime). Say it plainly: *an open session holds an A100 and
bills the whole time — stop it when you're done.*

**A2. Open a terminal in JupyterLab and bring up the project.** Clone the
skeleton (small — only their code lives here):
```bash
git clone https://github.com/MDaniol/shl2026.git && cd shl2026
```
Activate the team env (Python + libs + cache + MLflow URI, one line):
```bash
source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"
```
Make every future session do it automatically:
```bash
echo 'source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"' >> ~/.bashrc
```
**Prove it took** — this line is the whole game (it's why runs reach the team,
not a local folder):
```bash
python -c "import os; print(os.environ['MLFLOW_TRACKING_URI'])"
# expect: http://172.23.30.9:5000
```
Then register the team env as a notebook **kernel** (once) — a kernel doesn't
read `env.sh`, so this is what lets the notebook `import shl2026` and reach the
board:
```bash
./scripts/register_kernel.sh        # creates the "SHL 2026 (team)" kernel
```

**A3. Their own notebook.** Everyone works in their own folder, never edits
someone else's:
```bash
mkdir -p notebooks/<name>
cp notebooks/template_experiment.ipynb notebooks/<name>/my_first.ipynb
```
Open `notebooks/<name>/my_first.ipynb`, then **pick the kernel: Kernel → Change
Kernel → "SHL 2026 (team)"**. The template's first cell prints where the kernel
points — confirm it's the venv and the team URI:
```python
import sys; print(sys.executable)   # want: .../shl2026/venv/bin/python
# the config cell already prints:  MLflow → http://172.23.30.9:5000
```

**A4. Walk the cells.** This is the inner loop — type it live, explain as I go:
```python
from shl2026 import embeddings, make_head, evaluate, track, leaderboard

train = embeddings("synthetic", "train")        # instant, from the shared cache
val   = embeddings("synthetic", "validation")

with track("<name>", run_name="first-run", seed=0) as run:   # <name> = your experiment
    head = make_head("logreg", seed=0).fit(train.X, train.y)
    result = evaluate(head, val)                 # challenge metric = macro-F1
    run.log_eval(result)                         # auto-logged with full provenance

print(result.summary())
```
Point out: the `make_head(...)` line and anything done to `train.X` before
`.fit` is *theirs to vary*; everything else is plumbing they never touch. →
`result.summary()` prints a macro-F1.

**A5. Payoff.** New cell:
```python
leaderboard()
```
Then refresh the UI on the projector. → **their name shows up on the shared
board.** Let it land — this is the moment.

**A6. Vary one thing.** Change `"logreg"` → `"mlp"` in the `make_head` line,
re-run the cell, refresh the board — it reorders. *That's the whole job, all
month: change one thing, re-run, compare.*

**A7. Wrap.** **Stop the session:** File → Hub Control Panel → Stop (billing!).
Then: `STUDENTS.md` is your reference from here; claim a lane in `IDEAS.md`
before sinking a day into an idea.

---

## Path B — SSH + Slurm

Same goal, no notebook. The queue wait isn't dead air — it *is* the cost lesson.

**B1. SSH in.** Each student uses their own PLGrid login:
```bash
ssh <login>@athena.cyfronet.pl
```
Say it: this is a **login node** — fine to edit and submit jobs, **never** to
compute on.

**B2. Bring up the project** (same three lines as Path A):
```bash
git clone https://github.com/MDaniol/shl2026.git && cd shl2026
source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"
echo 'source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"' >> ~/.bashrc
```
Prove it took:
```bash
python -c "import os; print(os.environ['MLFLOW_TRACKING_URI'])"
# expect: http://172.23.30.9:5000
```

**B3. The script.** Open `notebooks/_demo/exp.py` — it's the *same SDK code* as
the notebook, with `print(...)` so output lands in the job log:
```python
from shl2026 import embeddings, make_head, evaluate, track

train = embeddings("synthetic", "train")
val   = embeddings("synthetic", "validation")

with track("<name>", run_name="first-batch", seed=0) as run:
    result = evaluate(make_head("logreg", seed=0).fit(train.X, train.y), val)
    run.log_eval(result)

print(result.summary())
```
Each student copies it to their own folder and sets `<name>`:
```bash
mkdir -p notebooks/<name>
cp notebooks/_demo/exp.py notebooks/<name>/exp.py
# edit notebooks/<name>/exp.py → set track("<name>", ...)
```

**B4. Submit + the cost moment.** Submit to a compute node:
```bash
sbatch scripts/student_job.sbatch notebooks/<name>/exp.py
squeue --me
```
*While it waits*, teach the model: `PD` = queued (free), `R` = running (billing),
gone = done. A batch job bills only while it runs — minutes, not the whole
session. → `PD` → `R` → gone.

**B5. Read the log** (filename uses the job id from `squeue`):
```bash
cat slurm-<jobid>.out
```
→ `result.summary()`, and **no** `MLflow unavailable` warning (that warning means
it logged nowhere useful).

**B6. Payoff** — light enough for the login node:
```bash
python -c "from shl2026 import leaderboard; print(leaderboard().head(15))"
```
Refresh the UI. → **their name on the board.**

**B7. Wrap.** Contrast with JupyterHub: batch = no idle A100 on the clock.
`STUDENTS.md`, then `IDEAS.md`.

---

## Bonus — "I need a library that's not in the shared env" (optional, show if asked)

The rule first: **don't copy or `pip install` into the shared venv** — it's
read-only on purpose, so nobody can break it for everyone. You make your *own*
isolated venv from the same Python and add your package there. We **create, not
clone**: venvs aren't portable (a copy breaks its paths), and isolation is
exactly what stops one person's install from hitting everyone else.

One-time, in a terminal:
```bash
export UV_CACHE_DIR="$SCRATCH/uv-cache"
uv venv "$SCRATCH/venvs/mine" \
    --python "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/venv/bin/python"   # same Python as the team
cd ~/shl2026 && ln -sfn "$SCRATCH/venvs/mine" .venv
source .venv/bin/activate
uv pip install -e ".[dev]" seaborn        # team packages + your extra (example: seaborn)
```
→ this venv has everything the team venv has **plus** `seaborn`, lives on
`$SCRATCH` (big — `$HOME` stays clean), and touches no one else.

Use it:
- **Notebook:** register it as your own kernel, then pick it —
  ```bash
  python -m ipykernel install --user --name shl2026-mine --display-name "SHL 2026 (mine)"
  ```
  Kernel → Change Kernel → **"SHL 2026 (mine)"**.
- **Scripts:** nothing extra — `scripts/student_job.sbatch` auto-activates
  `./.venv` on top of the team env, so `sbatch` just uses it.

Honesty rule: a run from a personal venv is **provisional** (its MLflow
`python_env` tag ≠ the team venv). When it's worth keeping → tell the lead → they
pin the package in the shared env → you re-run once in the team kernel (seconds)
→ now it's reproducible and submittable.

---

## If something breaks (live-fire)

- `uv: command not found` right after install → `export PATH="$HOME/.local/bin:$PATH"`.
- notebook `ImportError: … zmq Cython backend … not compiled` (or ipykernel/jupyter
  errors) → cluster modules pollute `PYTHONPATH` with system py3.13 packages →
  `unset PYTHONPATH`, re-run `./scripts/register_kernel.sh`, pick the team kernel.
  (env.sh and the kernelspec now clear `PYTHONPATH` automatically.)
- `leaderboard()` empty or **only my runs** → `env.sh` not sourced in this
  shell/kernel, so it logged local. `source "$PLG_GROUPS_STORAGE/plggmhealth/shl2026/env.sh"`,
  then re-check `python -c "import os; print(os.environ['MLFLOW_TRACKING_URI'])"`.
- `MLflow unavailable … No route to host` → wrong URI (public hostname) or server
  down. It must be `http://172.23.30.9:5000`; confirm with
  `curl -s http://172.23.30.9:5000/health`.
- batch `can't open file …: No such file or directory` → script is on `/tmp`
  (node-local). Keep it in the repo (`notebooks/<name>/`, on `$HOME` — every node
  sees it).
- `no cached embeddings for fm='synthetic'` → shared cache not seeded (pre-flight 3).
- `FileNotFoundError` / relative cache path → `SHL_EMB_CACHE` unset → source env.sh.
- `Disk quota exceeded` in `$HOME` → something heavy landed in the 10 GB home; it
  all belongs in group storage (`hpc-fs` shows usage), keep `$HOME` to code.

---

## After

Their reference is [`STUDENTS.md`](STUDENTS.md) (inner loop, honesty rules,
submission). Claim a lane in [`IDEAS.md`](IDEAS.md). Cluster details →
[`docs/CLUSTER.md`](docs/CLUSTER.md).
