# SHL 2026 — SDK reference (`shl2026`)

The lookup page for the team library. **You only ever call ~5 functions.** The
daily workflow (setup, honesty rules, submission) lives in
[`STUDENTS.md`](../STUDENTS.md); this is just *what each function is and what it
returns*, plus how to log from a hand-written training loop.

## The whole inner loop

```python
from shl2026 import embeddings, make_head, evaluate, track, leaderboard

train = embeddings("synthetic", "train")        # shared cache; instant, CPU
val   = embeddings("synthetic", "validation")

with track("alice", run_name="synthetic+mlp", seed=0,
           params={"fm": "synthetic", "head": "mlp"}) as run:
    head   = make_head("mlp", seed=0).fit(train.X, train.y)
    result = evaluate(head, val)                 # challenge metric = macro-F1
    run.log_eval(result)                         # logged with full provenance

print(result.summary())                          # macro-F1, acc, weakest class
leaderboard()                                    # how everyone compares
```

Everything you vary is the `make_head(...)` line and whatever you do to
`train.X` before `.fit`. The rest is plumbing you never change.

## Functions

| Function | Signature | Returns |
|---|---|---|
| `embeddings` | `embeddings(fm, split, location=None, *, revision=None)` | `EmbeddingSet`. `split` ∈ `train`/`validation`/`test`. `location` ∈ `Bag`/`Hips`/`Torso`/`Hand`; `None` pools all locations. `revision` auto-detected when one is cached. |
| `list_available` | `list_available()` | `[(fm, revision), …]` present in the cache (e.g. `[("synthetic", "v1")]`). |
| `make_head` | `make_head(name="logreg", *, seed=0, **kw)` | A scikit-learn `Pipeline` (`StandardScaler` → estimator). `kw` is forwarded to the estimator, e.g. `make_head("mlp", hidden_layer_sizes=(256,), alpha=1e-3)`. |
| `list_heads` | `list_heads()` | `["linear", "logreg", "mlp"]`. |
| `evaluate` | `evaluate(head, eval_set, labels=(1..8))` | `EvalResult`. Needs a labelled set — raises on `test`. |
| `evaluate_predictions` | `evaluate_predictions(y_true, y_pred, labels=(1..8))` | `EvalResult` — use this when you have raw predictions (custom loop, no sklearn `Head`). |
| `track` | `track(experiment, run_name=None, *, params=None, tags=None, seed=None, params_path="params.yaml", tracking_uri=None)` | Context manager yielding a `run` logger (see below). |
| `leaderboard` | `leaderboard(experiments=None, *, metric="macro_f1")` | A `pandas.DataFrame`, one row per run, sorted by validation macro-F1. `None` scans every experiment. |
| `write_submission` | `write_submission(frame_predictions, out)` | Broadcasts per-frame labels to the `92726 × 500` matrix, writes + validates it. Returns a `ValidationReport`. (Usually the lead's job — see `shl2026 format-submission`.) |

> **Labels are `1`–`8`**, not `0`–`7` (1=Still, 2=Walk, 3=Run, 4=Bike, 5=Car,
> 6=Bus, 7=Train, 8=Subway). `evaluate*` expects that space; a 0-indexed model
> needs `+ 1` before scoring (see the custom-loop example).

## What you get back

**`EmbeddingSet`** — frozen-FM features for one `(fm, split[, location])`:

| Attribute | Meaning |
|---|---|
| `.X` | `float32` array, shape `(n, dim)` — feed to `.fit` / your model. |
| `.y` | `int` labels `(n,)` in `1..8`, or `None` for the `test` split. |
| `.n`, `.dim` | sample count and feature width. |
| `.has_labels` | `False` only for `test`. |
| `.fm`, `.revision`, `.split`, `.location`, `.pooling` | provenance of this shard. |

**`EvalResult`** — the score of one prediction set:

| Attribute | Meaning |
|---|---|
| `.macro_f1` | **the challenge metric.** |
| `.accuracy` | overall accuracy. |
| `.per_class_f1` | `{class: f1}` for `1..8` — find your weak class here. |
| `.confusion` | confusion matrix (`np.ndarray`). |
| `.summary()` | one-line string: macro-F1, acc, n, weakest class. |

## Logging: `track()` and the `run` object

`track(...)` opens **one MLflow run** and hands you a `run` logger. You don't
build the MLflow run yourself, and you don't pass URIs around — the **SHL 2026**
kernel (notebooks) or `env.sh` (terminals/batch) sets `MLFLOW_TRACKING_URI`, and
`track` reads it.

**What it captures for free** (as run tags, zero effort): `git_sha`,
`git_dirty`, `container_sha256`, `slurm_job_id`, `helios_node`, `plgrid_grant`,
`params_sha256`, `student`, `python_env`. On a **dirty tree** it also attaches a
`code_snapshot/` artifact (diff vs HEAD + untracked files + executed notebook
cells) so even an uncommitted number stays recreatable. This provenance is the
whole reason a one-line `track()` is enough to certify a run later.

**Methods on `run`** — call any of them anywhere inside the `with` block:

| Call | Use |
|---|---|
| `run.log_eval(result)` | Log an `EvalResult` (macro_f1, accuracy, per-class F1). The usual one. |
| `run.log_metrics({...}, step=None)` | Arbitrary float metrics. Pass `step=epoch` to get a **curve** in the MLflow UI. |
| `run.log_params({...})` | Hyperparameters (or pass them up front via `track(params=...)`). |
| `run.log_artifact(path)` | Attach a file — weights, a plot, a CSV. |
| `run.set_tags({...})` | Extra string tags (or `track(tags=...)`). |

**If MLflow is unreachable** (laptop, server down) `track` prints a warning and
yields a **no-op** `run` — your code still runs, nothing is logged. So an empty
or "only my runs" `leaderboard()` almost always means you're on the wrong kernel
(notebook) or didn't `source env.sh` (terminal/batch), not a bug. See
`STUDENTS.md` → Troubleshooting.

## Integrating a custom training loop

`evaluate(head, set)` only works for things with sklearn's `.fit/.predict`. When
you write your own loop (PyTorch, a hand-rolled head, anything), do two things:
log per-step metrics with `step=`, and score at the end with
`evaluate_predictions(y_true, y_pred)` instead of `evaluate`.

A runnable version of this (mini-batch loop + per-epoch curves) is in
[`notebooks/_demo/exp_example.py`](../notebooks/_demo/exp_example.py) — copy it
like `exp.py` and `sbatch` it.

```python
import torch
from torch import nn
from shl2026 import embeddings, evaluate_predictions, track

tr  = embeddings("synthetic", "train")
val = embeddings("synthetic", "validation")

Xtr = torch.tensor(tr.X)
ytr = torch.tensor(tr.y - 1)                       # 1..8 labels -> 0..7 for CE loss
net = nn.Sequential(nn.Linear(tr.dim, 256), nn.ReLU(), nn.Linear(256, 8))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

with track("alice", run_name="torch-mlp", seed=0,
           params={"fm": "synthetic", "head": "torch-mlp", "lr": 1e-3}) as run:
    for epoch in range(20):
        opt.zero_grad()
        loss = loss_fn(net(Xtr), ytr)
        loss.backward()
        opt.step()
        run.log_metrics({"train_loss": loss.item()}, step=epoch)   # -> loss curve

    with torch.no_grad():
        y_pred = net(torch.tensor(val.X)).argmax(1).numpy() + 1     # back to 1..8
    result = evaluate_predictions(val.y, y_pred)                    # challenge macro-F1
    run.log_eval(result)

    torch.save(net.state_dict(), "model.pt")
    run.log_artifact("model.pt")                                    # ship weights with the run

print(result.summary())
```

`seed=0` in `track` already seeds Python/NumPy/PyTorch for you, so the loop is
reproducible without extra boilerplate. The pattern is identical for any
framework: train however you like, end with
`evaluate_predictions(val.y, your_labels_in_1..8)` → `run.log_eval(result)`.

## See also

- [`STUDENTS.md`](../STUDENTS.md) — setup, the daily loop, the honesty rules,
  what's yours to vary vs. baked into the cache, submission.
- [`METHODS.md`](../METHODS.md) — the task, the data, why the FMs are frozen.
- `shl2026 --help` — the CLI face (the lead's outer loop: `train-head`,
  `predict`, `format-submission`).
