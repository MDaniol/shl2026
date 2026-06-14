"""Demo batch script — a *custom training loop* with full MLflow logging.

The companion to exp.py. Same plumbing (embeddings -> train -> evaluate ->
track), but instead of a one-line scikit-learn `.fit` it trains a small PyTorch
MLP by hand, so you can see how `track()`'s `run` logger drops into your own
epoch loop:

  * `run.log_metrics({...}, step=epoch)` -> per-epoch curves in the MLflow UI.
  * `evaluate_predictions(y_true, y_pred)` -> the challenge macro-F1 straight
    from raw predictions. Use this (not `evaluate(head, set)`) whenever you
    don't have a scikit-learn `.fit/.predict` head.

The auto-provenance (git SHA, container, seed, code snapshot, ...) is identical
to exp.py -- it all rides on the same one-line `with track(...)`. Full SDK
reference: docs/API.md.

Copy it into your own folder, set STUDENT, then submit:

    cp notebooks/_demo/exp_example.py notebooks/<your-name>/exp_example.py
    # edit STUDENT below
    sbatch scripts/student_job.sbatch notebooks/<your-name>/exp_example.py
    cat slurm-<jobid>.out
"""

from shl2026 import embeddings, evaluate_predictions, track

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    raise SystemExit(
        "This example needs PyTorch (an optional extra). It's already in the team "
        "env on Athena; on a bare install run `uv pip install -e '.[torch]'`, or "
        "see STUDENTS.md -> 'Need an extra library?'."
    ) from None

STUDENT = "YOUR_NAME"   # <- your MLflow experiment name (keep it consistent!)
FM = "synthetic"        # <- FM id in the shared cache; see list_available()
EPOCHS = 30
BATCH = 256
LR = 1e-3

train = embeddings(FM, "train")
val = embeddings(FM, "validation")

# Labels are 1..8 in the cache; CrossEntropyLoss wants 0..7 -- shift, then
# shift back with `+ 1` before scoring (see the note in docs/API.md).
Xtr = torch.tensor(train.X)
ytr = torch.tensor(train.y - 1, dtype=torch.long)
Xval = torch.tensor(val.X)

net = nn.Sequential(nn.Linear(train.dim, 256), nn.ReLU(), nn.Linear(256, 8))
opt = torch.optim.Adam(net.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

with track(STUDENT, run_name=f"{FM}+torch-mlp|batch", seed=0,
           params={"fm": FM, "head": "torch-mlp", "epochs": EPOCHS,
                   "batch": BATCH, "lr": LR}) as run:
    # `seed=0` above already seeded torch, so this loop is reproducible.
    n = Xtr.shape[0]
    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * idx.numel()

        # Monitor the challenge metric on validation each epoch -> a curve.
        net.eval()
        with torch.no_grad():
            val_pred = net(Xval).argmax(1).numpy() + 1   # back to 1..8
        val_f1 = evaluate_predictions(val.y, val_pred).macro_f1
        run.log_metrics({"train_loss": epoch_loss / n, "val_macro_f1": val_f1}, step=epoch)
        print(f"epoch {epoch:2d}  train_loss={epoch_loss / n:.4f}  val_macro_f1={val_f1:.4f}")

    # Final score -> log_eval is what the shared leaderboard reads.
    result = evaluate_predictions(val.y, val_pred)
    run.log_eval(result)

print(result.summary())
