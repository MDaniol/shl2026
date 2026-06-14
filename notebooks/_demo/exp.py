"""Demo batch script — the inner loop as a plain script (Path B of DEMO.md).

Same SDK code as the notebook template, with prints so the output lands in the
Slurm job log. Copy it into your own folder and set STUDENT, then submit:

    cp notebooks/_demo/exp.py notebooks/<your-name>/exp.py
    # edit STUDENT below
    sbatch scripts/student_job.sbatch notebooks/<your-name>/exp.py
    cat slurm-<jobid>.out

The env (shared cache + MLflow URI) comes from student_job.sbatch sourcing the
team env.sh — no config block needed on the batch path.
"""

from shl2026 import embeddings, make_head, evaluate, track

STUDENT = "YOUR_NAME"   # <- your MLflow experiment name (keep it consistent!)
FM = "synthetic"        # <- FM id in the shared cache; see list_available()

train = embeddings(FM, "train")
val = embeddings(FM, "validation")

with track(STUDENT, run_name=f"{FM}+logreg|batch", seed=0,
           params={"fm": FM, "head": "logreg"}) as run:
    result = evaluate(make_head("logreg", seed=0).fit(train.X, train.y), val)
    run.log_eval(result)

print(result.summary())
