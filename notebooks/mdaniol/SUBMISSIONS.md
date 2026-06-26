# Submissions log

This file is the canonical, append-only history of challenge submissions. Each data row
below is written automatically by `modeling/submit_fusion.py` (one row per build); this
header documents what the version label means and when to bump it.

## Versioning policy

A submission **version** is a *provenance label only* — it does **not** change the model,
features, recipe, or score. It controls exactly two things:

1. the output filename → `AGH_predictions_<VERSION>_<EMB>-fusion.txt` (e.g. `..._v2_utica_V2-fusion.txt`)
2. the row appended to the table below.

A submission is **fully identified** by four fields, not the version alone:
`version` (human label) · `recipe`/`EMB` (what was built) · `git SHA` (the exact code) ·
the split used for FIT/calibration. The version is just the convenient handle; the git SHA
is the ground truth. (The matching MLflow run, when the step is tracked, carries the same
recipe + git SHA — see `AI_GUIDELINES.md` §8 traceability.)

### How to run
```bash
# BEST submission — calibrated soft-vote of the top-2 FMs (E-VOTE-01, 0.8342 lock):
VERSION=v3 EMBS=utica_V2,mantisv2_V1 sbatch notebooks/mdaniol/hpc/submit_vote_helios.sbatch
#   FAST PATH (no refit): if voting_head.py already ran it saved vote_models_*.joblib —
#   reuse it (minutes, identical predictions, no hours-long fit):
#   python notebooks/mdaniol/modeling/submit_vote.py --version v3 --embs utica_V2,mantisv2_V1 \
#       --from-models notebooks/mdaniol/vote_models_utica_V2+mantisv2_V1.joblib
# single-FM base (utica_V2, 0.8213 lock):
VERSION=v2 EMB=utica_V2 sbatch notebooks/mdaniol/hpc/submit_helios.sbatch
# (legacy Athena path: VERSION=v1 EMB=moment-small_V1 sbatch notebooks/mdaniol/hpc/submit.sbatch)
```
`submit_helios.sbatch` defaults to `VERSION=v2`, `EMB=utica_V2`, `HELDOUT="0.8157 (temporal lock…)"`.
Pass `HELDOUT=…` to record the right bake-off lock estimate when you change `EMB`. Every build now
also logs an MLflow run (`submit_<version>_<emb>`) + snapshots the .txt and SUBMISSIONS.md (rule §8).

### When to bump the version
- **Bump** (`v1 → v2 → …`) for every *distinct submission you want to keep side by side* —
  i.e. whenever the **recipe changes**: a new base FM/embedding, adding VehicleExpert,
  adding Run-recall thresholds, a different calibration, etc. Bumping preserves the prior
  `.txt` and keeps an honest, comparable history.
- **Do NOT bump** when only re-running the *same* recipe (e.g. re-running after an infra
  fix that doesn't change outputs). Reuse the same label — but note this **overwrites** that
  `.txt` and appends another row, so the table will show two rows with the same version and
  different git SHAs. That is the intended audit trail of a re-run.

### Planned version line (update as we go)
| version | intended recipe |
|---|---|
| v1 | MOMENT-small_V1 ⊕ 520 handcrafted (LGBM fusion) + per-class calibration — first ≥0.80 build |
| v2 | v1 + VehicleExpert {Car,Bus,Train,Subway} corrector — *only if it KEEPs on lock-test* |
| v3 | + Run-recall per-class thresholds |

### Naming note
The new scheme is `AGH_predictions_<version>_moment-fusion.txt`. The older
`AGH_predictions.txt` / `_v2` / `_v3` in `modeling/` are a **different, pre-existing series**
(no `_moment-fusion` suffix) from earlier work — not part of this recipe line. Don't conflate
the two.

## Log
<!-- submit_fusion.py appends one row per build below; keep this table as the last block. -->

| version | date (UTC) | recipe | git SHA | file | held-out (conservative) | notes |
|---|---|---|---|---|---|---|
| v1 | 2026-06-22 13:48 | moment-fusion(moment-small_V1)+cal | cda79f9 | AGH_predictions_v1_moment-fusion.txt | ~0.803 (temporal; bracket 0.725-0.803) | FIT-train + TUNE-calibrated; det |
