# Cross-team prediction-combination contract (SHL-2026)

How AGH and the collaborator combine two **independent** pipelines into one submission with a
**guaranteed-safe** ensemble. Tooling: `modeling/combine_external.py` (alignment-verified, gated).
The #1 risk is silent row-misalignment on the hidden test — this contract eliminates it.

## 0. What's already compatible (verified)
Same classes (1..8: Still/Walk/Run/Bike/Car/Bus/Train/Subway), window geometry (100 Hz, 500
samples, 5 s, no overlap), official split + counts (784,288 / 115,156 / 92,726), **no shuffle**
(row index = temporal identity), and **majority-vote window label with first-max tie-break**
(our `_majority_label` == the collaborator's `1+argmax(counts)`). The preprocessing/representations
differ (their SMV-magnitude images vs our spectrograms/FM-embeddings) — that is **good** (ensemble
diversity), not a problem.

## 1. Prediction file format (both sides emit)
- **Hidden test:** a NumPy array `preds_test.npy` of shape **(92726, 8)**, dtype float, **calibrated
  per-class probabilities**, rows in the **official hidden-test order** (no shuffle), **class axis =
  labels 1..8 IN ORDER** → column 0 = Still, …, column 7 = Subway. Rows should sum to ~1 (the tool
  re-normalizes regardless).

## 2. Alignment proof (MANDATORY before combining)
Both sides compute a per-window **signature from the RAW signal** (raw Acc/Gyr/Mag x-axes etc.,
**BEFORE any unit conversion or standardization**), in the official order:
```bash
# AGH:
python notebooks/mdaniol/modeling/combine_external.py --emit-sig test   # -> sig_test.npy
# collaborator: emit the SAME signature from their RAW arrays (see window_signature: per channel
#   [first, middle, mean, std] over the 500 samples, channels Acc_x,Acc_y,Acc_z,Gyr_x,Mag_x), float64.
```
Then verify they match (float-dtype-robust `allclose`):
```bash
python notebooks/mdaniol/modeling/combine_external.py --verify sig_test_AGH.npy sig_test_collab.npy
# -> "ALIGNED ✓" (PASS) or "MISALIGNED ✗ first mismatch at row N" (STOP — fix order before combining)
```
The signature is invariant to float32/float64 and to each pipeline's preprocessing **iff computed on
RAW values** — so it proves row i is the same window on both sides.

## 3. Combine -> submission (refuses to run unless aligned)
```bash
python notebooks/mdaniol/modeling/combine_external.py \
    --combine preds_test_AGH.npy preds_test_collab.npy \
    --ours-sig sig_test_AGH.npy --theirs-sig sig_test_collab.npy --w 0.5
# verifies alignment, weighted soft-votes (w = weight on AGH), writes a validated 92726x500 submission.
```

## 4. Choosing the blend weight `w` (leakage-safe)
Tune `w` ONLY on a held-out slice **neither model trained on**. Key asymmetry: **AGH trains on
User-1 + validation[FIT]; the collaborator trains on official train only.** So the common clean
held-out = **AGH's `validation[TUNE]` / `[TEST]` slices** (held out by AGH; the collaborator never
trains on validation). Also reconcile the **purity filter** there (collaborator drops purity<0.90;
AGH keeps all) — pick one policy for the shared slice. Procedure: AGH shares the val_split row indices
+ labels + signatures for that slice; collaborator predicts those exact rows; sweep `w` to maximize
macro-F1 on it; lock `w`; apply on the hidden test. Do NOT tune on the hidden test.

## 5. Differences to keep in mind (don't block test-ensembling)
- **Purity filter:** collaborator ≥0.90 on train+val; AGH none → train/val *row subsets differ*.
  Irrelevant for the test (unfilterable); matters only for a shared eval (§4).
- **Train-on-validation:** AGH uses validation[FIT] in training; collaborator doesn't → no
  leakage-clean shared eval on validation[FIT].
- **Eval protocol:** AGH = temporal+embargo FIT/TUNE/TEST, BHT-only; collaborator = official
  train/validation → held-out numbers not directly comparable (agree one convention for the paper).
- **Units:** collaborator's /9.81, /2π, /100 shape only their representation — zero alignment impact
  (which is why the signature is on RAW values).

## TL;DR
Emit signatures → `--verify` (must be ALIGNED) → `--combine` (gated on alignment) → submission.
That makes the ensemble **provably** row-aligned; the rest (weights, shared eval) is optimization.
