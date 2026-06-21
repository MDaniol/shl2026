# SHL 2026 — Pre-registered analysis plan (frozen before looking at TEST)

Written **before** reading any lock-test (TEST) result, to prevent lock-test
overfitting / multiple-comparisons leakage (the #1 risk in `ENDGAME_STRATEGY.md`).
Recorded with the git SHA at commit time. Deviations must be logged below with a reason.

## 1. Frozen experiment set (no new branches after this)
Exactly these are evaluated; nothing else gets a TEST read:
- **Branch A** — 520 handcrafted features → LightGBM (the baseline / F1 floor).
- **FM verdict** — {MantisV2, UTICA, MOMENT-small} × {V0, V1, V2} × {L2/PCA → logreg, ridge}
  (+ one nonlinear head). Question: does any FM beat/complement A?
- **Location MoE** — Bag/Hips/Torso experts + router; configs {global, oracle, uniform,
  hard-router, soft-router, global+soft β∈{.25,.40,.50,.70}+adaptive}.
- **freq+mag** linear branch; **rail** expert (Train/Subway gated); **branch fusion**
  (weighted-avg + logreg stacker).
- Conditional (only if its gate passes): **VehicleExpert** (only if G1 strong), **TTA**
  (only if a raw-FM branch is in the winner), **UniMTS** (only if all FMs lose + ≥2 days).

## 2. Splits & selection rule
- FIT/TUNE/TEST = blocked 60/20/20 of validation (`val_split.npy`), eval on Bag/Hips/Torso.
- **All model/config/threshold/β selection is on TUNE.** TEST is the lock.

## 3. TEST (lock) read budget: ≤ 3 total
1. Read #1 — the chosen winner from the decision tree (one consolidated read of all
   gate-relevant configs at once).
2. Read #2 — the winner + chosen high-ROI levers (calibration/thresholds/rail/MoE) combined.
3. Read #3 — reserve (e.g., final submission config sanity).
Every TEST read is logged in §6 with date + git SHA. The paper reports this count.

## 4. Decision tree (pre-committed; decided on TUNE)
- G1 oracle gain ≥ +1.5 macro-F1 → location real → G2: router realizes ≥50–60% via soft-MoE+β?
  yes → routing headline; no → "unreclaimable under unknown placement" limit result.
- G1 < +1.5 → drop MoE; headline = frozen-FM-vs-handcrafted benchmark.
- FM: beats A → FM experts+fusion; complements A → stack A⊕FM; none → A is the engine (negative result).

## 5. Primary & secondary endpoints
- **Primary:** held-out TEST **macro-F1** of the chosen config vs Branch A.
- **Secondary (always reported):** per-class F1 (esp. Run, Train, Subway, Car, Bus),
  vehicle-subset & rail macro-F1, Train/Subway & Car/Bus confusion, selection-lock gap.
- **Ship a component only if** (on TEST): macro-F1 +≥0.003, OR vehicle-subset +≥0.010 with
  no global loss; no non-target class F1 drops >0.010; positive net-gain; small sel-lock gap.

## 6. TEST-read log (append-only)
| # | date | git SHA | what was read | result |
|---|---|---|---|---|
| (none yet) | | | | |

## 7. Deviations log (append-only)
| date | deviation | reason |
|---|---|---|
| (none yet) | | |
