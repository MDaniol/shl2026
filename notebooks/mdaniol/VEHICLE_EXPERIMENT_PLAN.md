# Vehicle-disambiguation experiment plan (pre-registration)

**Status:** pre-registered 2026-06-22, before looking at any new lock-test numbers.
**Authority:** follows `AI_GUIDELINES.md` (8-stage, leakage gate, MLflow+artifacts §8),
`GUARDRAILS.md`, `DATA_SPLITS.md`. Selection on TUNE, lock-test read once, per-class +
confusion always reported. Honest split = temporal+embargo (`val_split_temporal.npy`).

## Motivation — the vehicle errors split into two physically distinct clusters

From our committed confusion matrix (temporal split, `artifacts/split_results.json`):

| true → pred | count | note |
|---|---|---|
| **Subway → Train** | **984 (37.8% of Subway)** | the single largest vehicle error |
| Car → Train | 348 (14.2%) | |
| Car → Bus | 263 (10.7%) | |
| Bus → Train | 159 | Train is a low-precision (0.52) "sink" |

Per-class F1 (temporal): Car 0.745, Bus 0.656, Train 0.641, Subway 0.613.

Two clusters, **two different sensors** (deep-research, 3 agents, 2026-06-22):

| cluster | right sensor | physics |
|---|---|---|
| **Rail: Subway↔Train** (biggest error) | **raw-axis magnetometer** | electric traction distorts field *direction/DC*; rail vibration (~63 Hz) is **above our 50 Hz Nyquist** so accelerometer can't reach it |
| **Road: Car/Bus, motorized-vs-Still** | **accelerometer engine band** | combustion **idle** vibration ≈ 20–35 Hz (firing freq = RPM/60 × cyl/2), sub-Nyquist and real |

**We blinded ourselves to both:** (a) spectral features stop at **20 Hz** (`CENTERS_HZ` max 15,
widest band [10,20]) → engine band unseen; (b) the orientation-invariant **magnitude transform**
discards magnetic direction → rail cue destroyed (consistent with our rail-expert "DISABLE").

## Hypotheses (falsifiable)

- **H1 (engine band):** combustion vehicles (Car/Bus) show a class-distinct accelerometer
  spectral peak in **18–40 Hz**, absent in Still and electric rail, that our current ≤20 Hz
  features miss. Adding engine-band features reduces Car/Bus + motorized-vs-Still confusion.
- **H2 (raw-axis mag rail):** Train vs Subway are separable via **raw-axis** magnetometer
  spectral/DC features (lost under the magnitude transform). A raw-axis rail sub-decision
  reduces the Subway↔Train error.

**Priors to stay honest about:** literature says ">10 Hz adds little overall" and "Car/Bus is
near-impossible without GPS" → expect **small, targeted** gains, strongest on motorized-vs-Still
and the rail pair, not a headline jump. Cruise engine (>1500 RPM → 50–100 Hz) and rail
high-freq are **aliased/lost** at 100 Hz — scope is the **idle/stop-go** regime only.

## Pre-registered success criteria (decide BEFORE seeing lock-test)

- **Diagnostic gate (Phase 1):** H1 "supported-to-proceed" iff per-class Welch PSD shows a
  distinct 18–40 Hz bump for Car and/or Bus, **absent in Still/rail**, **consistent across
  Bag/Hips/Torso**, and the candidate band-features have **higher MI with class than with
  placement**. H2 similarly: raw-axis mag PSD/DC separates Train vs Subway across placements.
  Fail the gate → stop that lever (record the negative; it's a finding).
- **Ablation (Phase 2):** a feature lever is **KEPT** only if, on the **locked temporal split**,
  read once: vehicle-subset macro-F1 delta CI lower bound **> 0** (N≥5 seeds, paired bootstrap)
  **AND** the targeted off-diagonal (Car↔Bus for H1, Subway↔Train for H2) decreases. A gain
  that exists only on the blocked split = the over-claim trap → discard.
- **Leakage tripwires:** band-feature MI with session/route must be low (no session column
  exists yet — use placement as the available confound + a leave-one-recording check if a
  session key can be recovered). Train-only normalization; no window overlap across split.

## Phased plan

- **Phase 0 — DONE.** Code facts confirmed: 20 Hz ceiling; VehicleExpert uses mag *magnitude*;
  raw 9-axis data intact for re-extraction.
- **Phase 1 — diagnostics (cheap, leakage-free, no training):** `vibration_psd_diagnostic.py`
  — per-class Welch PSD on raw acc magnitude, body-acc, and **raw mag axes**, placement-
  stratified; band-power table (18–40 Hz engine band vs 1–15 Hz); MI of candidate features vs
  class and vs placement. Output → JSON/CSV/MD + MLflow. **Gates Phase 2.**
- **Phase 2 — features + ablation (only if a gate passes):**
  - H1: engine-band acc features (centers 20–45 Hz, narrow bands; **ratio + peak-prominence +
    spectral flatness/entropy** — RPM-*invariant*, so we learn "engine-ness", not "this car").
  - H2: raw-axis magnetometer spectral/DC features for the rail sub-decision.
  Re-extract → re-fit fusion → calibrate on TUNE → lock-eval once.
- **Phase 3 — VehicleExpert** stays a **gated, mass-preserving post-processor**, now with the
  right features per sub-decision (rail→raw-mag, road→engine-band). Keep only on a CI-positive
  vehicle-confusion improvement on the honest split.

## Paper framing (insight, not just leaderboard)
Characterize *why* the hard pairs resist IMU-only sensing: **Car/Bus = physics ceiling without
GPS**; **Train/Subway = design-induced** (magnitude transform discards magnetic anisotropy +
no barometer); **engine band recoverable but idle-only at 100 Hz**. Negative + characterization
results, leakage-audited — an IMWUT/HASCA-style contribution.

## Sources
Engine firing-frequency + real 100 Hz idle (26 Hz car, 22 Hz minibus, 35 Hz bus); Sentiance
(Car/Bus near-impossible w/o GPS); SHL 3-year review; rail vibration ~63 Hz; magnetometer-TMD
patent US10,993,080 / PMC7795463. (Full URLs in the deep-research agent reports, 2026-06-22.)
