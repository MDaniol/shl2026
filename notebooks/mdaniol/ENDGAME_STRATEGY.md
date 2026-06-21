# SHL 2026 — End-game strategy (evidence-gated)

Synthesis of the har-fm-scientist deep research (2026-06-22). Governs the rest of
the challenge alongside `AI_GUIDELINES.md` / `GUARDRAILS.md`.

## Ranking reality (verified vs assumed)
- **Frozen-FM rule** (verbatim, 2026 site): FMs used frozen; only lightweight heads trainable. ✅
- **Novelty explicitly encouraged**, esp. *underutilized* FMs. ✅
- **A technical paper is mandatory** to be ranked. ✅
- **Ranking criterion is UNCERTAIN for 2026.** Only hard precedent (2025 Task 2):
  *"The ranking criterion was the F1 score."* → **Treat as DUAL-objective: novelty + macro-F1.
  Do NOT sacrifice F1.** Confirm with organizers (email).
- Deadline ambiguous ("60.06.2026" typo) → **plan to 30.06.2026 (~8 days)** until confirmed.

## The bet (headline contribution)
**Placement-conditioned routing of frozen FMs under *unknown* deployment placement** — our
location experts + learned router + soft-MoE + calibrated global fallback — paired with a
**rigorous frozen-FM-vs-handcrafted comparison**. It is (a) the most defensible as *new*
(no direct prior art found; verify against 2025 SHL adjunct + the two "Ensemble of FMs for
Locomotion" papers), (b) motivated directly by the 2026 test (Bag/Hips/Torso, no Hand,
location-unknown, shuffled), (c) **already built**, (d) generates the headline figure
(oracle G1 vs router-realized G2 vs global) that is compelling *either way the gates fall*.
If someone already did placement-MoE, pivot novelty to the **unknown-placement soft-routing
+ calibrated β fallback** mechanism specifically.

## Decision tree (decide end of Day 2, on TUNE)
```
G1 oracle per-location gain over global?
├ ≥ ~+1.5 macro-F1 → real. Check G2.
│   G2 router realizes ≥~50–60% of oracle via soft-MoE+β?
│   ├ yes → HEADLINE = placement-conditioned routing; double down.
│   └ no  → HEADLINE = "oracle gain unreclaimable under unknown placement" (honest limit
│            result + calibrated-fallback analysis); ship global model for F1.
└ < ~+1.5 → drop MoE as F1 lever; HEADLINE = frozen-FM-vs-handcrafted benchmark/negative
            result; router shown as a probe of why placement-conditioning fails.

FM verdict (orthogonal): FM beats A → FM experts+fusion; FM complements A → stack A⊕FM
(likely); no FM helps → strong honest negative result, A is the F1 engine.
```

## F1 levers, ranked by gain/effort (TUNE numbers override these priors)
HIGH: 1) **probability calibration** (temperature/Platt on TUNE) — cheap, prerequisite for
β/gating; 2) **A ⊕ best-FM fusion** (logreg stacker) — likely biggest legit gain; 3) **TTA**
on the raw-FM branch (K=4–8 rotations + mild jitter, mean-softmax; not on magnitude features);
4) **per-class / Run-Subway threshold** tuning (macro-F1 is worst-class-dominated).
MEDIUM (gate-conditional): 5) soft-MoE + β (also the novelty); 6) magnetometer rail corrector
(conservative gated). LOW/skip: heavy aug beyond TTA, PCA sweeps as an end, generalized
VehicleExpert (unless G1 strong), stacking many FMs.

## Worth-it vs not (more FMs / ROCKET)
- **UniMTS** — defer; wire ONLY if all Phase-1 FMs lose AND ≥2 spare days (best pick if so:
  orientation-agnostic bias matches the shift; counts as "underutilized").
- **MOMENT-base / NormWear / vision (IMU→image)** — not worth it (cost/crowded/marginal).
- **ROCKET/MultiROCKET** — not worth it AND an **eligibility risk** (arguably not a "foundation
  model"); at most a labelled non-FM reference baseline in the paper.

## Hard process guards (anti lock-test overfitting)
- **Freeze the experiment set now.** Pre-register the analysis plan before looking at results.
- **≤3 TEST (lock) reads total**; all selection on TUNE; report #TEST evaluations in the paper.
- Soft gating + calibrated global-fallback floor so routing can never do worse than global.

## Do-this-next (≈8-day plan)
0. **Today:** email organizers (ranking, deadline, test-release); freeze pre-registered plan; cap TEST reads ≤3.
1. **Day 1–2:** run the built gates — Branch A floor; FM verdict (Mantis/UTICA/MOMENT × V0/V1/V2 × linear/kNN/MLP); G1 oracle; G2 router. All on TUNE.
2. **End Day 2:** decide via the tree above.
3. **Day 3–5:** double down on the winner + HIGH-ROI levers (calibration → fusion stacker →
   TTA → Run/Subway thresholds → soft-MoE+β + rail if gates pass). One consolidated TEST read.
4. **Day 3–8:** write the paper in parallel. Headline fig = G1/G2 oracle-vs-router-vs-global;
   backbone = frozen-FM-vs-handcrafted benchmark; net both outcomes as planned contributions.
5. **DROP:** vision/IMU-image, new FMs (UniMTS conditional), ROCKET-as-core, generalized
   VehicleExpert (unless G1 strong), any extra TEST reads.
