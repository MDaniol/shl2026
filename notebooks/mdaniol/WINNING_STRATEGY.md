# SHL 2026 — winning strategy (deep-research synthesis, 2026-06-23)

Synthesis of two deep-research passes (SHL/HASCA challenge history + arxiv 2024-26 frozen-FM
literature), reconciled against our own banked results. Confidence: [V]=verified primary source,
[I]=inference. Full agent reports in session log; agent memory under har-fm-scientist/.

## The reframing that sets expectations
- **Temporal smoothing (HMM/Viterbi/majority) is DEAD for us** — worth ~+10 pp historically [V],
  but our test is shuffled/mixed-location/single-file → no smoothing, no order reconstruction.
  So historical 90%+ leaderboards are NOT our target; shuffled-test editions landed **70-80%
  (2024) / 88.5% (2020, raw-axis DL, no frozen rule)**. **Our honest ~0.80 is competitive.**
- **2025 = our direct analogue** (first frozen-FM edition): won by an **ensemble of 3 FROZEN FMs
  (BIOT+CBraMod+MOMENT) → shallow MLP late fusion** [V]. Cross-FM frozen ensembling is the
  proven winning family. (Winner reported *weighted*-F1 ~92% — flatters vs macro; not comparable.)
- **Train↔Subway is the dominant residual error** (10% even for winners, 20-35% mid-pack) [V];
  Car↔Bus secondary and partly irreducible without GPS.

## Ranked next steps (merged; both agents' priorities reconciled)

### Tier 1 — CPU-only, post-hoc, highest-confidence, do first
1. **Calibration → unknown-prior estimation → per-class F1-optimal thresholds.** Our single
   biggest known lever (~+0.16) [V-ours], and *under-exploited in SHL lit* because everyone
   leaned on smoothing → strong paper point. Upgrade the ad-hoc per-class calibration to:
   Dirichlet/vector scaling → estimate the test prior from the **unlabeled shuffled test scores**
   (MLLS/Saerens-EM, valid on a shuffled marginal) → per-class thresholds (optimal ≠ 0.5) +
   logit adjustment. **Run is weak from scarcity (~4%), not confusability → prior correction
   hits it at the root.** [arXiv:1901.06852, 1402.1892, 2007.07314] Expected +1 to +3 macro.
2. **Cheap head comparators** (StandardScaler+LogReg, RBF-SVM) alongside LightGBM — the MantisV2
   paper shows a linear head sometimes wins on frozen embeddings. Near-free.

### Tier 2 — one GPU re-extraction, high-confidence (both agents converge)
3. **MOMENT layer selection + richer token pooling.** (a) Use ~**layer 10**, not the final layer
   (MOMENT plateaus/over-specializes after ~10) [V]; (b) **concat pooling** (mean⊕max⊕std, and
   **per-channel** concat) instead of mean — the one IMU-specific frozen result, COMODO, shows
   concat > mean by +1.8-6.8 [V, arXiv:2503.07259]. Modify extract_embeddings (output_hidden_states
   + multi-stat/per-channel pooling), re-extract MOMENT-small_V1. Expected +1 to +4 macro.
4. **Gravity-frame canonicalization before the FM** (rotate so z∥gravity per window) — the 2020
   winner's key move [V] + EqNIO [arXiv:2408.06321]; collapses orientation SO(3)→yaw, making our
   rotation-TTA (already built) cheaper + stronger. Re-extract on canonicalized signal.

### Tier 3 — GPU, the "how 2025 was won" + the novelty hedge
5. **Cross-FM frozen ensemble** — add a complementary frozen FM to MOMENT, concat **top-2 only**
   (kitchen-sink loses), normalize per-FM + PCA before the tree [arXiv:2512.01405, 2411.01645].
   Candidates: **UniMTS** (orientation/location-invariant motion FM — best structural match to our
   shift; caveats: 6-ch no-mag, 20 Hz, graph encoder, frozen-probe untested) [arXiv:2410.19818];
   **NormWear** (the only FM that natively ingests our **magnetometer** + a spectral CWT view —
   orthogonal to MOMENT) [arXiv:2412.09758]. Doubles as the 2026 "new/underutilized FM" novelty.
6. **Temporal FM bake-off** (Mantis8M/UTICA/MOMENT under the honest split) — settles the FM
   comparison for the paper; gates which 2nd FM to ensemble. (probe already wired, temporal.)

### Already in flight / lower priority
- **Rotation-TTA** (built, runnable) + add **multi-resample-length** self-ensembling (128/256/
  512/1024 + first-difference), small gain, shuffle-safe; check it doesn't hurt Run/vehicle F1.

## Do NOT re-chase (banked negatives — confirmed on honest eval)
- **Any temporal smoothing / order reconstruction** — invalid on the shuffled test.
- **Location Mixture-of-Experts** (≈0 gain), **vehicle/engine-vibration** (modest-negative),
  **MantisV2/UTICA alone** (< handcrafted; only useful inside an ensemble).
- **Magnetometer rail expert (Train/Subway).** NOTE: the challenge-history agent recommended
  "engineer the magnetometer aggressively for rail" — but **we already did this** (magnitude AND
  raw-axis gravity-referenced, leakage-safe leave-one-bout-out → ~chance, H2 negative). The rail
  residual is likely irreducible here without a **barometer** (which SHL lacks). This is a
  *result*, not an open lever.

## The HASCA paper (a real asset, not an afterthought)
- A paper is **mandatory to be ranked**; challenge papers are reviewed for **methodology, not
  novelty** [V]; the most-cited papers in this ecosystem are **characterization-of-failure**
  ("Top in the Lab, Flop in the Field?") [V]; **leakage from improper splits is the named,
  organizer-acknowledged pitfall** (96%→84%→61% across users →54% across positions) [V].
- Our headline: **the leakage audit** (blocked 0.87 → honest temporal+embargo 0.80, +0.07
  inflation) is a textbook instance of that pitfall — report it as a *result*, not a footnote.
- Plus the **leakage-audited negatives** (MoE, rail, vehicle, MantisV2/UTICA) + **per-class
  calibration as the surviving lever**. This is exactly the valued contribution type.

## Recommended execution order (7-day window)
Tier 1 (CPU, today) → Tier 2 re-extraction (GPU) → Tier 3 cross-FM (UniMTS/NormWear) →
write the paper around the audit + negatives + calibration. Validate every step on the
leave-subjects/temporal-embargo split; never peek at the shuffled test. Track all (rule §8).
