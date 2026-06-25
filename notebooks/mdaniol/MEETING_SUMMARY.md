# SHL 2026 — progress summary (for the team)

Plain-language overview of what we're doing, what we've found, and what's next.

## 1. The challenge — in one paragraph
We get motion-sensor data from a smartphone (accelerometer, gyroscope, magnetometer) and must
guess **how the person is travelling**: one of 8 modes — **Still, Walk, Run, Bike, Car, Bus, Train,
Subway**. The data comes in **5-second windows**. We're scored by **macro-F1** — the average
accuracy across the 8 classes, where every class counts equally (so a rare class like *Run*
matters just as much as a common one). It's the official **SHL challenge** at the HASCA/UbiComp
workshop; there's a leaderboard **and** a paper.

## 2. The rules that shape everything
- **Frozen AI models only.** We may use big pretrained "foundation models" (FMs) as-is, but we are
  **not allowed to retrain them** — only to train small, lightweight classifiers on top.
- **The test data is shuffled.** The 5-second windows arrive in random order with no labels, so we
  **cannot use time** (e.g. "the previous window was a train, so this one probably is too"). Every
  window must be judged on its own.
- **Unknown phone position.** In the test, the phone can be in a bag, hip pocket, or hand — we don't
  know which. So our method must be robust to phone orientation/position.

## 3. Our approach — two "lanes" combined
For each 5-second window we build two kinds of description and combine them:
1. **Hand-crafted features** (520 numbers) — things we engineered ourselves: how much it vibrates,
   at which frequencies, rotation patterns, etc.
2. **Foundation-model "embedding"** — we run a frozen AI model once over the window and it outputs a
   summary vector. (We pre-compute and save these — that's "embedding extraction".)

We feed **both** into a small classifier (a gradient-boosted tree) that we *do* train, then we
**calibrate** it (adjust the decision thresholds) to maximise macro-F1.

## 4. Where we stand
- **Best model so far ≈ 0.80 macro-F1** (honest estimate). Recipe: hand-crafted features + the
  **MOMENT** foundation model + calibration. This is our submitted **version 1**.
- **Is 0.80 good?** Yes. Past winners hit 90%+ **only** when they could use the time-order trick —
  which is banned for us this year. On *shuffled* tests like ours, the realistic ceiling is the
  **high-70s to 80s**, so we're competitive.

## 5. The most important lesson: don't fool ourselves
Last year the team measured **0.90** internally but scored **0.71** for real — because the
evaluation leaked information ("over-optimistic"). We fixed this:
- We built an **honest evaluation** that splits the data by time so we can't accidentally cheat.
- We found the *naive* way of splitting gives **0.87**, but the honest way gives **0.80** — a 7-point
  gap that would have been an embarrassing over-claim. **Catching this is itself a result** worth
  writing up.

## 6. What works, and what doesn't (negatives are useful!)
**Helps:**
- **Calibration** — biggest single improvement; rescues the weak *Run* class.
- **The MOMENT foundation model** — the only FM (so far) that adds value on top of hand-crafted
  features.

**Tried and *didn't* help (so we stop and document them — these become paper content):**
- Splitting into per-position "expert" models → no gain.
- Using the magnetometer to tell **Train vs Subway** apart → no reliable gain (the magnetic signal
  turned out to depend on the *route*, not the *mode*).
- Detecting **engine vibration** to separate Car/Bus → only a weak effect.
- Two other foundation models (MantisV2, UTICA) → worse than our hand-crafted features.

## 7. The plan from here
1. **Squeeze more from the scoring step** (cheap, smart): better-calibrated decision thresholds that
   specifically help the weakest class (*Run*). **Running now.**
2. **Fairly test all the foundation models** on our honest split (MantisV2, Mantis8M, UTICA, MOMENT)
   — so we know definitively which is best.
3. **Try newer, motion-specific foundation models** (UniMTS, NormWear) — the most likely source of a
   real improvement, and a novelty point for the paper.
4. **Make the model robust to phone orientation** (re-orient signals to a common reference + average
   over rotations).
5. **Write the paper** — its strength is our **honest evaluation** + the documented list of what
   works and what doesn't. The workshop specifically values this kind of rigorous, careful study.

## 8. How the work is organised
- All experiments are **version-controlled, logged to a tracking server (MLflow), and reproducible**.
- Heavy AI computations run on the **Athena** GPU cluster; the lighter training/scoring runs on the
  **Ares** CPU cluster — so we use resources efficiently.
- Live status of every experiment is in **`EXPERIMENT_LOG.md`** (a job tracker + results table).

**Bottom line:** we have a solid, *honestly-measured* ~0.80 system and a clear, prioritised plan to
improve it — with the discipline to not over-claim, which is exactly what cost the team last year.
