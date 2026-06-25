# Tier-1 macro-F1 post-hoc (moment-small_V1, temporal split, Bag/Hips/Torso; select=TUNE, lock=TEST). v1=0.8029; TUNE-selected variant **cal(v1)** -> KEEP v1 (oracle-prior ceiling 0.8003 is BELOW v1 -> prior-shift HURTS macro-F1).

| variant | TUNE macro | TEST macro | Run F1 TUNE | Run F1 TEST |
|---|---|---|---|---|
| raw | 0.8018 | 0.7981 | 0.823 | 0.918 |
| **cal(v1)** | 0.8266 | 0.8029 | 0.919 | 0.957 |
| logit_adj(t=1.5) | 0.8153 | 0.8046 | 0.912 | 0.959 |
| mlls | 0.7772 | 0.7941 | 0.677 | 0.885 |
| cal+mlls | 0.8122 | 0.7978 | 0.857 | 0.948 |
| _oracle-prior (n/d)_ | — | 0.8003 | — | 0.890 |

## Interpretation (honest)
- **KEEP v1.** On TUNE (the selection criterion), `cal(v1)`=0.8266 is the **best of all variants** —
  no post-hoc beats plain calibration on the honest selection set. (logit_adj's +0.0017 on TEST is
  noise: it *loses* to cal(v1) on TUNE by 0.011, so adopting it would be selecting on the lock set.)
- **Oracle-prior ceiling 0.8003 < v1 0.8029** → even with the TRUE test prior, label-shift adaptation
  **hurts** macro-F1. The metric weights classes uniformly; adapting to the (imbalanced) test prior
  optimizes accuracy, not macro-F1. So MLLS / prior-shift is the WRONG direction here — a clean
  negative, and it also rules out a Bayesian prior-estimation variant (same sub-v1 ceiling).
- **Run is already solved** (F1≈0.96 after v1 calibration — the 0.20 was uncalibrated). The macro-F1
  bottleneck is the **rail pair (Train/Subway ≈0.61)**, which is hard/irreducible (see H2), not class
  imbalance — so threshold/prior levers can't move it.
- **Lesson:** per-class calibration (already in v1) is the post-hoc lever; it is tapped out. Effort
  moves to the FM dimension (bake-off, new FMs) + orientation robustness.
