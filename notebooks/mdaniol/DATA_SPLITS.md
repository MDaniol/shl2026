# SHL 2026 — Data Splitting Standard (train / eval / test)

**Status:** authoritative. This is how we construct and evaluate splits; keep it in
sync with `modeling/split.py` and `modeling/train_split.py`. Grounded in the SHL
2026 challenge spec and the Bock et al. *Deep Learning for HAR* tutorial
(`TUTORIAL/dl-for-har`, `HAR_TUTORIAL.pdf`).

---

## 0. The dataset as given by the challenge (Level 1 — by participant)

| Split | Users | Locations | Frames/file | Labels? | Notes |
|---|---|---|---|---|---|
| **train** | User 1 | Bag, Hips, Torso, Hand | 196 072 | yes | consecutive in time |
| **validation** | Users 2 & 3 (mixed) | Bag, Hips, Torso, Hand | 28 789/file | yes | consecutive in time |
| **test** (hidden) | Users 2 & 3 (mixed) | Bag, Hips, Torso (**no Hand**) | 92 726 | **no** | **frames shuffled** |

- One frame = 5 s @ 100 Hz = **500 samples**, 9 channels (Acc/Gyr/Mag × x/y/z).
- Ranking metric = **macro-F1** over 8 transport modes (user-independent).
- **The official split is by participant** (User 1 vs Users 2&3). The test users are
  the **same** as the validation users (2 & 3) — so the task is *not* "generalize to a
  new person"; it is "generalize to unseen **time / the missing Hand location** of
  users already seen."

### Two distinct "shuffles" — do not conflate
| | Operation | In challenge? | Consequence |
|---|---|---|---|
| **split-shuffle** | randomly *assigning windows* to folds | **No — and we must not** | random assignment leaks temporally-adjacent windows → inflated estimate |
| **order-shuffle** | reordering the *fixed* test frames | **Yes** (real-time framing) | **none** for a per-window model: predictions and macro-F1 are order-invariant |

The order-shuffle reorders whole frames; each frame's internal 5 s signal is intact.
Its purpose is to forbid temporal-context cheating (HMM/Viterbi smoothing across
consecutive windows). We comply by using **window-internal features only** (no
cross-window features) and classifying each window independently.

---

## 1. Our internal split (Level 2 — `modeling/split.py`)

We split the **validation** set (Users 2 & 3) into three slices for honest internal
evaluation, because validation shares the test users:

```
fit  (~60%) -> trained on, together with ALL of User-1 train
tune (~20%) -> early-stopping + per-class threshold calibration
test (~20%) -> held out, never touched in fit/tune  (unbiased macro-F1)
```

Output: `modeling/artifacts/val_split.npy` — `int8` array (0=fit, 1=tune, 2=test),
row-aligned to the `LOCATIONS = (Bag, Hips, Torso, Hand)`-order concatenation of the
validation feature files (so the same indices map across features, embeddings, labels).

### How assignment is done (and why)
- **Per-class, contiguous time-blocks, NO shuffling.** For each class we take its
  windows in time order, cut ~10 contiguous blocks, and cycle them 60/20/20.
  - *Contiguous, not random* → avoids the tutorial's leakage warning (shuffling
    destroys time-dependencies; adjacent windows of one activity bout must not be
    split across slices).
  - *Per-class (not global)* → guarantees every class — especially the rare,
    temporally-clustered **Run (~4.3%)** — appears in every slice. This achieves
    stratification **without** shuffling (a hybrid of the tutorial's two competing
    concerns).
- **Held-out TEST restricted to test locations (Bag/Hips/Torso).** Hand windows that
  would land in TEST are reassigned to FIT (not wasted). This mirrors the real test
  (no Hand) and keeps the held-out estimate faithful.
- **Mixed users in every slice.** We do **not** hold out a whole user (no LOSO here):
  the real test mixes Users 2 & 3, so the held-out TEST must mix them too.

Effective ratio is ~65/20/15 (not exactly 60/20/20) because Hand cannot enter TEST,
so its share flows into FIT.

---

## 2. Two-phase training protocol (`modeling/train_split.py`)

**Phase A — honest estimate (uses `val_split.npy`):**
- `fit  = ALL User-1 train + validation[FIT]`  (model sees Users 2&3)
- `tune = validation[TUNE]`                      (early-stop + per-class calibration)
- `test = validation[TEST]`  (held out, Bag/Hips/Torso) → **unbiased macro-F1**
- Also trains a **User-1-only** model (train→Users 2&3) = the **cross-participant**
  reference (the only LOSO-style estimate the data allows — there is one training user).

**Phase B — submission (nothing wasted):**
- Retrain on `User-1 + ALL validation`, apply the Phase-A calibration, predict the
  hidden test, write `AGH_predictions_v3.txt` as **92 726 × 500** (per-frame label
  repeated across the 500 columns, in the test file's given order).

The FM lane (`modeling/probe_fusion.py --use-split`) uses the **identical** protocol,
so handcrafted and FM numbers are directly comparable.

---

## 3. Best-practice checklist (vs `dl-for-har` tutorial)

| Practice | Status |
|---|---|
| No shuffle when splitting (preserve time-dependencies) | ✅ contiguous blocks |
| Window before/independent of split (no window straddles a boundary) | ✅ non-overlapping 5 s frames |
| Test slice never used for tuning | ✅ TUNE ≠ TEST |
| Stratify rare classes across slices | ✅ per-class blocking |
| Report **per-class** + macro (not just averaged) | ✅ always (see metrics.class_report) |
| Report train–val **generalization gap** | ☐ planned |
| Cross-participant / LOSO (gold standard) | △ deliberate deviation — test is same-user; captured by the User-1-only reference + the official Level-1 subject split |
| Subject-id available for per-user (User2 vs User3) diagnostics | ☐ not yet (features carry `label` only) |

---

## 4. Reproducibility — what produces what

| Artifact | Producer | Path |
|---|---|---|
| 520-feature parquets | `feature_extraction/extract_features.py` | `dataset_parquet_features/<split>/<loc>.parquet` |
| `val_split.npy` (+ `.json` summary) | `modeling/split.py` | `modeling/artifacts/` |
| held-out report (per-class + macro) | `modeling/train_split.py` | `modeling/artifacts/split_results.json` |
| submission matrix | `modeling/train_split.py` (Phase B) | `modeling/AGH_predictions_v3.txt` |
| robustness sweep | `modeling/robustness_sweep.py` | `modeling/artifacts/robustness.json` |
| FM split-protocol bake-off | `modeling/probe_fusion.py --use-split` | `BAKEOFF_SPLIT.md` |

**Determinism:** `split.py` is deterministic (no RNG — pure positional blocking), so
`val_split.npy` is reproducible from the features alone. Augmentation RNG is seeded
(`--seed`). Record the git commit + `val_split.json` counts with every result.

---

## 5. Invariants to never break
1. Never **shuffle** when assigning windows to fit/tune/test (use contiguous blocks).
2. Never let the held-out **TEST** influence training or calibration.
3. Never use **cross-window** features (test frames are order-shuffled).
4. Held-out **TEST** = Bag/Hips/Torso only, mixed Users 2 & 3 (mirror the real test).
5. Final submission = **92 726 × 500**, per-frame label repeated, original file order.
