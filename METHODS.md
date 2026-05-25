# Methods (scientific contract)

This document is the human-readable description of *what* this pipeline does
and *why*. It is paired with `REPRODUCIBILITY.md`, which formalises *how* we
guarantee that the same code + data give the same answer.

## Task

Recognise 8 modes of locomotion / transportation from inertial smartphone
sensors, **user-independent**: training data is from user 1 only; validation
and test data mix users 2 and 3.

## Data

- 5 s windows at 100 Hz → frames of shape (samples=500, channels=9): Acc, Gyr,
  Magn on x/y/z.
- Train: 59 days, 4 phone locations (Bag, Hips, Torso, Hand); 196 072 frames
  per location-channel file. Frames are consecutive in time.
- Validation: 6 days, users 2&3 mixed, same 4 locations; 28 789 frames.
- Test: 28 days, users 2&3 mixed, **only Bag/Hips/Torso (no Hand)**; 92 726
  frames. Frames are **shuffled** to test real-time-style robustness.

Raw files live on Helios under
`$PLG_GROUPS_STORAGE/<GRANT>/shl2026/data/raw/` and are immutable. Their
SHA-256 hashes are recorded in `MANIFEST.sha256` and re-verified at the start
of every pipeline run.

## Modelling rule (binding)

> *"Foundation models must be used in a frozen manner, i.e., without any
> fine-tuning or retraining. However, participants may train lightweight,
> task-specific components (e.g., classification heads) on top of frozen
> foundation models, without updating the model parameters."*
> — SHL 2026 challenge website, retrieved 2026-05-25.

We therefore split modelling into:

1. **Foundation-model embedding** (frozen). The FM is loaded from a pinned
   Hugging Face revision into `models/foundation/<name>@<revision>` and
   executed in `torch.inference_mode()` with all parameters frozen.
2. **Head training** (the only trainable component). A lightweight classifier
   (MLP / linear probe / shallow temporal head) trained on cached embeddings.

This split is enforced by the DVC pipeline: the `frozen_embeddings` stage is
the only place that touches FM weights, and its outputs are content-addressed.

## Evaluation

- Primary metric: macro-averaged F1 across the 8 classes (the canonical SHL
  metric; verify wording against the live challenge page before submission).
- Secondary breakdowns: per-class F1, per-user F1, per-location F1, latency.

## Submission

A plain-text `teamName_predictions.txt` of shape **92 726 × 500** with
per-sample class labels (integers in the challenge's class set), produced by
`dvc repro format_submission` and validated by
`src/shl2026/submission/validate.py` before being emailed (with a Dropbox /
Drive link) to `shldataset.challenge@gmail.com`.

## Known caveats

- The test set has no `Hand` location. We document the choice (train with
  Hand or drop it) explicitly per experiment and log it as a Hydra config.
- Test frames are shuffled. Any feature that crosses frame boundaries is
  forbidden at inference time.
- Foundation-model revision pins are critical: a silent HF model update would
  break reproducibility. All revisions are pinned by commit SHA in
  `params.yaml` and cached on Helios.
