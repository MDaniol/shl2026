# SHL 2026 — Frozen-FM Usage Strategy (the HOW)

How to use frozen FMs with best efficiency/accuracy for SHL 2026. Complements
`STRATEGY.md` (which branches/models); this is the *usage* layer: head, pooling,
normalization, fusion, imbalance, generalization. Decision-oriented.

> Status note: detailed **SHL 2025** team results (refs [8] Task 1, [9] Task 2)
> are **paywalled (ACM)** and were not openly retrievable. Recommend pulling those
> PDFs directly (AGH library / authors) — they are the single most valuable
> precedent. What we *can* say: 2025 used MOMENT/Chronos/Flamingo/BERT in
> **pre-trained OR retrained** settings; **2026 forbids retraining (frozen only)**,
> so 2026 is strictly harder and the **head + fusion carry the load**.

## 1. Head on frozen embeddings — what to train
| Option | Use | Verdict |
|---|---|---|
| **Logistic regression / linear probe** | bake-off, calibrated baseline | ✅ default for *model comparison* (standard frozen-FM protocol; Mantis/MOMENT report this) |
| **LightGBM / gradient boosting** | final + fusion of embeddings ⊕ handcrafted | ✅ best for **heterogeneous, different-scale** features; scale-invariant; what won SHL 2024 (classical) |
| **Small MLP (LayerNorm+dropout)** | learned fusion, class-weighted/focal | ✅ when we want learned cross-feature interaction + macro-F1 loss |
| kNN | quick sanity | 🟡 diagnostic only |
**Call:** linear probe to *rank* FMs in the bake-off; **LightGBM and a small MLP** for the *final* fused model — pick by val macro-F1.

## 2. Embedding extraction — pooling & layer
- Use each model's **documented embedding output** (MantisV2 `.transform`, MOMENT embedding task) — don't hand-roll layer extraction first.
- **Channel-independent models (MantisV2/UTICA/MOMENT)**: encode each of 9 channels → **concatenate** (9×d). Ablate **concat vs mean-pool across channels** (concat keeps per-sensor detail; mean-pool is smaller, more invariant).
- Multi-layer concat can add signal but inflates dim → only if headroom.
**Call:** documented embedding, per-channel **concat**; ablate mean-pool.

## 3. Embedding normalization (matters for linear/MLP)
- **Standardize (z-score, train stats)** or **L2-normalize** embeddings before a linear/MLP head — frozen embeddings often have anisotropic scales; this materially helps linear probes.
- **LightGBM**: no normalization needed (trees are scale-invariant).
**Call:** standardize for linear/MLP; raw for trees. (Note: the FM's *input* normalization stays per its contract — MantisV2 z-scores internally, LIMU-BERT bespoke; see `PREPROCESSING_PLAN.md`.)

## 4. Fusion & ensembling (the axis the challenge rewards)
- **Concat-fusion**: [MantisV2 ⊕ UTICA ⊕ … ⊕ 520 handcrafted] → one head. Simple, strong.
- **Stacking**: train a head per branch → combine out-of-fold logits with a meta-learner. More robust, controls per-branch overfit; recommended for the final.
- Evidence: fusion beats any single branch (TiViT: frozen-ViT + Mantis > either; TimesFM-fusion: handcrafted ⊕ FM > FM alone).
**Call:** concat-fusion baseline → **stacking ensemble** for the final submission. Optional novel "interface": a tiny trainable **cross-FM attention/`LinearChannelCombiner`** adapter (allowed — backbone frozen) for the paper's novelty.

## 5. Generalization: train(1 user, 4 loc) → test(other users, 3 loc, no Hand)
- **Train the head on all 4 locations pooled** → it sees location diversity → better transfer. **Validate per-location**, and ensure **Bag/Hips/Torso** (the test locations) are strong; Hand stays in training (still teaches motion) but is *down-weighted in model selection*.
- **Per-frame test-time robustness**: test is per-frame independent → the FMs' **internal instance-norm** already adapts each frame; for handcrafted features, rely on **window-internal** + train-fit standardization (no cross-frame stats — shuffled).
- **Lightweight domain alignment** (allowed, no FM update): **CORAL / feature whitening** of embeddings toward the val/location distribution, or simply standardize per-location. Cheap, ablate.
- Classic **DANN over users is impossible** (1 train user) → exploit the **location** axis only.
**Call:** pool locations, per-location validation, optional CORAL/whitening ablation.

## 6. Macro-F1 + Run imbalance (4.3%)
- **Class-weighted loss** (inverse-frequency) or **focal loss** (MLP); **balanced/stratified sampling** for trees.
- **Per-class decision-threshold tuning on validation** to directly maximize **macro-F1** (don't use argmax-only). Biggest cheap macro-F1 lever.
- Watch the known-hard pairs (Car↔Bus, Train↔Subway) and rare **Run** in the confusion matrix.
**Call:** class-weighted training + **per-class threshold calibration on val**.

## 7. Efficiency / ROI (≈10 days, 36 GB MacBook)
- **Cache embeddings once** (MantisV2/UTICA tiny → local CPU/MPS).
- ROI order: (1) handcrafted + LightGBM → valid submission + first macro-F1; (2) add MantisV2/UTICA embeddings via concat-fusion; (3) stacking ensemble + threshold calibration; (4) UniMTS/LIMU-BERT/NormWear and vision branch only if they add on val.

## Prioritized action list
1. **Cache** MantisV2 + UTICA embeddings (all splits) with `fm_input` packers.
2. **Linear-probe bake-off** → rank FMs by **val macro-F1** (users 2&3), per-location.
3. **Fuse** best FM embeddings ⊕ 520 handcrafted → **LightGBM + MLP**; compare.
4. **Calibrate** per-class thresholds on val for macro-F1; class-weight Run.
5. **Stack** branches; add UniMTS/LIMU-BERT/NormWear/vision iff they lift val.
6. **Predict** test → 92 726×500; sanity-check class prior; submit with buffer.

## Honest caveats
- No SHL-transport macro-F1 published for these FMs → all priors; **bake-off decides**.
- Frozen < fine-tuned (Mantis) and fine-tuning is banned → head + fusion + handcrafted are where we win.
- SHL 2025 detailed results not retrieved (paywalled) — get refs [8],[9] for direct precedent.
- Citation fix (from agent): **RevIN = Kim et al., ICLR 2022, OpenReview `cGDAkQo1C0p`** — *not* arXiv:2105.03801 (a different paper).
</content>
