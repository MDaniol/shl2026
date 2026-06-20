# SHL 2025 Challenge — Lessons for 2026 (from the official summaries)

Source: `LITERATURE/1_SHL_2025_Task1_Summary.pdf` (our exact task) +
`2_SHL_2025_Task2_Summary.pdf` (cross-domain; has a frozen-FM ablation). 2025 =
**identical setup** to 2026 (U1 B/T/Hi/Ha → U2/U3 B/T/Hi, 8 classes, 5 s, macro-F1).
Difference: 2025 *allowed* retraining/fine-tuning; **2026 is strictly frozen.**

## The single biggest fact: ranking = NOVELTY + paper quality, F1 = tiebreaker
Task 1 was ranked by a **6-member panel on novelty + paper quality**, mostly
**blind to test F1**; F1 only broke ties. Final: UT-IR 9.5, HELP 7, TDU-BSA 5.5.
(They also happened to be top-3 on F1.) → **Invest in a genuinely novel method
and an excellent paper, not only macro-F1.** (Confirm 2026 keeps this criterion.)

## Task 1 results (our task) — what scored what
| Rank | Team | F1 | Approach | Frozen-legal in 2026? |
|---|---|---|---|---|
| 1 | UT-IR | **82.97%** | **3 TS-FMs (MOMENT+BIOT+CBraMod)** → fuse → MLP | ❌ fine-tuned |
| 2 | TDU-BSA | 82.78% | TimesFM (partial FT) + multi-res + **handcrafted** → conv | ❌ fine-tuned |
| 3 | **HELP** | **81.37%** | **frozen pre-trained ViT on IMU→RGB image** → CNN | ✅ **pre-trained/frozen** |
| — | **Baseline** | **80.30%** | **CNN on Euclidean magnitude** (CNN_freq) | (reference) |
| 4 | SIAT-BIT | 77.52% | **TSFEL 1,872 handcrafted features + CatBoost** (no FM) | ✅ no FM |
| 5 | KMET | 70.60% | TimesNet + handcrafted → HIVE-COTE | ❌ re-trained |
| 6 | Li-Win | 54.58% | **TimesNet+Chronos+BERT+Flamingo (4 FMs)** → ensemble | mixed |
| 7 | BD-Canada | 49.30% | DistilBERT | ❌ |

## Direct frozen-FM head evidence (Task 2 organizer ablation, frozen MOMENT)
| Head on frozen MOMENT embeddings | F1 |
|---|---|
| **Random Forest** | **0.378** |
| Logistic Regression | 0.241 |
| SVM | 0.201 |
| (MOMENT fine-tuned, for reference) | 0.435 |
→ On frozen embeddings, a **nonlinear head (RF/GBM) ≫ linear probe**, and
fine-tuning beats frozen. *(Cross-domain 29-class numbers, so absolute values
differ from Task 1 — but the ordering is the actionable signal.)*

## Actionable lessons → how it changes our plan
1. **Vision branch is the strongest *frozen-legal* precedent.** HELP's **frozen ViT on IMU→RGB image (81.4%)** was the best pure-frozen result and beat the magnitude-CNN baseline. → **Promote the visual branch from "secondary" to co-primary.** HELP recipe: Acc→R, Gyr→G, Mag→B single-channel images combined into one RGB → frozen ViT. Simple, reproducible.
2. **Use a nonlinear head (LightGBM/RF/MLP), not linear probe**, on frozen embeddings (Task 2 ablation: RF 0.378 vs LR 0.241). Linear probe is for *ranking* models only.
3. **Winning architecture = ensemble of multiple COMPATIBLE FMs → MLP fusion** (UT-IR: 3 TS/biosignal models). But **kitchen-sink cross-modal fusion fails** (Li-Win's 4-modality ensemble → 6th). Fuse coherent models + handcrafted; don't pile on modalities.
4. **Handcrafted features are a top-tier branch**: pure TSFEL+CatBoost = 77.5%; magnitude-CNN baseline = 80.3% (beat 4 of 7 FM entries). → Our Branch A is well-justified; **TSFEL is a proven tool** (we have it installed).
5. **Avoid language/LLM FMs** — BERT/DistilBERT/Flamingo-inclusive entries were bottom. Deprioritize SensorLLM and any LLM-tokenization path.
6. **New model candidates the winner used: BIOT + CBraMod** (EEG/biosignal transformers, public) applied to IMU — novel for transport, and CBraMod's criss-cross attention models cross-channel interaction. Worth adding to the bake-off (novel + winner-validated).
7. **Realistic frozen target ≈ 81%** (HELP), with the **80.3% magnitude-CNN baseline** as the must-beat. Best-ever (fine-tuned/2020 non-FM) ≈ 88.5% — out of reach under frozen rule. Don't promise >85%.
8. **Confusions** confirmed: **Run→Walk** (≈28% of Run lost), **Car↔Bus**, **Train↔Subway**. Run recall is the macro-F1 killer → class-weight + threshold calibration.
9. **FMs were robust to overfitting** (predicted vs actual F1 gap <10pp) — frozen embeddings generalize train→test reasonably; good for us.
10. **Input contracts confirmed**: MOMENT = per-channel z-score, patch-based, attention pooling; Chronos = per-channel min-max [0,1], VQ tokenization. (Matches `PREPROCESSING_PLAN.md`.)

## Net pivots for 2026
- **Three co-primary branches**: (A) handcrafted (520 + TSFEL) + LightGBM; (B) frozen TS/IMU-FM embeddings (MantisV2/UTICA/UniMTS/LIMU-BERT; consider BIOT/CBraMod) → nonlinear head; (C) **frozen ViT on IMU→RGB image (HELP-style)** — now elevated.
- **Fuse coherently** (compatible models + handcrafted) → MLP/stacking; avoid LLM/kitchen-sink.
- **Optimize for novelty + paper** as much as macro-F1.
- **Target ≈81%+ frozen**, beat the 80.3% baseline; win on novelty.
</content>
