# SHL 2026 — Novelty Strategy

The challenge ranks on **novelty + paper quality** (panel, mostly F1-blind; see
`SHL_2025_LESSONS.md`). So novelty is a first-class deliverable, not a garnish.
Our locked headline hook is **(b) a cross-modal frozen ensemble (TS-FM +
IMU-image-ViT + handcrafted) with a learned fusion adapter** (`STRATEGY.md`).
This file records *where* to find novelty and *how* to validate it.

## Where to look — the open gaps in SHL precedent
Every 2025 team bolted a frozen FM to a `concat → MLP`. The gaps:

1. **The interface, not the model.** Novelty lives in *how* models connect:
   - learned **cross-FM / cross-modal fusion adapter** (attention or gating that
     weights *which FM to trust per-class or per-window*),
   - a **channel-combiner** modeling **Acc↔Gyr↔Mag interaction** on top of
     channel-independent backbones (MantisV2/UTICA encode channels independently),
   - **embedding alignment / whitening** (e.g. CORAL) across the location shift.
   → This is our **headline (b)**.
2. **New/underutilized frozen models.** MantisV2/UTICA (2026, never used in SHL)
   used *frozen*; **biosignal FMs (BIOT, CBraMod) transferred to transport IMU** —
   EEG/physiology → locomotion transfer is a genuinely fresh framing.
3. **The frozen-only constraint as a contribution.** 2025's top teams *fine-tuned*
   (now illegal). A principled **frozen-only ensemble that matches fine-tuned
   performance** is itself a novel, publishable result.
4. **Representation / input novelty.** Which **IMU→image encoding × which frozen
   vision FM** (unresolved in the literature — see `VISUAL_STRATEGY.md`);
   **magnetometer-centric modeling** for rail vs road; **physics-guided embedding
   fusion**.
5. **The shuffled / per-frame test angle.** Most teams ignore that the test is
   per-frame and shuffled. A method **explicitly designed for per-frame robustness
   without temporal context** is sharp and defensible.
6. **Error-structure-driven design.** The universal confusions (**Run→Walk,
   Car↔Bus, Train↔Subway**) invite a **targeted expert head** — e.g. a
   **magnetometer-spectral expert for rail** — as concrete, evaluable novelty.

## How to do novelty research — a method, not a vibe
1. **Build a "what's-been-done" matrix.** Axes = {model family × interface ×
   fusion × generalization tactic × representation}, filled from the SHL 2018–2025
   summaries + our FM/visual/preprocessing deep-research. **Empty cells = candidate
   novelty.** (Starter below.)
2. **Phrase each idea as a falsifiable claim + its ablation.** e.g. *"A learned
   cross-FM attention adapter beats concat-fusion by X macro-F1 under the frozen
   rule."* If you can't design the ablation, it's a feature, not a contribution.
3. **Triangulate three lenses** and keep only ideas scoring on all three:
   - **Novelty** — cell empty / absent from recent arXiv,
   - **Feasibility** — buildable + ablatable in ~10 days, public weights,
   - **Impact** — moves macro-F1 *or* teaches the community something (negative
     results count under a frozen-rule challenge).
4. **Search the adjacent frontier deliberately** (deep-research pattern): newest
   2025–2026 work on frozen-FM fusion/adapters, parameter-efficient probing,
   cross-domain / biosignal→IMU transfer, time-series-as-image. Most novelty is
   *"newest technique from field X, first applied to SHL transport."*
5. **Verify the gap before committing.** Confirm no SHL team already did it (mine
   the 2024/2025 team papers) and it isn't trivially in a recent paper — treat
   *"this is novel"* as a hypothesis to **disprove**.
6. **Let the data reveal novelty.** Bake-off findings are themselves publishable
   novelty you can't predict: e.g. *"frozen biosignal FMs transfer surprisingly
   well to transport,"* or *"magnetometer-only embeddings carry most of the rail
   signal."*

## "What's-been-done" matrix — starter (fill as we mine papers)
| Axis | Done in SHL ≤2025 | Open cell (candidate novelty) |
|---|---|---|
| Interface | concat → MLP (all teams) | **learned cross-FM attention/gating adapter** |
| Channel modeling | per-channel concat; CBraMod criss-cross (winner) | **trainable Acc↔Gyr↔Mag combiner on frozen channel-indep backbones** |
| Fusion scope | 3 TS-FMs (UT-IR); kitchen-sink failed (Li-Win) | **coherent TS + image-ViT + handcrafted, frozen-only** |
| Generalization | data selection, location pooling | **embedding whitening/CORAL for location shift, frozen** |
| Representation | IMU→RGB image→ViT (HELP); spectrograms | **encoding × frozen-vision-FM sweep; magnetometer-spectral expert** |
| Training regime | fine-tune (top 2, now illegal) | **frozen-only matching fine-tuned = the contribution** |
| Test handling | (ignored shuffling) | **explicit per-frame, temporal-context-free design** |

## Candidate falsifiable claims (to ablate)
- C1: *A learned cross-FM attention adapter > concat-fusion by ≥X macro-F1 (frozen).*
- C2: *A trainable Acc↔Gyr↔Mag channel-combiner on frozen MantisV2/UTICA > per-channel concat.*
- C3: *Frozen biosignal FMs (BIOT/CBraMod) transfer competitively to transport IMU.*
- C4: *A magnetometer-spectral expert head raises Train/Subway F1 by ≥X.*
- C5: *Embedding whitening/CORAL improves the Hand→{Bag,Hips,Torso} location shift.*
- C6: *A frozen-only ensemble matches the 2025 fine-tuned top (~83%) — the headline result.*

## How this enters the workflow
- Bake-off logs every cell we test → fills the matrix → each filled cell is a paper ablation.
- Headline = **(b)** interface novelty (C1/C2); supporting = C3 (biosignal transfer), C6 (frozen-only result), C4/C5 as targeted ablations.
- Anything that fails its ablation is reported honestly (negative results are valid here).
</content>
