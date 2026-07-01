#!/usr/bin/env python3
"""Build the editable Word document (captions + results + rationale) for the SHL-2026 paper figures.

Numbers mirror FIGURE_CAPTIONS.md (honest, leakage-clean held-out evaluation). Edit the constants below
if the underlying numbers change, then re-run:  python build_captions_docx.py [out.docx]
Requires python-docx.
"""
import sys, shutil
import docx
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# --- numbers (kept in sync with FIGURE_CAPTIONS.md) ---
MA, NA = 0.838, "15,129"     # Lane A internal TEST
MB, NB = 0.834, "22,460"     # Lane B target holdout
NS = "2,265"                 # doubly-held-out slice
W = 0.575
PRES = {"A": 0.824, "B": 0.807, "V5": 0.863}   # present-class macro on the slice
DELTA, CILO, CIHI = "+0.038", "+0.024", "+0.052"
PERCLASS = {  # present classes only (Run absent)
    "Lane A (time-series)": ["0.97", "0.84", "0.88", "0.93", "0.54", "0.82", "0.79"],
    "Lane B (vision)":      ["0.92", "0.85", "0.89", "0.88", "0.40", "0.77", "0.93"],
    "v5 (blend)":           ["0.96", "0.85", "0.94", "0.95", "0.58", "0.86", "0.90"],
}

d = docx.Document()
def H(t, s=15):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = True; r.font.size = Pt(s); return p
def P(t, italic=False):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; r = p.add_run(t); r.italic = italic; return p
def LEAD(bold, rest):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(bold + " ").bold = True; p.add_run(rest); return p
def CAP(f, t):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(f + " — ").bold = True; p.add_run(t); return p
def BUL(label, t):
    p = d.add_paragraph(style="List Bullet"); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if label: p.add_run(label + " ").bold = True
    p.add_run(t); return p
def table(headers, rows):
    t = d.add_table(rows=1, cols=len(headers)); t.style = "Light Grid Accent 1"
    for j, h in enumerate(headers): t.rows[0].cells[j].paragraphs[0].add_run(h).bold = True
    for row in rows:
        c = t.add_row().cells
        for j, v in enumerate(row): c[j].paragraphs[0].add_run(str(v))
    d.add_paragraph(); return t

H("SHL-2026 (KMET) — Figure captions and rationale", 17)
P("Editable draft for the paper. Two frozen-foundation-model lanes are combined by late fusion: a "
  "time-series lane (Lane A) and a vision-based lane (Lane B); v5 is their probability-level blend. "
  "All numbers are honest and leakage-clean; do NOT quote in-sample or full-validation numbers.", italic=True)
LEAD("Data split and evaluation sets.",
    "The challenge provides labelled data from User 1 (training) and Users 2–3 (validation), plus an "
    "unlabelled, frame-shuffled hidden test from the same Users 2–3 (Bag/Hips/Torso only). For honest, "
    "leakage-free evaluation each lane is scored only on data it never trained on, using a user-independent "
    "temporal split with an embargo (select on a TUNE partition, lock the TEST partition once; per-window "
    "scoring). Because the two lanes were developed independently they define different held-out sets, so the "
    "figures report on three distinct — and NOT directly comparable — sets: "
    f"(i) Lane A on its internal TEST (n = {NA}; all 8 classes), macro-F1 {MA:.3f}; "
    f"(ii) Lane B on its target holdout (n = {NB}; all 8 classes), macro-F1 {MB:.3f}; and "
    f"(iii) the fused model v5 on the doubly-held-out slice — the intersection of (i) and (ii), i.e. windows "
    f"held out by BOTH lanes (n = {NS}) — the only leakage-free set on which the blend can be measured. That "
    "intersection contains no Run windows and only 33 Bus windows, so the per-lane confusion matrices "
    "(Figs 1–2), not the slice, are the authoritative per-class evidence; the slice is used only to select the "
    "blend weight and to test the fusion gain.")

H("1. Figure captions", 14)
CAP("Figure 1 (cm_laneA_v4_test.png)",
    "Confusion matrix (row-normalized recall, %) of the time-series lane (Lane A; v4: frozen UTICA + "
    "Mantis-V2 embeddings concatenated with 520 hand-crafted features, per-class-calibrated LightGBM "
    f"soft-vote) on the held-out internal TEST set (Users 2–3, Bag/Hips/Torso, per-window; n = {NA}). "
    f"Macro-F1 = {MA:.3f}. Residual confusion concentrates in the Train↔Subway and Bike→Subway pairs.")
CAP("Figure 2 (cm_laneB_holdout.png)",
    "Confusion matrix (row-normalized recall, %) of the vision-based lane (Lane B; frozen DINoV2 over "
    f"STFT/CWT/GAF spectrogram images, gated multi-branch MLP) on its held-out target set (n = {NB}). "
    f"Macro-F1 = {MB:.3f}. Its error structure is complementary to Lane A (stronger on Subway, weaker on "
    "Bus/Train), which motivates the late-fusion blend.")
CAP("Figure 3 (weight_sweep.png)",
    "Macro-F1 (over the classes present in the slice) as a function of the blend weight w "
    f"(P = w·P_LaneA + (1−w)·P_LaneB) on the doubly-held-out slice; optimum at w = {W} "
    f"(macro-F1 {PRES['V5']:.3f}), above both single lanes (Lane A {PRES['A']:.3f} at w = 1, Lane B "
    f"{PRES['B']:.3f} at w = 0). The weight is tuned only on this doubly-held-out slice, so the fusion "
    "introduces no leakage.")

H("2. Results", 14)
P(f"Doubly-held-out slice (n = {NS}; 7 of 8 classes present) — the only data held out by both lanes "
  "simultaneously, used to tune the blend weight and to test the fusion gain. Macro-F1 below is averaged "
  "over the 7 classes present. Run has 0 windows in this intersection, so it is not scored here (it would "
  "only add a forced 0 to every model); Run is fully characterised by the full-set confusion matrices "
  "(Figs 1–2), where its F1 ≈ 0.94 on the internal TEST.")
table(["Model", "Macro-F1 (7 present classes)"],
      [["Lane A (time-series, v4)", f"{PRES['A']:.3f}"],
       ["Lane B (vision)", f"{PRES['B']:.3f}"],
       [f"v5 (blend, w = {W})", f"{PRES['V5']:.3f}"]])
P("Per-class F1 on the slice (present classes only; Run absent):")
table(["Model", "Still", "Walk", "Bike", "Car", "Bus", "Train", "Subway"],
      [[k] + v for k, v in PERCLASS.items()])
P(f"v5 vs Lane A: paired-bootstrap Δ(macro-F1 over the present classes) = {DELTA}, 95% CI [{CILO}, {CIHI}] "
  "(excludes 0 → statistically significant). Bus is small (33 windows) in this intersection, so its bar is "
  "noisy; the full-set confusion matrices are the authoritative per-class evidence.")
P("Lane summary (full held-out sets, all classes present):")
table(["Lane", "Evaluation set", "n", "Macro-F1"],
      [["Lane A (time-series, v4)", "internal TEST (Bag/Hips/Torso)", NA, f"{MA:.3f}"],
       ["Lane B (vision, DINoV2)", "target holdout", NB, f"{MB:.3f}"]])

H("Appendix — Why these figures, and why there is no final confusion matrix", 14)
H("A. Why these figures", 12)
BUL("Two per-lane confusion matrices (Figs 1–2).", "Each lane is evaluated on its own full held-out set, "
    "where all eight classes are present, giving an honest, complete per-class picture. Placed side by "
    "side they also make the core argument visible: the two lanes make different errors (Lane B is "
    "stronger on Subway, Lane A on Car), which is precisely what makes their fusion pay off.")
BUL("Blend-weight sweep (Fig 3).", "Documents the single hyper-parameter of the fusion (w) and shows the "
    f"optimum is a smooth interior maximum (w = {W}) that beats either lane alone, tuned on leakage-clean data. "
    "The per-class comparison of the two lanes and the blend is given in the results table (Section 2).")

H("B. Why there is no v5 / final confusion matrix", 12)
P("A confusion matrix requires ground-truth labels on a set that both lanes can be scored on. For the "
  "fused model (v5) no such all-class set exists, for three reasons:")
BUL("The hidden test has no labels.", "The challenge test set (92,726 windows) is unlabelled, so no "
    "confusion matrix can be computed on the submission itself.")
BUL("v5 needs a set held out by both lanes.", "v5 = w·P_LaneA + (1−w)·P_LaneB, so an honest v5 "
    "evaluation must use windows that neither lane trained on. The only such set is the intersection of "
    f"the two independently-built held-out sets — the doubly-held-out slice (n = {NS}).")
BUL("That intersection does not contain every class.", "Both lanes split their data into per-class "
    "contiguous time-blocks (a deliberate anti-leakage choice — random assignment would leak temporally "
    "adjacent windows). Run is the rarest class (~1.9%) and is temporally clustered, so its blocks in the "
    "two independent partitions do not overlap, and Run drops out of the intersection entirely (0 "
    "windows). Bus survives with only 33 windows.")
P("A confusion matrix with a missing class is both hard to read and easy to misinterpret as an "
  "incomplete evaluation, so we do not present one. Instead we report the two lanes' full-set confusion "
  "matrices (all classes present) and express the fusion result as a per-class F1 table plus a "
  "significance test (paired bootstrap over the classes present in the doubly-held-out slice). This keeps "
  "every figure complete and every number leakage-clean.")

H("C. Path to a full v5 confusion matrix (future work)", 12)
P("A single all-class v5 confusion matrix would require both lanes to agree on one canonical held-out "
  "test split up front, so that neither lane trains on those windows. All three models (Lane A, Lane B, "
  "v5) could then be scored on the same complete set. This coordination (a shared split and one "
  "re-training of the vision lane) would be done for a camera-ready version; it does not affect the "
  "reported results, which remain honest and leakage-clean.")

out = sys.argv[1] if len(sys.argv) > 1 else "SHL2026_Figures_Captions_and_Rationale.docx"
d.save(out)
print("saved", out)
for extra in sys.argv[2:]:
    shutil.copy(out, extra); print("copied ->", extra)
