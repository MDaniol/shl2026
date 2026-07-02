#!/usr/bin/env python3
"""Build the ADDENDUM Word document: the all-class v5 confusion matrix on a shared leakage-clean set.
Supplements SHL2026_Figures_Captions_and_Rationale.docx. Edit constants below if numbers change, re-run:
    python build_addendum_docx.py [out.docx [copy_dest ...]]
"""
import sys, shutil
import docx
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# --- numbers (final_cm.py, embargo=100, BHT-only, w-tuning windows excluded) ---
N = "14,580"
EMB = 100
W = 0.575
SHARED = {"A": 0.856, "B": 0.854, "V5": 0.892}
DELTA, CILO, CIHI = "+0.036", "+0.032", "+0.041"
FULL = {"A": 0.838, "B": 0.834}          # representative full-set numbers
SLICE_GAIN = "+0.038"                     # gain on the doubly-held-out slice (for the consistency point)
LEAK_SHIFT = 0.008                        # Lane A drop when the embargo was added

d = docx.Document()
def H(t, s=15):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = True; r.font.size = Pt(s); return p
def P(t, italic=False):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; r = p.add_run(t); r.italic = italic; return p
def LEAD(bold, rest):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(bold + " ").bold = True; p.add_run(rest); return p
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

H("SHL-2026 (KMET) — Addendum: all-class v5 confusion matrix", 16)
P("This addendum supplements the figure-captions document (SHL2026_Figures_Captions_and_Rationale). After "
  "that document was prepared we produced a full all-class confusion matrix for the fused model (v5) on a "
  "leakage-clean shared held-out set, resolving the limitation described in its Appendix B/C. All numbers "
  "remain honest and leakage-clean; do NOT quote in-sample or full-validation numbers.", italic=True)

H("1. Method — a shared, leakage-clean, all-class held-out set", 13)
P("A confusion matrix for v5 needs windows that BOTH lanes held out. We reuse the vision lane's "
  "target_holdout (which its model never trained on, and which contains all 8 classes) and re-fit only the "
  f"time-series lane (Lane A) to hold out exactly those windows, applying a temporal embargo of {EMB} "
  "windows so Lane A never trains on windows adjacent to the evaluation set. The two lanes' probabilities "
  f"are then blended with the pre-locked weight w = {W}. The set is restricted to Bag/Hips/Torso (matching "
  f"the real test, which has no Hand) and excludes the windows previously used to select w. Shared set: "
  f"n = {N}, all 8 classes present. No part of the vision lane is re-run.")

H("2. Results (shared held-out set)", 13)
table(["Model", f"Macro-F1 (shared set, n={N})"],
      [["Lane A (time-series)", f"{SHARED['A']:.3f}"],
       ["Lane B (vision)", f"{SHARED['B']:.3f}"],
       [f"v5 (blend, w = {W})", f"{SHARED['V5']:.3f}"]])
P(f"v5 vs Lane A: paired-bootstrap Δ(macro-F1) = {DELTA}, 95% CI [{CILO}, {CIHI}] (excludes 0 → "
  "statistically significant). The confusion matrix itself is the file cm_v5_final.png (all 8 classes, "
  "row-normalized recall); per-class F1 for all three models is in FINAL_CM_RESULTS.md.")

H("3. How to read these numbers (important)", 13)
LEAD("The fusion gain is the robust, reportable result.",
     f"v5 improves on the stronger lane by {DELTA} on this set — essentially identical to the {SLICE_GAIN} "
     "measured earlier on the independent doubly-held-out slice. The same gain appearing on two independent "
     "leakage-clean sets is strong evidence the blend genuinely helps.")
LEAD("The absolute level (~0.89) is subset-specific, not a headline number.",
     f"It is higher than the full-set per-lane scores (Lane A {FULL['A']:.3f}, Lane B {FULL['B']:.3f}) and "
     "than the expected hidden-test level (~0.84–0.85) because this particular clean subset is somewhat "
     "easier. This is NOT a leakage artifact: Lane B here uses the exact same fixed, already-clean "
     f"predictions and also rises to {SHARED['B']:.3f}; and adding the {EMB}-window embargo changed Lane A "
     f"by only ~{LEAK_SHIFT:.3f}. Different (scattered) held-out windows simply make this subset easier.")
LEAD("Recommended use.",
     "Present the v5 confusion matrix for its per-class structure (all classes, including Run) and for the "
     f"same-set {DELTA} gain over each lane. Keep the full-set per-lane macro-F1 (Lane A {FULL['A']:.3f}, "
     f"Lane B {FULL['B']:.3f}) and the organizers' hidden-test score as the representative overall "
     "performance. Do not headline the ~0.89.")

H("4. Relation to the main document", 13)
BUL("Supersedes Appendix B.", "The main document stated that no all-class v5 confusion matrix was available; "
    "this addendum provides one, computed on a shared leakage-clean set (Lane A re-fit to hold it out).")
BUL("Partially addresses Appendix C.", "The gold-standard remains a single canonical held-out split "
    "excluded by BOTH lanes with a full coordinated re-run of the vision lane; here we approximate it by "
    "re-fitting only Lane A. The residual limitation is that the shared set is the vision lane's own "
    "(scattered) holdout rather than a jointly-designed contiguous block, which is why its absolute level is "
    "subset-specific. A fully coordinated shared split is left for a camera-ready version.")

out = sys.argv[1] if len(sys.argv) > 1 else "SHL2026_Addendum_FinalCM.docx"
d.save(out)
print("saved", out)
for extra in sys.argv[2:]:
    shutil.copy(out, extra); print("copied ->", extra)
