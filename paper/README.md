# Paper — HASCA-SHL 2026

This directory holds the HASCA-SHL workshop paper (3–6 pages, ACM SIGCHI
Master template, sigconf 2-column).

## Template

Use the official template from the challenge page ("TEMPLATES
ISWC/UBICOMP2026"). The ACM `acmart` class is the canonical choice
(`\documentclass[sigconf,nonacm=false]{acmart}`). Do **not** vendor the class
files into this repo; rely on the TeX Live distribution inside the container.

## Build

```bash
latexmk -pdf -interaction=nonstopmode -file-line-error paper.tex
```

## Figures

Figures are generated, not hand-drawn:

```bash
python -m shl2026.eval.make_paper_figures --run <mlflow-run-id> --out figures/
```

`paper/figures/runs.yaml` (to be added) records the exact MLflow run IDs that
back each figure and table. Camera-ready edits must rerun the figure
generator to ensure the PDF matches the artefacts on Zenodo.

## Submission

PrecisionConference, track: *SIGCHI / UbiComp 2026 Workshop – HASCA-SHL*.
Deadline (from the challenge page snapshot 2026-05-25): **04.07.2026** for
submission, **22.07.2026** for camera-ready.
