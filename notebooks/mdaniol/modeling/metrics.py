#!/usr/bin/env python3
"""Shared metrics: per-class precision/recall/F1/support + averaged + confusion.

Every experiment records the full per-class breakdown (so we can see where Run /
Car↔Bus / Train↔Subway hurt) AND the averaged macro/weighted/accuracy, persisted
to JSON for traceability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

CLASSES = list(range(1, 9))
CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]


def class_report(y_true, y_pred, labels=CLASSES, names=CLASS_NAMES) -> dict:
    """Per-class precision/recall/F1/support + macro/weighted/accuracy + confusion."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = {names[i]: {"precision": round(float(p[i]), 4),
                            "recall": round(float(r[i]), 4),
                            "f1": round(float(f[i]), 4),
                            "support": int(s[i])} for i in range(len(labels))}
    return {
        "per_class": per_class,
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels,
                                         average="macro", zero_division=0)), 4),
        "macro_precision": round(float(np.mean(p)), 4),
        "macro_recall": round(float(np.mean(r)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels,
                                            average="weighted", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def print_report(tag: str, rep: dict) -> None:
    print(f"  [{tag}] macro-F1={rep['macro_f1']:.4f}  acc={rep['accuracy']:.4f}  "
          f"macro-P={rep['macro_precision']:.4f} macro-R={rep['macro_recall']:.4f}")
    for n, d in rep["per_class"].items():
        print(f"      {n:7s} P={d['precision']:.3f} R={d['recall']:.3f} "
              f"F1={d['f1']:.3f} (n={d['support']})")


def save_report(path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(path, "w"), indent=2)
