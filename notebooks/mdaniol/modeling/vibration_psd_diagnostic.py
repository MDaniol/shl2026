#!/usr/bin/env python3
"""Phase-1 vibration diagnostic (pre-registered in VEHICLE_EXPERIMENT_PLAN.md).

Leakage-free, training-free descriptive analysis that GATES the vehicle feature work.
Answers two questions before we spend compute on feature engineering:

  H1 (engine band): do combustion vehicles (Car/Bus) show a class-distinct accelerometer
     spectral peak in ~18-40 Hz that is absent in Still / electric rail, and that our
     current <=20 Hz feature bank misses?
  H2 (raw-axis mag rail): do Train vs Subway separate on RAW-AXIS magnetometer spectra/DC
     (the cue the orientation-invariant magnitude transform destroys)?

Method: per-class Welch PSD (fs=100, 1 s Hann segments -> 1 Hz resolution) on raw acc
magnitude, body-acc magnitude, and the three raw magnetometer axes, STRATIFIED by placement
(Bag/Hips/Torso, mirroring the test). Plus mutual information of candidate band-features vs
class AND vs placement (the leakage tripwire: a real cue has high MI with class, low with
placement). Computed on the VALIDATION set (Users 2&3 — same distribution as the test).

Outputs (all MLflow-logged per AI_GUIDELINES.md §8): a markdown verdict table, the full PSD
curves as JSON (so figures can be drawn later without re-running), and figures IF matplotlib
is available (it is optional — the numbers are the deliverable; we do not touch uv.lock).

Usage:  python vibration_psd_diagnostic.py            # validation, Bag/Hips/Torso
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.feature_selection import mutual_info_classif

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "feature_extraction"))
import shl_features as shl  # noqa: E402  (reuse the validated gravity filter + magnitude)

FS = shl.FS                                   # 100 Hz
NPERSEG = 100                                 # 1 s segment -> 1 Hz resolution, freqs 0..50
CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]
BHT = ("Bag", "Hips", "Torso")               # the test placements (no Hand)
ENGINE_BAND = (18.0, 40.0)                    # combustion idle vibration (firing freq ~20-35 Hz)
LOW_BAND = (1.0, 15.0)                        # what our current features already cover
RAIL_CLASSES = (7, 8)                         # Train, Subway


def majority_label(label_cells) -> np.ndarray:
    """Per-window majority vote over the 500 per-sample labels (matches feature extraction)."""
    return np.array([np.bincount(np.asarray(r)).argmax() for r in label_cells], dtype=int)


def stack_axes(df: pd.DataFrame, prefix: str, idx: np.ndarray) -> np.ndarray:
    """Selected windows of one tri-axial sensor -> (len(idx), 3, 500) float32."""
    ax = [np.stack(df[f"{prefix}_{c}"].values[idx]).astype(np.float32) for c in "xyz"]
    return np.stack(ax, axis=1)               # (n, 3, N)


def band_energy(P: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Sum of PSD over [lo, hi] Hz (last axis)."""
    return P[..., (f >= lo) & (f <= hi)].sum(-1)


def spectral_flatness(P: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Geometric/arithmetic mean of PSD over 0.5-50 Hz: ~1 broadband (road/rail), <1 tonal (engine)."""
    band = P[..., f >= 0.5]
    gm = np.exp(np.log(band + 1e-20).mean(-1))
    return gm / (band.mean(-1) + 1e-20)


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--data-dir", type=Path, default=root / "dataset_parquet")
    ap.add_argument("--split", default="validation", help="validation = Users 2&3 (test-like)")
    ap.add_argument("--locs", default=",".join(BHT))
    ap.add_argument("--max-per-class-loc", type=int, default=1500,
                    help="subsample per (class, placement) for a bounded, fast diagnostic")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol/VIBRATION_DIAGNOSTIC.md")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    locs = args.locs.split(",")

    # --- collect Welch PSDs (subsample first, then load only selected windows) -----------
    # per stream: {class_id: [list of (n_seg, 51) PSD rows]} accumulated across placements
    streams = ("acc_mag", "acc_body_mag", "mag_x", "mag_y", "mag_z")
    psd = {s: {c: [] for c in range(1, 9)} for s in streams}
    feat_rows, feat_cls, feat_loc = [], [], []          # for MI (per window)
    mag_dc = {c: [] for c in RAIL_CLASSES}              # raw |mag| DC (mean field) for rail
    freqs = None

    for loc in locs:
        df = pd.read_parquet(args.data_dir / args.split / f"{loc}.parquet")
        y = majority_label(df["label"].values)
        for c in range(1, 9):
            cand = np.where(y == c)[0]
            if len(cand) == 0:
                continue
            sel = cand if len(cand) <= args.max_per_class_loc else \
                rng.choice(cand, args.max_per_class_loc, replace=False)
            acc = stack_axes(df, "Acc", sel)                       # (n,3,N)
            mag = stack_axes(df, "Mag", sel)
            grav = shl._lowpass_gravity(acc)
            sig = {"acc_mag": shl._mag(acc), "acc_body_mag": shl._mag(acc - grav),
                   "mag_x": mag[:, 0], "mag_y": mag[:, 1], "mag_z": mag[:, 2]}
            for s in streams:
                f, P = welch(sig[s], fs=FS, nperseg=NPERSEG, noverlap=NPERSEG // 2, axis=-1)
                freqs = f
                psd[s][c].append(P)
            # candidate band-features (per window) on body-acc -> MI vs class / placement
            f, Pb = welch(sig["acc_body_mag"], fs=FS, nperseg=NPERSEG, noverlap=NPERSEG // 2, axis=-1)
            eng = band_energy(Pb, f, *ENGINE_BAND)
            tot = band_energy(Pb, f, 0.5, 50.0) + 1e-20
            in_band = Pb[:, (f >= ENGINE_BAND[0]) & (f <= ENGINE_BAND[1])]
            peak_prom = in_band.max(1) / (np.median(in_band, 1) + 1e-20)
            feat_rows.append(np.column_stack([eng, eng / tot, peak_prom, spectral_flatness(Pb, f)]))
            feat_cls.append(np.full(len(sel), c)); feat_loc.append(np.full(len(sel), loc))
            if c in RAIL_CLASSES:
                mag_dc[c].append(shl._mag(mag).mean(1))            # |B| per window
        del df
        print(f"[psd] {loc} done ({time.time()-t0:.0f}s)", flush=True)

    # --- aggregate: per-class median PSD ------------------------------------------------
    def med(s, c):
        return np.median(np.concatenate(psd[s][c]), axis=0) if psd[s][c] else np.full_like(freqs, np.nan)

    curves = {"freqs": freqs.tolist(),
              "median_psd": {s: {CLASS_NAMES[c-1]: med(s, c).tolist() for c in range(1, 9)}
                             for s in streams}}

    # band-power table (acc body-acc): engine ratio per class
    def ratio(c):
        m = med("acc_body_mag", c)
        return float(band_energy(m, freqs, *ENGINE_BAND) / (band_energy(m, freqs, 0.5, 50.0) + 1e-20))

    def peakf(c):                                          # dominant freq within engine band
        m = med("acc_body_mag", c); band = (freqs >= ENGINE_BAND[0]) & (freqs <= ENGINE_BAND[1])
        return float(freqs[band][m[band].argmax()])

    eng_ratio = {CLASS_NAMES[c-1]: ratio(c) for c in range(1, 9)}
    eng_peak = {CLASS_NAMES[c-1]: peakf(c) for c in range(1, 9)}

    # --- MI: engine band-features vs class vs placement (leakage tripwire) ---------------
    F = np.concatenate(feat_rows); yc = np.concatenate(feat_cls)
    yl = np.concatenate([np.array(a) for a in feat_loc])
    fnames = ["engine_energy", "engine_ratio", "peak_prominence", "spectral_flatness"]
    mi_class = mutual_info_classif(F, yc, random_state=0)
    mi_loc = mutual_info_classif(F, pd.factorize(yl)[0], random_state=0)

    # --- verdict heuristics -------------------------------------------------------------
    veh = ["Car", "Bus"]; ref = ["Still", "Train", "Subway"]
    veh_ratio = np.mean([eng_ratio[c] for c in veh]); ref_ratio = np.mean([eng_ratio[c] for c in ref])
    h1 = veh_ratio > 1.25 * ref_ratio and mi_class.mean() > mi_loc.mean()
    dc_tr = float(np.mean(np.concatenate(mag_dc[7]))) if mag_dc[7] else float("nan")
    dc_sub = float(np.mean(np.concatenate(mag_dc[8]))) if mag_dc[8] else float("nan")

    # --- write markdown + JSON ----------------------------------------------------------
    rows = "".join(f"| {CLASS_NAMES[c-1]} | {eng_ratio[CLASS_NAMES[c-1]]:.4f} "
                   f"| {eng_peak[CLASS_NAMES[c-1]]:.1f} |\n" for c in range(1, 9))
    mi_tbl = "".join(f"| {n} | {mc:.4f} | {ml:.4f} | {'class' if mc > ml else 'PLACEMENT⚠'} |\n"
                     for n, mc, ml in zip(fnames, mi_class, mi_loc))
    md = (f"# Vibration diagnostic ({args.split}, {','.join(locs)}; engine band {ENGINE_BAND} Hz)\n\n"
          f"**H1 (engine band, Car/Bus vs Still/rail):** "
          f"{'SUPPORTED — proceed to engine-band features' if h1 else 'NOT supported — do not invest'} "
          f"(vehicle ratio {veh_ratio:.4f} vs reference {ref_ratio:.4f}; "
          f"MI_class {mi_class.mean():.4f} vs MI_placement {mi_loc.mean():.4f}).\n\n"
          f"**H2 (raw-axis mag rail):** raw |B| DC — Train {dc_tr:.2f} vs Subway {dc_sub:.2f} "
          f"(Δ={dc_tr-dc_sub:+.2f}); inspect mag_x/y/z PSD curves in the JSON for spectral "
          f"separation. (Acc vibration CANNOT separate rail — it is above the 50 Hz Nyquist.)\n\n"
          "## Engine-band ratio per class (body-acc; [18-40 Hz] / total)\n\n"
          "| class | engine-band ratio | peak freq in band (Hz) |\n|---|---|---|\n" + rows +
          "\n## MI of candidate features (leakage tripwire: want class > placement)\n\n"
          "| feature | MI vs class | MI vs placement | stronger w/ |\n|---|---|---|---|\n" + mi_tbl)
    args.out.write_text(md)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(
        {"engine_ratio": eng_ratio, "engine_peak_hz": eng_peak,
         "mi_class": dict(zip(fnames, mi_class.tolist())),
         "mi_placement": dict(zip(fnames, mi_loc.tolist())),
         "rail_dc_field": {"Train": dc_tr, "Subway": dc_sub},
         "h1_supported": bool(h1), "curves": curves}, indent=2))

    # --- optional figures (skipped cleanly if matplotlib absent — no uv.lock change) -----
    fig_paths = []
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for s in ("acc_body_mag", "mag_x"):
            plt.figure(figsize=(8, 5))
            for c in range(1, 9):
                plt.semilogy(freqs, med(s, c), label=CLASS_NAMES[c-1])
            if s == "acc_body_mag":
                plt.axvspan(*ENGINE_BAND, color="grey", alpha=0.15, label="engine band")
            plt.xlabel("Hz"); plt.ylabel("PSD"); plt.title(f"median PSD per class — {s}")
            plt.legend(fontsize=7); p = args.out.with_name(f"psd_{s}.png"); plt.savefig(p, dpi=120)
            plt.close(); fig_paths.append(p)
    except Exception as e:                                # matplotlib missing or backend issue
        print(f"[psd] figures skipped ({type(e).__name__}: {e}); curves are in {json_path.name}", flush=True)

    # --- MLflow: track + snapshot artifacts (rule §8) -----------------------------------
    from shl2026 import track
    with track("mdaniol", run_name="vibration_psd_diagnostic", seed=0, params_path=None,
               params={"split": args.split, "locs": ",".join(locs),
                       "max_per_class_loc": args.max_per_class_loc,
                       "engine_band": str(ENGINE_BAND), "n_windows": int(len(F))},
               tags={"phase": "diagnostic", "branch": "vibration",
                     "h1_verdict": "SUPPORTED" if h1 else "NOT_SUPPORTED"}) as run:
        run.log_metrics({"engine_ratio_vehicle": veh_ratio, "engine_ratio_reference": ref_ratio,
                         "mi_class_mean": float(mi_class.mean()),
                         "mi_placement_mean": float(mi_loc.mean()),
                         "rail_dc_delta": dc_tr - dc_sub})
        run.log_artifact(args.out); run.log_artifact(json_path)
        for p in fig_paths:
            run.log_artifact(p)

    print(f"\n[psd] H1 {'SUPPORTED' if h1 else 'NOT supported'} | "
          f"engine ratio veh {veh_ratio:.4f} vs ref {ref_ratio:.4f} | "
          f"wrote {args.out} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
