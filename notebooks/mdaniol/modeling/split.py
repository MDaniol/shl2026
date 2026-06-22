#!/usr/bin/env python3
"""Blocked 60/20/20 split of the VALIDATION set (Users 2&3) for two-phase eval.

The challenge lets us build the model on train+validation. Validation is Users
2&3 — the SAME users as the (hidden) test set — so we feed part of it into
training and hold out a slice for an honest internal macro-F1.

  fit  (60%)  -> trained on, together with ALL of User-1 train
  tune (20%)  -> early-stopping + per-class threshold calibration
  test (20%)  -> held out, never touched in fit/tune (unbiased estimate)

Leakage control: validation frames are consecutive in time (like train), so a
random split would put adjacent windows of the same activity bout into different
slices. We assign **contiguous time-blocks** (interleaved per location so every
class is represented in each slice). The held-out test slice is restricted to the
**test locations (Bag/Hips/Torso)** to mirror the real test set (no Hand).

Output: `val_split.npy` — int8 array (0=fit,1=tune,2=test), row-aligned to the
LOCATIONS-order concatenation used everywhere else.

NOTE: this is for DEVELOPMENT/estimation. The FINAL submission model retrains on
User-1 + ALL validation (no held-out wasted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOCATIONS = ("Bag", "Hips", "Torso", "Hand")
TEST_LOCS = {"Bag", "Hips", "Torso"}
CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]
FIT, TUNE, TEST = 0, 1, 2
UNUSED = -1            # embargo gap rows (excluded from every slice)
N_BLOCKS = 30          # ~30 contiguous blocks per location, cycled 60/20/20


def load_split_with_location_map(split_path, feat_dir):
    """Load val_split.npy and recover per-location row ranges.

    val_split.npy is row-aligned to the LOCATIONS-order concatenation of the
    validation feature files (the same order load_feats/load_emb use), so we
    rebuild offsets from each location's row count.

    Returns (assign, loc_offsets) where assign is the int8 FIT/TUNE/TEST array
    and loc_offsets = {loc: (start, end)} indexes into it.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    assign = np.load(split_path)
    loc_offsets, offset = {}, 0
    for loc in LOCATIONS:
        n = len(pd.read_parquet(Path(feat_dir) / "validation" / f"{loc}.parquet",
                                columns=["label"]))
        loc_offsets[loc] = (offset, offset + n)
        offset += n
    assert offset == len(assign), f"loc rows {offset} != split {len(assign)}"
    return assign, loc_offsets


def temporal_phase(y: np.ndarray, embargo: int) -> np.ndarray:
    """Conservative, class-complete time-cut for one location (rows in recording
    order). PER CLASS: that class's earliest 60% -> FIT, next 20% -> TUNE, latest
    20% -> TEST, with a (class-size-capped) embargo gap at each boundary. So FIT and
    TEST are temporally separated (early vs late occurrences -> no shared journey,
    unlike interleaved blocks) AND every class stays present in every slice (a GLOBAL
    cut drops session-clustered classes like Subway -> F1=0). Per class:
    max(FIT idx) < min(TEST idx)."""
    n = len(y)
    phase = np.full(n, FIT, dtype=np.int8)
    for c in range(1, 9):
        idx = np.where(y == c)[0]                      # this class's rows, time-ordered
        nc = len(idx)
        if nc < 5:                                     # too few to split; leave in FIT
            continue
        e = min(embargo, max(0, nc // 20))             # cap embargo at ~5% of the class
        cut1, cut2 = int(0.60 * nc), int(0.80 * nc)
        ph = np.full(nc, FIT, dtype=np.int8)
        ph[cut1:cut1 + e] = UNUSED
        ph[cut1 + e:cut2] = TUNE
        ph[cut2:cut2 + e] = UNUSED
        ph[cut2 + e:] = TEST
        phase[idx] = ph
    return phase


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "val_split.npy")
    ap.add_argument("--scheme", choices=["blocked", "temporal"], default="blocked",
                    help="blocked = per-class interleaved blocks (optimistic — FIT/TEST share "
                         "sessions); temporal = global time-cut + embargo (conservative, mirrors "
                         "the held-out-period real test). Use temporal for the trustworthy estimate.")
    ap.add_argument("--embargo", type=int, default=100,
                    help="temporal only: windows dropped at each FIT|TUNE|TEST boundary so "
                         "adjacent same-journey windows can't straddle it (López de Prado / "
                         "TimeSeriesSplit gap). 100 windows = ~8 min.")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    assign_all, label_all, loc_all = [], [], []
    for loc in LOCATIONS:
        y = pd.read_parquet(args.feat_dir / "validation" / f"{loc}.parquet",
                            columns=["label"])["label"].to_numpy()
        n = len(y)
        phase = np.full(n, FIT, dtype=np.int8)
        if args.scheme == "temporal":
            phase = temporal_phase(y, args.embargo)
        else:
            # PER-CLASS contiguous-block interleave so every class appears in every slice
            # (Run is rare + temporally clustered). NOTE: optimistic — FIT/TEST blocks are
            # interleaved in time, so they can share a recording session (see GUARDRAILS L3.2).
            for c in range(1, 9):
                idx = np.where(y == c)[0]               # this class's rows, time-ordered
                nc = len(idx)
                if nc == 0:
                    continue
                block_len = max(1, nc // 10)            # ~10 blocks per class
                bid = np.arange(nc) // block_len
                ph = np.where(bid % 5 < 3, FIT, np.where(bid % 5 == 3, TUNE, TEST))
                if nc >= 3 and not (ph == TUNE).any():
                    ph[-2] = TUNE
                if nc >= 3 and not (ph == TEST).any():
                    ph[-1] = TEST
                phase[idx] = ph
        # held-out test only on test locations; Hand 'test' rows -> fit (not wasted)
        if loc not in TEST_LOCS:
            phase[phase == TEST] = FIT
        assign_all.append(phase)
        label_all.append(y); loc_all.append(np.full(n, loc))

    assign = np.concatenate(assign_all)
    labels = np.concatenate(label_all)
    locs = np.concatenate(loc_all)
    np.save(args.out, assign)

    # report
    n_embargo = int((assign == UNUSED).sum())
    print(f"validation rows: {len(assign)}  scheme={args.scheme}"
          f"{f' embargo={args.embargo} (dropped {n_embargo})' if args.scheme=='temporal' else ''}"
          f"  -> saved {args.out}")
    for name, code in (("fit", FIT), ("tune", TUNE), ("test(held-out)", TEST)):
        m = assign == code
        frac = m.mean()
        cls = {CLASS_NAMES[c-1]: int((labels[m] == c).sum()) for c in range(1, 9)}
        locd = {l: int((locs[m] == l).sum()) for l in LOCATIONS}
        print(f"  {name:15s} n={m.sum():6d} ({frac:.2f})  locs={locd}")
        print(f"      class counts: {cls}")
    # sanity: every class present in every slice
    for code, nm in ((FIT, "fit"), (TUNE, "tune"), (TEST, "test")):
        present = set(np.unique(labels[assign == code]))
        miss = set(range(1, 9)) - present
        if miss:
            print(f"  WARNING: {nm} missing classes {sorted(miss)}")
    json.dump({"n": int(len(assign)), "scheme": args.scheme,
               "embargo": args.embargo if args.scheme == "temporal" else 0,
               "fit": int((assign == FIT).sum()),
               "tune": int((assign == TUNE).sum()),
               "test": int((assign == TEST).sum()),
               "unused_embargo": n_embargo},
              open(args.out.with_suffix(".json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
