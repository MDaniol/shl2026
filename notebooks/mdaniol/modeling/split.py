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


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "val_split.npy")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    assign_all, label_all, loc_all = [], [], []
    for loc in LOCATIONS:
        y = pd.read_parquet(args.feat_dir / "validation" / f"{loc}.parquet",
                            columns=["label"])["label"].to_numpy()
        n = len(y)
        phase = np.full(n, FIT, dtype=np.int8)
        # PER-CLASS contiguous-block interleave so every class appears in every
        # slice (Run is rare + temporally clustered, so global blocking dropped it).
        for c in range(1, 9):
            idx = np.where(y == c)[0]               # this class's rows, time-ordered
            nc = len(idx)
            if nc == 0:
                continue
            block_len = max(1, nc // 10)            # ~10 blocks per class
            bid = np.arange(nc) // block_len
            ph = np.where(bid % 5 < 3, FIT, np.where(bid % 5 == 3, TUNE, TEST))
            # guarantee tune+test get >=1 sample for tiny classes
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
    print(f"validation rows: {len(assign)}  -> saved {args.out}")
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
    json.dump({"n": int(len(assign)),
               "fit": int((assign == FIT).sum()),
               "tune": int((assign == TUNE).sum()),
               "test": int((assign == TEST).sum())},
              open(args.out.with_suffix(".json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
