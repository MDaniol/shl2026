#!/usr/bin/env python3
"""Safely combine SHL-2026 predictions from two INDEPENDENT pipelines (ours + a collaborator's).

The only way a cross-team ensemble silently fails is ROW MISALIGNMENT on the hidden test — if our
row i and their row i are different windows, the blended submission is garbage. This tool makes
alignment *verifiable* and refuses to combine until it is verified.

Representation-robust alignment fingerprint
-------------------------------------------
Both sides derive a per-window signature from the RAW sensor signal (BEFORE any unit conversion /
standardization), so it is invariant to: float32-vs-float64 reads of the same official files, and
to each pipeline's preprocessing. Signature per window = a few coarse stats of the raw Acc/Gyr/Mag
x-axes; compared with np.allclose (tolerance) — robust to float dtype, unique enough that two
distinct 500-sample windows won't collide. Order is guaranteed by construction (both read the
official files unshuffled); the signature is the *proof* it held.

Modes
-----
  --emit-sig SPLIT      write our window signatures for SPLIT (test|validation) -> <out>.
  --verify A.npy B.npy  np.allclose the two signature files; report first mismatch. PASS == aligned.
  --combine OURS THEIRS write the blended 92726x500 submission from two (N,8) calibrated proba
                        matrices (class axis = labels 1..8 in order); REQUIRES --ours-sig/--theirs-sig
                        to verify alignment first unless --no-verify is given (not recommended).

Reuses: probe_fusion.{split_locs, CLASSES}, shl2026.write_submission.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import split_locs, CLASSES  # noqa: E402

N_TEST = 92726
SIG_CHANNELS = ("Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Mag_x")   # raw axes used for the alignment key


def window_signature(arr_by_channel: dict) -> np.ndarray:
    """{channel: (N,500) raw float} -> (N, 4*len(channels)) coarse per-window signature.
    Stats: first sample, middle sample, mean, std — float-dtype-robust (compare with allclose)."""
    feats = []
    for ch in SIG_CHANNELS:
        a = np.asarray(arr_by_channel[ch], dtype=np.float64)        # (N,500)
        feats += [a[:, 0], a[:, a.shape[1] // 2], a.mean(1), a.std(1)]
    return np.stack(feats, axis=1).astype(np.float64)              # (N, 4*K)


def load_raw_signature(data_dir: Path, split: str) -> np.ndarray:
    """Compute window signatures from the RAW dataset_parquet for SPLIT (concatenated over its
    location files in the canonical order — identical ordering to our embeddings/features)."""
    sigs = []
    for loc in split_locs(split):
        df = pd.read_parquet(data_dir / split / f"{loc}.parquet", columns=list(SIG_CHANNELS))
        by_ch = {ch: np.stack(df[ch].to_numpy()) for ch in SIG_CHANNELS}   # each (n,500)
        sigs.append(window_signature(by_ch))
    return np.concatenate(sigs, 0)


def aligned(a: np.ndarray, b: np.ndarray, atol: float = 1e-2) -> tuple[bool, int]:
    """True iff signatures match row-for-row (allclose). Returns (ok, first_mismatch_index|-1)."""
    if a.shape != b.shape:
        return False, 0
    row_ok = np.all(np.isclose(a, b, atol=atol, rtol=0), axis=1)
    return bool(row_ok.all()), (-1 if row_ok.all() else int(np.argmax(~row_ok)))


def combine_probas(ours: np.ndarray, theirs: np.ndarray, w: float) -> np.ndarray:
    """Weighted soft-vote of two (N,8) calibrated probability matrices (class axis 1..8).
    Row-normalizes each first so different calibrations are comparable. w = weight on OURS."""
    assert ours.shape == theirs.shape and ours.shape[1] == len(CLASSES), \
        f"shape mismatch: ours {ours.shape} theirs {theirs.shape} (need (N,{len(CLASSES)}))"
    o = ours / (ours.sum(1, keepdims=True) + 1e-12)
    t = theirs / (theirs.sum(1, keepdims=True) + 1e-12)
    return w * o + (1.0 - w) * t


def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--data-dir", type=Path, default=root / "dataset_parquet")
    ap.add_argument("--emit-sig", choices=["test", "validation"])
    ap.add_argument("--verify", nargs=2, metavar=("A.npy", "B.npy"))
    ap.add_argument("--combine", nargs=2, metavar=("OURS.npy", "THEIRS.npy"))
    ap.add_argument("--ours-sig", type=Path); ap.add_argument("--theirs-sig", type=Path)
    ap.add_argument("--w", type=float, default=0.5, help="weight on OURS in the blend")
    ap.add_argument("--no-verify", action="store_true", help="skip alignment check (NOT recommended)")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol" / "AGH_predictions_combined.txt")
    args = ap.parse_args()

    if args.emit_sig:
        sig = load_raw_signature(args.data_dir, args.emit_sig)
        outp = root / "notebooks/mdaniol" / f"sig_{args.emit_sig}.npy"
        np.save(outp, sig)
        print(f"[sig] {args.emit_sig}: {sig.shape} -> {outp}  (share this; collaborator emits theirs "
              f"from RAW Acc/Gyr/Mag, pre-unit-conversion, same official order)")
        return 0

    if args.verify:
        a, b = np.load(args.verify[0]), np.load(args.verify[1])
        ok, idx = aligned(a, b)
        print(f"[verify] shapes {a.shape} vs {b.shape} -> "
              f"{'ALIGNED ✓ (row order matches)' if ok else f'MISALIGNED ✗ first mismatch at row {idx}'}")
        return 0 if ok else 1

    if args.combine:
        from shl2026 import write_submission
        ours, theirs = np.load(args.combine[0]), np.load(args.combine[1])
        if not args.no_verify:
            assert args.ours_sig and args.theirs_sig, \
                "alignment unverified: pass --ours-sig and --theirs-sig (or --no-verify to override)"
            ok, idx = aligned(np.load(args.ours_sig), np.load(args.theirs_sig))
            if not ok:
                print(f"[combine] ABORT — signatures MISALIGNED (first mismatch row {idx}). "
                      f"Do NOT combine.", file=sys.stderr); return 1
            print("[combine] alignment verified ✓")
        assert len(ours) == N_TEST, f"expected {N_TEST} test rows, got {len(ours)}"
        blended = combine_probas(ours, theirs, args.w)
        pred = np.asarray(CLASSES)[blended.argmax(1)].astype(int)
        uniq, cnt = np.unique(pred, return_counts=True)
        print(f"[combine] w(ours)={args.w} dist={ {int(c): int(n) for c, n in zip(uniq, cnt)} }")
        rep = write_submission(pred, args.out)
        print(f"[combine] wrote {args.out} ok={rep.ok} rows={rep.n_rows} cols={rep.n_cols}")
        return 0 if rep.ok else 1

    ap.error("choose one of --emit-sig / --verify / --combine")


def self_test() -> None:
    rng = np.random.default_rng(0)
    n = 200
    by = {ch: rng.standard_normal((n, 500)) for ch in SIG_CHANNELS}
    s64 = window_signature(by)
    s32 = window_signature({c: v.astype(np.float32) for c, v in by.items()})   # float32 read
    ok, idx = aligned(s64, s32)
    assert ok, f"float32/float64 signature must match, mismatch at {idx}"
    # a permuted (misaligned) order must be DETECTED
    perm = rng.permutation(n); perm[perm == np.arange(n)] = (perm[perm == np.arange(n)] + 1) % n
    ok2, _ = aligned(s64, s64[perm])
    assert not ok2, "misalignment not detected"
    # combine: row-normalized weighted blend; identical inputs -> argmax preserved
    P = rng.random((n, len(CLASSES)))
    b = combine_probas(P, P, 0.5)
    assert np.allclose(b.sum(1), 1.0, atol=1e-9) and np.array_equal(b.argmax(1), P.argmax(1))
    print("combine_external self_test OK — signature float-robust + detects misalignment; blend normalized")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        raise SystemExit(main())
