#!/usr/bin/env python3
"""Mean-pool a per-channel embedding dir into a mean-pooled VOTER dir — number-identical to a
non-per-channel extraction, so no GPU / re-extraction is needed.

`extract_embeddings.py` builds `emb = pooled if per_channel else pooled.mean(1)`. Therefore the
channel-mean of the per-channel files (`<tag>_pc/<split>__<loc>.npy`, shape (n,C,d)) reproduces the
non-per-channel embedding (`<tag>/<split>__<loc>.npy`, shape (n,d)) EXACTLY. We make that explicit
here so `dinov2_V2_pc` (extracted only for the cross-channel head) can be reused as a fusion VOTER
(`probe_fusion.load_emb` / `submit_fusion.fuse` expect (n,d)) without burning GPU hours.

    python pool_per_channel.py --src <emb>/dinov2_V2_pc --dst <emb>/dinov2_V2

Guarded by tests/test_guardrails.py::test_pool_per_channel_equals_channel_mean.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def pool_dir(src: Path, dst: Path) -> int:
    """Channel-mean every <split>__<loc>.npy in src (n,C,d) -> dst (n,d). Returns file count."""
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy under {src}")
    for f in files:
        a = np.load(f)
        assert a.ndim == 3, f"{f.name}: expected per-channel (n,C,d), got {a.shape}"
        out = a.mean(axis=1).astype(a.dtype)          # channel-mean == non-per-channel extraction
        assert out.ndim == 2 and len(out) == len(a), f"{f.name}: bad pool {a.shape}->{out.shape}"
        np.save(dst / f.name, out)
        print(f"  {f.name}: {a.shape} -> {out.shape}", flush=True)
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="per-channel embedding dir (n,C,d)")
    ap.add_argument("--dst", type=Path, required=True, help="output mean-pooled voter dir (n,d)")
    args = ap.parse_args()
    k = pool_dir(args.src, args.dst)
    print(f"[pool] wrote {k} files -> {args.dst} "
          f"(channel-mean; number-identical to a non-per-channel extraction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
