#!/usr/bin/env python3
"""Extract FROZEN foundation-model embeddings for every SHL window → cached .npy.

Memory-stable: streams each (split,location) parquet in small chunks straight to a
disk memmap (no RAM accumulation), float32, frees the GPU cache every chunk, and
for the V0 variant builds only the 9 raw channels (skips the 23-channel derived
library). Reuses validated `fm_input` packers for non-V0 variants.

Supported --model: mantisv2, utica, mantis8m, moment-small, moment-base, moment-large.
Output: <emb-dir>/<model>_<variant>/<split>__<loc>.npy  (float32, row-aligned).

    python extract_embeddings.py --model mantisv2 --variant V0
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "feature_extraction"))
import fm_input as fi          # noqa: E402
import shl_features as shl      # noqa: E402

N = shl.N
LOCATIONS = ("Bag", "Hips", "Torso", "Hand")
JOBS = ([("train", l) for l in LOCATIONS]
        + [("validation", l) for l in LOCATIONS]
        + [("test", "all")])


def empty_cache(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def stack_axes(batch, prefix: str) -> np.ndarray:
    cols = [batch.column(batch.schema.get_field_index(f"{prefix}_{a}")) for a in "xyz"]
    arrs = [c.flatten().to_numpy(zero_copy_only=False).reshape(-1, N) for c in cols]
    return np.stack(arrs, axis=1).astype(np.float32)        # (n, 3, N) float32


def raw_lib(acc, gyr, mag) -> dict:
    """Minimal channel dict for V0 (raw 9 channels) — avoids the 23-channel library."""
    lib = {}
    for i, a in enumerate("xyz"):
        lib[f"Acc_{a}"] = acc[:, i]; lib[f"Gyr_{a}"] = gyr[:, i]; lib[f"Mag_{a}"] = mag[:, i]
    return lib


def build_embedder(model: str, device: str, tf_batch: int):
    if model in ("mantisv2", "mantis8m", "utica"):
        from mantis.trainer import MantisTrainer
        if model == "mantisv2":
            from mantis.architecture import version2
            net = version2.MantisV2(device=device).from_pretrained("paris-noah/MantisV2")
        elif model == "mantis8m":
            from mantis.architecture import version1
            net = version1.Mantis8M(device=device).from_pretrained("paris-noah/Mantis-8M")
        else:  # utica = Mantis8M arch + UTICA weights
            from mantis.architecture import version1
            from huggingface_hub import hf_hub_download
            net = version1.Mantis8M(device=device)
            ckpt = hf_hub_download("fegounna/Utica", "pytorch_model.bin")
            net.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
            net = net.to(device)
        tr = MantisTrainer(device=device, network=net)

        def embed(x):
            z = np.asarray(tr.transform(x, batch_size=tf_batch, three_dim=True))
            z = z.reshape(z.shape[0], -1).astype(np.float32)
            np.nan_to_num(z, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            empty_cache(device)
            return z
        return embed, fi.pack_mantis

    if model.startswith("moment"):
        from momentfm import MOMENTPipeline
        size = model.split("-", 1)[1] if "-" in model else "small"
        m = MOMENTPipeline.from_pretrained(
            f"AutonLab/MOMENT-1-{size}", model_kwargs={"task_name": "embedding"})
        m.init(); m.to(device).eval()

        def embed(x):
            outs = []
            with torch.no_grad():
                for i in range(0, len(x), tf_batch):
                    xb = torch.tensor(x[i:i + tf_batch]).to(device)
                    outs.append(m(x_enc=xb).embeddings.detach().cpu().numpy())
                    del xb
            empty_cache(device)
            out = np.concatenate(outs, 0).astype(np.float32)
            np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return out
        return embed, fi.pack_moment

    raise ValueError(f"unknown model {model}")


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="V0")
    ap.add_argument("--data-dir", type=Path, default=root / "dataset_parquet")
    ap.add_argument("--emb-dir", type=Path, default=root / "embeddings")
    ap.add_argument("--chunk-size", type=int, default=4000)
    ap.add_argument("--tf-batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap windows per file for a smoke test (0 = all). Use a "
                         "throwaway --emb-dir so the partial output isn't cached.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()

    embed, packer = build_embedder(args.model, args.device, args.tf_batch)
    outdir = args.emb_dir / f"{args.model}_{args.variant}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[embed] model={args.model} variant={args.variant} device={args.device} "
          f"chunk={args.chunk_size} tf_batch={args.tf_batch}", flush=True)

    for split, loc in JOBS:
        src = args.data_dir / split / f"{loc}.parquet"
        outp = outdir / f"{split}__{loc}.npy"
        if outp.exists():
            print(f"  skip {outp.name} (exists)", flush=True); continue
        pf = pq.ParquetFile(src)
        n_rows = pf.metadata.num_rows
        if args.limit:
            n_rows = min(n_rows, args.limit)
        mm, pos, t0 = None, 0, time.time()
        for batch in pf.iter_batches(batch_size=args.chunk_size):
            if pos >= n_rows:
                break
            acc, gyr, mag = (stack_axes(batch, p) for p in ("Acc", "Gyr", "Mag"))
            if pos + len(acc) > n_rows:          # trim final batch to the limit
                k = n_rows - pos
                acc, gyr, mag = acc[:k], gyr[:k], mag[:k]
            lib = raw_lib(acc, gyr, mag) if args.variant == "V0" \
                else fi.build_channel_library(acc, gyr, mag)
            X, _ = packer(lib, args.variant)
            e = embed(X)
            if mm is None:
                mm = open_memmap(outp, mode="w+", dtype=np.float32, shape=(n_rows, e.shape[1]))
            mm[pos:pos + len(e)] = e
            pos += len(e)
            del acc, gyr, mag, lib, X, e
            gc.collect()
            print(f"    {split}/{loc} {pos}/{n_rows} ({pos/(time.time()-t0):.0f}/s)", flush=True)
        mm.flush(); del mm
        print(f"  saved {outp.name} ({n_rows} rows, {time.time()-t0:.0f}s)", flush=True)
    print("[embed] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
