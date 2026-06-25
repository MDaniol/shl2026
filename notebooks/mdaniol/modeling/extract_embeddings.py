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


def rotate_sensors(acc, gyr, mag, mode, max_tilt_deg, rng):
    """Apply ONE per-window 3D rotation jointly to acc/gyr/mag (shared device frame).
    Reuses the validated augment.rotation_matrices. Orthogonal -> per-sample magnitude
    preserved, so handcrafted magnitude features are invariant; only axis-based FM inputs
    change. Physically = 'what if the phone were mounted at a different orientation'."""
    import augment as aug
    M = aug.rotation_matrices(acc, mode, max_tilt_deg, rng)        # (n,3,3)
    return aug._apply_R(M, acc), aug._apply_R(M, gyr), aug._apply_R(M, mag)


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
    # --- rotation TTA (test-time aug): mean embedding over K random reorientations -------
    ap.add_argument("--rotation", choices=["none", "so3", "gravity_aware"], default="none",
                    help="rotate raw axes before the FM (none=current behavior). Helps the "
                         "axis-based FM input under the unknown test orientation; no-op for "
                         "magnitude features.")
    ap.add_argument("--tta-k", type=int, default=0,
                    help="if >0 with --rotation, write the MEAN embedding over K rotated copies "
                         "(rotation-marginalized representation). Output dir gets a _tta tag so "
                         "originals are not clobbered; the existing probe/submit consume it via --emb.")
    ap.add_argument("--max-tilt-deg", type=float, default=30.0, help="gravity_aware tilt bound")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="train,validation,test",
                    help="which splits to extract (comma list) — e.g. 'validation,test' for "
                         "eval-time-only TTA to save the costly train re-extraction.")
    ap.add_argument("--prefetch", action="store_true",
                    help="only download + cache the model weights (HF_HOME) then exit. "
                         "Run serially per model BEFORE a parallel array to avoid the "
                         "concurrent-download race. Works on CPU (no GPU needed).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()

    if args.prefetch:
        dev = "cpu"   # download only; avoid CUDA init so it runs anywhere
        print(f"[prefetch] downloading {args.model} weights -> HF cache ...", flush=True)
        build_embedder(args.model, dev, args.tf_batch)
        print(f"[prefetch] {args.model} cached. Safe to run the parallel array now.", flush=True)
        return 0

    if args.tta_k and args.rotation == "none":
        ap.error("--tta-k requires --rotation so3|gravity_aware")
    embed, packer = build_embedder(args.model, args.device, args.tf_batch)
    tag = f"{args.model}_{args.variant}" + (f"_tta{args.tta_k}{args.rotation}" if args.tta_k else "")
    outdir = args.emb_dir / tag
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    want = set(args.splits.split(","))
    jobs = [(s, l) for s, l in JOBS if s in want]
    print(f"[embed] model={args.model} variant={args.variant} device={args.device} "
          f"chunk={args.chunk_size} tf_batch={args.tf_batch} "
          f"tta={args.tta_k or 'off'}/{args.rotation} splits={sorted(want)} -> {tag}", flush=True)

    manifest, total_win, emb_dim, t_all = [], 0, None, time.time()
    for split, loc in jobs:
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
            def embed_one(a, g, m):
                lib = raw_lib(a, g, m) if args.variant == "V0" \
                    else fi.build_channel_library(a, g, m)
                X, _ = packer(lib, args.variant)
                return embed(X)

            if args.tta_k:                       # mean embedding over K random reorientations
                stack = None
                for _k in range(args.tta_k):
                    ra, rg, rm = rotate_sensors(acc, gyr, mag, args.rotation, args.max_tilt_deg, rng)
                    ek = embed_one(ra, rg, rm)
                    stack = ek if stack is None else stack + ek
                e = (stack / args.tta_k).astype(np.float32)
            else:
                e = embed_one(acc, gyr, mag)
            if mm is None:
                emb_dim = e.shape[1]
                mm = open_memmap(outp, mode="w+", dtype=np.float32, shape=(n_rows, emb_dim))
            mm[pos:pos + len(e)] = e
            pos += len(e)
            del acc, gyr, mag, e          # lib/X are local to embed_one() now
            gc.collect()
            print(f"    {split}/{loc} {pos}/{n_rows} ({pos/(time.time()-t0):.0f}/s)", flush=True)
        mm.flush(); del mm
        manifest.append({"file": f"{tag}/{split}__{loc}.npy", "rows": int(n_rows), "dim": int(emb_dim or 0)})
        total_win += n_rows
        print(f"  saved {outp.name} ({n_rows} rows, {time.time()-t0:.0f}s)", flush=True)

    # MLflow: log this extraction (params + throughput + manifest) for traceability (rule §8).
    secs = time.time() - t_all
    import json
    from shl2026 import track
    mpath = outdir / "extract_manifest.json"
    mpath.write_text(json.dumps({"tag": tag, "model": args.model, "variant": args.variant,
                                 "rotation": args.rotation, "tta_k": args.tta_k,
                                 "n_windows": int(total_win), "emb_dim": int(emb_dim or 0),
                                 "files": manifest}, indent=2))
    with track("mdaniol", run_name=f"extract_{tag}", seed=args.seed, params_path=None,
               params={"model": args.model, "variant": args.variant, "device": args.device,
                       "rotation": args.rotation, "tta_k": args.tta_k, "chunk": args.chunk_size,
                       "tf_batch": args.tf_batch, "emb_dim": int(emb_dim or 0),
                       "n_windows": int(total_win)},
               tags={"phase": "extraction", "branch": "fm_embeddings"}) as run:
        run.log_metrics({"n_windows": float(total_win), "seconds": secs,
                         "windows_per_sec": total_win / max(secs, 1e-9)})
        run.log_artifact(mpath)
    print(f"[embed] done — {total_win} win in {secs:.0f}s "
          f"({total_win/max(secs,1e-9):.0f}/s); MLflow logged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
