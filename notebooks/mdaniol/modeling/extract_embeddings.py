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


def embed_per_channel(X, embed_fn):
    """Embed each channel independently and stack -> (n, C, d).

    Mantis/MOMENT are channel-independent: feeding ONE channel as a univariate series
    yields that channel's embedding, and the mean over channels reproduces the pooled
    embedding the default path returns (so this is a strict generalization). Keeping the
    channels SEPARATE is the input the cross-channel head needs — it recovers cross-axis
    coupling a channel-independent FM provably discards (LIGHTWEIGHT_HEAD_PLAN H-chan).
    embed_fn: (n,1,T) -> (n, d)."""
    chans = [embed_fn(X[:, c:c + 1, :]) for c in range(X.shape[1])]
    return np.stack(chans, axis=1).astype(np.float32)            # (n, C, d)


def imu_log_spectrogram(x, n_mels: int, n_frames: int, fs: int = 100):
    """(n_signals, T) raw IMU -> (n_signals, n_frames, n_mels) log-power spectrogram on the AST
    input grid. STFT over the IMU band (0-50 Hz @100 Hz — NOT the 16 kHz audio mel front-end, which
    would squash our signal into one bin); log1p power; bilinear-resize to AST's (n_frames, n_mels);
    per-spectrogram standardize to mean 0 / std 0.5 (AST's documented input contract). Library-first:
    scipy STFT + torch resize. Returns a torch.FloatTensor."""
    from scipy.signal import spectrogram as _spec
    _, _, Sxx = _spec(np.asarray(x, dtype=np.float64), fs=fs, nperseg=64, noverlap=48, axis=-1)
    S = torch.from_numpy(np.log1p(Sxx).astype(np.float32))[:, None]      # (n,1,F,Tt)
    S = torch.nn.functional.interpolate(S, size=(n_mels, n_frames), mode="bilinear",
                                        align_corners=False)[:, 0]        # (n, n_mels, n_frames)
    S = S.transpose(1, 2)                                                 # (n, n_frames, n_mels) AST layout
    mu = S.mean(dim=(1, 2), keepdim=True); sd = S.std(dim=(1, 2), keepdim=True) + 1e-6
    return ((S - mu) / sd) * 0.5                                          # AST contract: mean 0, std 0.5


def imu_spectrogram_image(x, size: int, mean, std, fs: int = 100):
    """(n_signals, T) raw IMU -> (n_signals, 3, size, size) normalized RGB-replicated spectrogram
    IMAGE for a frozen vision ViT (DINOv2/CLIP/SigLIP). STFT over the IMU band -> log1p -> per-image
    min-max to [0,1] -> bilinear resize to size x size -> replicate to 3 channels -> apply the ViT's
    (mean,std) normalization. Library-first: scipy STFT + torch resize. Returns a torch.FloatTensor."""
    from scipy.signal import spectrogram as _spec
    _, _, Sxx = _spec(np.asarray(x, dtype=np.float64), fs=fs, nperseg=64, noverlap=48, axis=-1)
    S = torch.from_numpy(np.log1p(Sxx).astype(np.float32))[:, None]      # (n,1,F,Tt)
    S = torch.nn.functional.interpolate(S, size=(size, size), mode="bilinear",
                                        align_corners=False)              # (n,1,size,size)
    lo = S.amin(dim=(2, 3), keepdim=True); hi = S.amax(dim=(2, 3), keepdim=True)
    S = (S - lo) / (hi - lo + 1e-6)                                       # per-image [0,1]
    S = S.repeat(1, 3, 1, 1)                                              # RGB-replicate
    m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    sd = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (S - m) / sd


# vision ViTs usable as frozen image encoders on IMU spectrogram-images (TiViT, arXiv:2506.08641).
# Register-free DINOv2 — the with-registers checkpoints need transformers>=4.48 (we pin 4.44.2); the
# n_prefix logic below handles both (register-free -> drop CLS only). TiViT's intermediate-layer
# patch-pooling result doesn't depend on registers.
VIT_IDS = {"dinov2": "facebook/dinov2-base",
           "dinov2-large": "facebook/dinov2-large"}
# models whose embed() batches all (b*C) channel-spectrograms in ONE forward and can return per-channel
# (n,C,d) natively — so main must NOT wrap them in the slow embed_per_channel C-loop (5x small GPU calls).
NATIVE_PC_MODELS = {"ast"} | set(VIT_IDS)


def build_embedder(model: str, device: str, tf_batch: int, vit_layer_frac: float = 0.65,
                   per_channel: bool = False):
    if model in VIT_IDS:
        # frozen vision ViT over per-channel IMU spectrogram-IMAGES; use an INTERMEDIATE layer
        # (TiViT: ~40-70% depth beats the final block); mean-pool tokens, then mean over channels.
        from transformers import AutoModel, AutoImageProcessor
        vid = VIT_IDS[model]
        proc = AutoImageProcessor.from_pretrained(vid)
        net = AutoModel.from_pretrained(vid, output_hidden_states=True).to(device).eval()
        size = int(proc.crop_size["height"]) if getattr(proc, "crop_size", None) else 224
        mean, std = proc.image_mean, proc.image_std
        L = max(1, int(round(vit_layer_frac * net.config.num_hidden_layers)))
        # drop the non-spatial prefix tokens (CLS + register tokens) before mean-pooling — the
        # registers are high-norm internal-computation tokens (arXiv:2309.16588) and pooling them
        # in injects noise; TiViT pools PATCH tokens only. n_prefix = 1 (CLS) + num_register_tokens.
        n_prefix = 1 + int(getattr(net.config, "num_register_tokens", 0) or 0)

        def embed(x):                                                    # x: (n, C, 500) raw channels
            outs = []
            with torch.no_grad():
                for i in range(0, len(x), tf_batch):
                    xb = x[i:i + tf_batch]; b, C, T = xb.shape
                    img = imu_spectrogram_image(xb.reshape(b * C, T), size, mean, std).to(device)
                    hs = net(pixel_values=img).hidden_states[L]          # (b*C, tokens, d)
                    pooled = hs[:, n_prefix:].mean(1).reshape(b, C, -1)  # patch-token pool -> (b, C, d)
                    emb = pooled if per_channel else pooled.mean(1)      # keep channels, or mean-pool
                    outs.append(emb.cpu().numpy()); del xb, img, hs, pooled, emb
            empty_cache(device)
            out = np.concatenate(outs, 0).astype(np.float32)
            np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return out
        return embed, fi.pack_ast

    if model == "imagebind":
        # Meta ImageBind's NATIVE frozen IMU encoder (arXiv:2305.05665) -> 1024-d joint-space vector.
        # Trained on Ego4D head-mounted Aria IMU -> domain gap to phone; value = decorrelated errors.
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        net = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()

        def embed(x):                                                   # x: (n, 6, 2000) acc+gyr, mean-subtracted
            outs = []
            with torch.no_grad():
                for i in range(0, len(x), tf_batch):
                    xb = torch.from_numpy(np.ascontiguousarray(x[i:i + tf_batch])).float().to(device)
                    e = net({ModalityType.IMU: xb})[ModalityType.IMU]   # (b, 1024) L2-normalized
                    outs.append(e.cpu().numpy()); del xb, e
            empty_cache(device)
            out = np.concatenate(outs, 0).astype(np.float32)
            np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return out
        return embed, fi.pack_imagebind

    if model == "ast":
        # frozen Audio Spectrogram Transformer over per-channel IMU log-spectrograms -> 768-d,
        # mean-pooled over channels. A representation ORTHOGONAL to the temporal FMs (diversity voter).
        from transformers import ASTModel
        m = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device).eval()
        n_mels, n_frames = m.config.num_mel_bins, m.config.max_length

        def embed(x):                                                    # x: (n, C, 500) raw channels
            outs = []
            with torch.no_grad():
                for i in range(0, len(x), tf_batch):
                    xb = x[i:i + tf_batch]
                    b, C, T = xb.shape
                    spec = imu_log_spectrogram(xb.reshape(b * C, T), n_mels, n_frames).to(device)
                    e = m(input_values=spec).pooler_output               # (b*C, 768)
                    pe = e.reshape(b, C, -1)                             # (b, C, 768)
                    outs.append((pe if per_channel else pe.mean(1)).cpu().numpy())
                    del xb, spec, e, pe
            empty_cache(device)
            out = np.concatenate(outs, 0).astype(np.float32)
            np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return out
        return embed, fi.pack_ast

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
    ap.add_argument("--per-channel", action="store_true",
                    help="keep channels SEPARATE: embed each channel as univariate -> (n, C, d) "
                         "instead of the pooled/flattened (n, d). Output dir gets a _pc tag. This "
                         "is the input for the cross-channel head (LIGHTWEIGHT_HEAD_PLAN H-chan); "
                         "use with an axis variant (V1), not magnitudes (V2).")
    ap.add_argument("--max-tilt-deg", type=float, default=30.0, help="gravity_aware tilt bound")
    ap.add_argument("--vit-layer-frac", type=float, default=0.65,
                    help="for vision-ViT models (dinov2*): which hidden layer to read, as a fraction "
                         "of depth — TiViT (arXiv:2506.08641) finds 40-70%% beats the final block.")
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
    # pin torch/cuda determinism BEFORE any forward pass (track() seeds too late — it opens at the
    # END of main, after extraction). GUARDRAILS §4 "determinism or it didn't happen".
    from shl2026.tracking.run_context import set_global_seeds
    set_global_seeds(args.seed)

    if args.prefetch:
        dev = "cpu"   # download only; avoid CUDA init so it runs anywhere
        print(f"[prefetch] downloading {args.model} weights -> HF cache ...", flush=True)
        build_embedder(args.model, dev, args.tf_batch, args.vit_layer_frac, args.per_channel)
        print(f"[prefetch] {args.model} cached. Safe to run the parallel array now.", flush=True)
        return 0

    if args.tta_k and args.rotation == "none":
        ap.error("--tta-k requires --rotation so3|gravity_aware")
    if args.per_channel and args.tta_k:
        ap.error("--per-channel and --tta-k are mutually exclusive (combine later if needed)")
    embed, packer = build_embedder(args.model, args.device, args.tf_batch, args.vit_layer_frac,
                                   args.per_channel)
    tag = (f"{args.model}_{args.variant}"
           + ("_pc" if args.per_channel else "")
           + (f"_tta{args.tta_k}{args.rotation}" if args.tta_k else ""))
    outdir = args.emb_dir / tag
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    want = set(args.splits.split(","))
    jobs = [(s, l) for s, l in JOBS if s in want]
    print(f"[embed] model={args.model} variant={args.variant} device={args.device} "
          f"chunk={args.chunk_size} tf_batch={args.tf_batch} per_channel={args.per_channel} "
          f"tta={args.tta_k or 'off'}/{args.rotation} splits={sorted(want)} -> {tag}", flush=True)

    manifest, total_win, emb_dim, n_chan, t_all = [], 0, None, None, time.time()
    for split, loc in jobs:
        src = args.data_dir / split / f"{loc}.parquet"
        outp = outdir / f"{split}__{loc}.npy"
        if outp.exists():                                # resume: COUNT the existing file (§8 manifest)
            arr = np.load(outp, mmap_mode="r")
            n_ex, d_ex = int(arr.shape[0]), int(arr.shape[-1])
            c_ex = int(arr.shape[1]) if arr.ndim == 3 else 0
            del arr
            manifest.append({"file": f"{tag}/{split}__{loc}.npy", "rows": n_ex, "dim": d_ex, "n_chan": c_ex})
            total_win += n_ex
            if emb_dim is None: emb_dim = d_ex
            if c_ex and n_chan is None: n_chan = c_ex
            print(f"  skip {outp.name} (exists, {n_ex} rows counted)", flush=True); continue
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
                # AST/DINOv2 batch all channels in one forward (NATIVE_PC) -> call embed directly;
                # only the generic (channel-collapsing) FMs need the slow per-channel C-loop wrapper.
                if args.per_channel and args.model not in NATIVE_PC_MODELS:
                    return embed_per_channel(X, embed)
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
                if args.per_channel:                         # e: (len, C, d) -> file (n, C, d)
                    n_chan, emb_dim = e.shape[1], e.shape[2]
                    mm = open_memmap(outp, mode="w+", dtype=np.float32,
                                     shape=(n_rows, n_chan, emb_dim))
                else:
                    emb_dim = e.shape[1]
                    mm = open_memmap(outp, mode="w+", dtype=np.float32, shape=(n_rows, emb_dim))
            mm[pos:pos + len(e)] = e
            pos += len(e)
            del acc, gyr, mag, e          # lib/X are local to embed_one() now
            gc.collect()
            print(f"    {split}/{loc} {pos}/{n_rows} ({pos/(time.time()-t0):.0f}/s)", flush=True)
        if mm is None:                                   # no batches produced (empty/edge input)
            print(f"  WARN {outp.name}: 0 rows written — nothing saved", flush=True); continue
        mm.flush(); del mm
        manifest.append({"file": f"{tag}/{split}__{loc}.npy", "rows": int(n_rows),
                         "dim": int(emb_dim or 0), "n_chan": int(n_chan or 0)})
        total_win += n_rows
        print(f"  saved {outp.name} ({n_rows} rows, {time.time()-t0:.0f}s)", flush=True)

    # MLflow: log this extraction (params + throughput + manifest) for traceability (rule §8).
    secs = time.time() - t_all
    import json
    from shl2026 import track
    mpath = outdir / "extract_manifest.json"
    mpath.write_text(json.dumps({"tag": tag, "model": args.model, "variant": args.variant,
                                 "per_channel": bool(args.per_channel), "n_chan": int(n_chan or 0),
                                 "rotation": args.rotation, "tta_k": args.tta_k,
                                 "n_windows": int(total_win), "emb_dim": int(emb_dim or 0),
                                 "files": manifest}, indent=2))
    with track("mdaniol", run_name=f"extract_{tag}", seed=args.seed, params_path=None,
               params={"model": args.model, "variant": args.variant, "device": args.device,
                       "per_channel": int(args.per_channel), "n_chan": int(n_chan or 0),
                       "rotation": args.rotation, "tta_k": args.tta_k, "chunk": args.chunk_size,
                       "tf_batch": args.tf_batch, "emb_dim": int(emb_dim or 0),
                       "n_windows": int(total_win)},
               tags={"phase": "extraction", "branch": "fm_embeddings"}) as run:
        run.log_metrics({"n_windows": float(total_win), "seconds": secs,
                         "windows_per_sec": total_win / max(secs, 1e-9)})
        run.log_artifact(mpath)
    print(f"[embed] done — {total_win} win in {secs:.0f}s "
          f"({total_win/max(secs,1e-9):.0f}/s); manifest {mpath} "
          f"(MLflow logged iff mlflow is installed + MLFLOW_TRACKING_URI set)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
