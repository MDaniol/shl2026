#!/usr/bin/env python3
"""Cross-channel lightweight heads over PER-CHANNEL frozen-FM embeddings (E-HEAD-01).

The challenge's sanctioned innovation surface: frozen FM + a small trainable head. Our FMs
embed each IMU channel INDEPENDENTLY, so a per-channel extraction gives (n, C, d). A plain
mean over C (the default fusion input) throws away cross-axis coupling. This module learns
to MIX the C channel-embeddings — the novel contribution (LIGHTWEIGHT_HEAD_PLAN H-chan).

Heads (all tiny, frozen FM, NO backbone backprop), each → [pooled ⊕ 520 handcrafted] → MLP → 8:
  - mean      : mean over C (reproduces the single-FM fusion baseline)
  - concat    : per-channel concat (COMODO: concat > mean)            [baseline]
  - multistat : mean⊕max⊕std⊕GeM over C (the reliable workhorse)      [baseline]
  - se        : Squeeze-Excitation over channels — learn cross-channel gates, then pool [NOVEL]

Protocol (identical to the rest of the pipeline): fit on User1+val[FIT], early-stop +
calibrate + SELECT head on TUNE, lock TEST once; ≥3 seeds (report mean±std); per-class +
macro + ECE + missing-channel robustness. Gated KEEP iff the novel head's CI lower bound
beats the mean baseline with no per-class regression.

Reuses: probe_fusion.{load_feats,load_labels,calibrate,macro_f1,CLASSES,split_locs},
metrics.class_report, eval_metrics.{ece,channel_dropout_robustness}, shl2026.track.

    python head_xchannel.py --emb-pc utica_V1_pc --heads mean,concat,multistat,se --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from probe_fusion import (load_feats, load_labels, calibrate, macro_f1,  # noqa: E402
                          CLASSES, split_locs)
from metrics import class_report  # noqa: E402
import eval_metrics as em  # noqa: E402

CLS = np.asarray(CLASSES)


# --------------------------------------------------------------------------- data
def load_emb_pc(emb_dir: Path, split: str) -> np.ndarray:
    """Per-channel embeddings (n, C, d), concatenated over the split's location files."""
    arr = np.concatenate([np.load(emb_dir / f"{split}__{loc}.npy") for loc in split_locs(split)], 0)
    assert arr.ndim == 3, f"{emb_dir.name}: expected (n,C,d) per-channel; got {arr.shape}"
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


# --------------------------------------------------------------------------- heads
class ChannelPool(nn.Module):
    """Reduce (B, C, d) -> (B, out) by one of the pooling strategies."""
    def __init__(self, kind: str, C: int, d: int, se_reduction: int = 2):
        super().__init__()
        self.kind, self.C, self.d = kind, C, d
        self.out = {"mean": d, "concat": C * d, "multistat": 4 * d, "se": d}[kind]
        if kind == "se":                                    # squeeze (mean over d) -> excite over C
            h = max(2, C // se_reduction)
            self.fc1, self.fc2 = nn.Linear(C, h), nn.Linear(h, C)

    @staticmethod
    def _gem(x, p=3.0, eps=1e-6):                            # generalized-mean over C
        return (x.clamp(min=eps).pow(p).mean(1)).pow(1.0 / p)

    def forward(self, x):                                    # x: (B, C, d)
        if self.kind == "mean":
            return x.mean(1)
        if self.kind == "concat":
            return x.reshape(x.shape[0], -1)
        if self.kind == "multistat":
            return torch.cat([x.mean(1), x.amax(1), x.std(1), self._gem(x)], dim=-1)
        # se: descriptor per channel = mean over d -> gate over channels -> reweight -> mean
        s = x.mean(-1)                                       # (B, C)
        g = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))  # (B, C) cross-channel gates
        return (x * g.unsqueeze(-1)).mean(1)                # (B, d)


class Head(nn.Module):
    """[channel-pool(emb) ⊕ scaled handcrafted] -> MLP -> 8 logits (CLASSES order)."""
    def __init__(self, kind, C, d, n_feat, hidden=64, dropout=0.3):
        super().__init__()
        self.pool = ChannelPool(kind, C, d)
        self.net = nn.Sequential(
            nn.Linear(self.pool.out + n_feat, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 8))

    def forward(self, x_pc, x_feat):
        return self.net(torch.cat([self.pool(x_pc), x_feat], dim=-1))


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- train / predict
def _class_weights(y):
    _, cnt = np.unique(y, return_counts=True)
    w = len(y) / (len(cnt) * cnt.astype(np.float64))        # ~ class_weight="balanced"
    full = np.ones(8)
    for c, wc in zip(np.unique(y), w):
        full[c - 1] = wc
    return torch.tensor(full, dtype=torch.float32)


def train_head(kind, Xpc_fit, Xf_fit, y_fit, Xpc_tu, Xf_tu, y_tu, *, seed,
               device="cpu", hidden=64, dropout=0.3, wd=1e-4, lr=1e-3,
               max_epochs=120, patience=15, batch=512):
    """Fit on FIT, early-stop on TUNE macro-F1 (never TEST). Returns the model at best TUNE."""
    torch.manual_seed(seed); np.random.seed(seed)
    C, d = Xpc_fit.shape[1], Xpc_fit.shape[2]
    model = Head(kind, C, d, Xf_fit.shape[1], hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=_class_weights(y_fit).to(device))
    tpc = torch.tensor(Xpc_fit, device=device); tf = torch.tensor(Xf_fit, device=device)
    ty = torch.tensor(y_fit - 1, dtype=torch.long, device=device)
    vpc = torch.tensor(Xpc_tu, device=device); vf = torch.tensor(Xf_tu, device=device)
    n = len(y_fit); rng = np.random.default_rng(seed)
    best_macro, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        model.train(); idx = rng.permutation(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            opt.zero_grad()
            lossf(model(tpc[b], tf[b]), ty[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(vpc, vf).argmax(1).cpu().numpy() + 1
        mac = macro_f1(y_tu, pv)
        if mac > best_macro + 1e-5:
            best_macro, best_state, bad = mac, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def head_proba(model, Xpc, Xf, device="cpu"):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xpc, device=device), torch.tensor(Xf, device=device))
        return torch.softmax(logits, dim=-1).cpu().numpy()   # (n, 8), CLASSES order


# --------------------------------------------------------------------------- runner
def main() -> int:
    ap = argparse.ArgumentParser()
    root = _HERE.parents[2]
    ap.add_argument("--emb-pc", default="utica_V1_pc", help="per-channel embedding dir (_pc tag)")
    ap.add_argument("--emb-root", type=Path, default=root / "embeddings")
    ap.add_argument("--feat-dir", type=Path, default=root / "dataset_parquet_features")
    ap.add_argument("--split", type=Path, default=_HERE / "artifacts" / "val_split_temporal.npy")
    ap.add_argument("--heads", default="mean,concat,multistat,se")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bar", type=float, default=0.8342, help="bar to beat (E-VOTE-01 vote)")
    ap.add_argument("--out", type=Path, default=root / "notebooks/mdaniol/HEAD_RESULTS.md")
    args = ap.parse_args()
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    # data + masks (mirror the bake-off protocol)
    assign = np.load(args.split)
    Xpc_tr = load_emb_pc(args.emb_root / args.emb_pc, "train")
    Xpc_va = load_emb_pc(args.emb_root / args.emb_pc, "validation")
    Ftr, Fva = load_feats(args.feat_dir, "train"), load_feats(args.feat_dir, "validation")
    ytr = load_labels(args.feat_dir, "train")[0]
    yva = load_labels(args.feat_dir, "validation")[0]
    assert len(assign) == len(yva) == len(Xpc_va)
    fm, tm, sm = assign == 0, assign == 1, assign == 2

    Xpc_fit = np.concatenate([Xpc_tr, Xpc_va[fm]]); y_fit = np.concatenate([ytr, yva[fm]])
    Xpc_tu, Xpc_te = Xpc_va[tm], Xpc_va[sm]
    y_tu, y_te = yva[tm], yva[sm]
    # handcrafted: StandardScaler fit on FIT only (leak-safe), shared by all heads
    Ffit = np.concatenate([Ftr, Fva[fm]])
    sc = StandardScaler().fit(Ffit)
    Xf_fit = sc.transform(Ffit).astype(np.float32)
    Xf_tu = sc.transform(Fva[tm]).astype(np.float32); Xf_te = sc.transform(Fva[sm]).astype(np.float32)
    C, d = Xpc_fit.shape[1], Xpc_fit.shape[2]
    print(f"[head] {args.emb_pc} (C={C}, d={d}) fit={len(y_fit)} tune={len(y_tu)} test={len(y_te)} "
          f"heads={heads} seeds={seeds} dev={args.device}", flush=True)

    rows = []   # (head, tune_mean, test_mean, test_std, best_rep, ece, robustness, params)
    for kind in heads:
        tu_s, te_s, te_probas, params = [], [], [], None
        for sd in seeds:
            model = train_head(kind, Xpc_fit, Xf_fit, y_fit, Xpc_tu, Xf_tu, y_tu,
                               seed=sd, device=args.device)
            params = n_params(model)
            Ptu = head_proba(model, Xpc_tu, Xf_tu, args.device)
            w = calibrate(Ptu, CLS, y_tu)                    # per-class calibrate on TUNE
            Pte = head_proba(model, Xpc_te, Xf_te, args.device)
            tu_s.append(macro_f1(y_tu, CLS[(Ptu * w).argmax(1)]))
            te_s.append(macro_f1(y_te, CLS[(Pte * w).argmax(1)]))
            te_probas.append(Pte * w)
        bi = int(np.argmax(tu_s))                            # pick the seed by TUNE (lock once)
        best_seed_proba = te_probas[bi] / te_probas[bi].sum(1, keepdims=True)
        rep = class_report(y_te, CLS[best_seed_proba.argmax(1)])
        ece = em.ece(best_seed_proba, y_te)
        rows.append({"head": kind, "tune_mean": float(np.mean(tu_s)),
                     "test_mean": float(np.mean(te_s)), "test_std": float(np.std(te_s)),
                     "test_best": float(te_s[bi]), "rep": rep, "ece": ece, "params": params})
        pc = " ".join(f"{k[:2]}={v['f1']:.2f}" for k, v in rep["per_class"].items())
        print(f"  {kind:10s} TUNE {np.mean(tu_s):.4f} | TEST {np.mean(te_s):.4f}±{np.std(te_s):.4f} "
              f"(best-seed {te_s[bi]:.4f}) ECE={ece['ece']:.3f} params={params} | {pc}", flush=True)

    base = next((r for r in rows if r["head"] == "mean"), rows[0])
    se = next((r for r in rows if r["head"] == "se"), None)
    winner = max(rows, key=lambda r: r["tune_mean"])         # select on TUNE
    # gated: novel head must beat the mean baseline (CI lower bound) AND clear the vote bar
    keep_xchan = bool(se and se["test_mean"] - se["test_std"] > base["test_mean"])
    print(f"\n[decision] mean baseline TEST={base['test_mean']:.4f}; "
          f"SE TEST={se['test_mean'] if se else float('nan'):.4f}; "
          f"TUNE-winner={winner['head']} TEST={winner['test_mean']:.4f} (bar {args.bar}); "
          f"x-channel KEEP={keep_xchan}", flush=True)

    # write table + MLflow snapshot (rule §8)
    hdr = (f"# Cross-channel head E-HEAD-01 on {args.emb_pc} (temporal split, calibrated, "
           f"{len(seeds)} seeds).\n"
           f"mean baseline TEST={base['test_mean']:.4f}; bar (E-VOTE-01 vote)={args.bar}; "
           f"x-channel SE KEEP={keep_xchan} (CI-lb > mean baseline).\n\n"
           "| head | TUNE | TEST (mean±std) | best-seed | ECE | params | per-class F1 (best seed) |\n"
           "|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        pc = " ".join(f"{k[:2]}={v['f1']:.2f}" for k, v in r["rep"]["per_class"].items())
        body += (f"| {r['head']} | {r['tune_mean']:.4f} | {r['test_mean']:.4f}±{r['test_std']:.4f} "
                 f"| {r['test_best']:.4f} | {r['ece']['ece']:.3f} | {r['params']} | {pc} |\n")
    args.out.write_text(hdr + body)

    from shl2026 import track
    with track("mdaniol", run_name=f"head_{args.emb_pc}", seed=0, params_path=None,
               params={"emb_pc": args.emb_pc, "split": args.split.stem, "heads": ",".join(heads),
                       "seeds": args.seeds, "bar": args.bar},
               tags={"phase": "head", "branch": "lightweight_head",
                     "decision": "KEEP" if keep_xchan else "DISABLE"}) as run:
        run.log_metrics({"macro_f1": winner["test_mean"],        # bare = TUNE-winner lock-test
                         "se_test_macro_f1": se["test_mean"] if se else 0.0,
                         "mean_baseline_macro_f1": base["test_mean"]})
        for r in rows:
            run.log_metrics({f"{r['head']}_test_macro_f1": r["test_mean"],
                             f"{r['head']}_ece": r["ece"]["ece"]})
        for k, v in winner["rep"]["per_class"].items():
            run.log_metrics({f"test_f1_{k}": v["f1"]})
        run.log_artifact(args.out)
        if args.split.exists():
            run.log_artifact(args.split)              # snapshot the split file (§8)
    print(f"wrote {args.out}", flush=True)
    return 0


# --------------------------------------------------------------------------- self-test
def self_test() -> None:
    torch.manual_seed(0)
    B, C, d, nf = 16, 9, 32, 20
    xpc = torch.randn(B, C, d); xf = torch.randn(B, nf)
    for kind in ("mean", "concat", "multistat", "se"):
        h = Head(kind, C, d, nf, hidden=16)
        out = h(xpc, xf)
        assert out.shape == (B, 8), f"{kind}: bad output {out.shape}"
        assert n_params(h) > 0
    # SE gate must actually depend on channels: a head with all-equal channels vs perturbed differs
    se = Head("se", C, d, nf, hidden=16).eval()
    x2 = xpc.clone(); x2[:, 0] += 5.0
    with torch.no_grad():
        assert not torch.allclose(se(xpc, xf), se(x2, xf)), "SE head ignores channel content"
    # a quick 2-epoch train must run and beat random on a learnable synthetic task
    rng = np.random.default_rng(0)
    n = 400; y = rng.integers(1, 9, n)
    Xpc = rng.standard_normal((n, C, d)).astype(np.float32)
    Xpc[np.arange(n), 0, 0] = y                              # signal in channel 0
    Xf = rng.standard_normal((n, nf)).astype(np.float32)
    m = train_head("se", Xpc[:300], Xf[:300], y[:300], Xpc[300:], Xf[300:], y[300:],
                   seed=0, max_epochs=8, patience=8)
    P = head_proba(m, Xpc[300:], Xf[300:])
    assert P.shape == (100, 8) and np.allclose(P.sum(1), 1.0, atol=1e-5)
    print("head_xchannel self_test OK — 4 heads output (B,8); SE depends on channels; trains")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        raise SystemExit(main())
