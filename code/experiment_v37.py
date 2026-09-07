"""
v37 experiment runner: reruns all OM (on-manifold spiking) configurations
with the FIXED ManifoldLIFNeuron and the ACTIVE L_sr regularizer.

!! 修改本文件前必读（项目硬性约定 2026-08-29）!!
本文件是论文 Algorithm 1 与全部 OM 主实验的实现。改动后必须：
  1. 同步检查/更新 verify_algorithm_mapping.py 的 CHECKS 指纹清单
  2. 同步更新论文附录 C（Algorithm 1 -> 实现对照表）与 Algorithm 1 伪代码
  3. 跑 `bash check_all.sh` 四重核验，全绿才可交付
注意：核验只防"已有条目漂移"，不防"新增内容漏记"。
若本次改动引入了对照表中没有的新机制，必须手工补条目，否则核验不会报警。

Fixes relative to the v30_ext runs:
  1. Input magnitude clamp is an ABSOLUTE constant (0.6), no longer scaled
     by u_th — the learnable threshold regains control of the firing rate.
  2. Spike counters accumulate across forward calls (reset in
     reset_membrane()), so rates are true episode rates on all call paths.
  3. L_sr = lambda_sr * (surrogate_rate - r*)^2 uses the differentiable
     surrogate rate (gradient flows to u_th_learnable and W).

Protocol (paper Setup): 3 seeds (42, 123, 456) x 3-fold stratified CV
(random_state=42 — same folds as the v30_ext baselines, keeping paired
tests valid), Adam lr=1e-3 / wd=5e-4 with a separate u_th group
(lr=1e-4, wd=0), ATan surrogate (beta=2), lambda_sr=0.5, r*=0.4,
grad clip 5.0, cosine annealing.

Configs (canonical v37 protocol):
  MUTAG    : L=3, T=20, epochs=50, bs=16
  PROTEINS : L=2, T=10, epochs=25, bs=16
  DD       : L=2, T=10, epochs=25, bs=16
  NCI1     : L=2, T=5,  epochs=10, bs=32  (default; ablation varies one)
Baselines (NoSpikes/GIN/GCN/SAGE/MSG/SGC) are unaffected by the neuron
fix and are NOT rerun; their v30_ext results remain valid.

Outputs -> spikergnn_results/v37/<key>.json with incremental saving and
resume support. Each fold appends: accuracy, test spike rate, final u_th.

Usage:
  python experiment_v37.py            # everything, in order
  python experiment_v37.py main       # 4 main configs only
  python experiment_v37.py ablation   # ablation configs only
  python experiment_v37.py status     # progress summary
"""

# ── path resolution (adapted for the post/ layout) ─────────────────
# Results live in ../results relative to this file; the manuscript lives
# two directories up (the project root). Override with env vars if needed.
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)                     # .../post
RESULTS_ROOT = _os.environ.get(
    "RESULTS_ROOT", _os.path.join(_ROOT, "results"))
POST_ROOT = _os.environ.get("POST_ROOT", _ROOT)
FIGURES_ROOT = _os.environ.get(
    "FIGURES_ROOT", _os.path.join(_ROOT, "figures"))
# project root is TWO levels above post/ (post/ lives in programfiles/)
PAPER_TEX = _os.environ.get(
    "PAPER_TEX", _os.path.join(_os.path.dirname(_os.path.dirname(_ROOT)),
                               "ManifoldValuedSpikingNeurons_Neurocomputing_v38.tex"))
# ───────────────────────────────────────────────────────────────────
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from spikergnn.neurons import ManifoldLIFNeuron
from spikergnn.manifolds import ProductManifold

OUT = _os.path.join(RESULTS_ROOT, "v37")
DT_ROOT = _os.environ.get("TU_ROOT", _os.path.join(_os.path.dirname(_ROOT), "data", "TU"))
os.makedirs(OUT, exist_ok=True)
NF = 3
SEEDS = [42, 123, 456]
LAMBDA_SR = 0.5
R_TARGET = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class OnManifoldGNN(nn.Module):
    """GIN backbone + ManifoldLIFNeuron, full-T call, L_sr-ready."""

    def __init__(self, in_dim, n_class, h_dim=16, e_dim=16, hidden=64,
                 L=3, T=20, c=1.0):
        super().__init__()
        D = h_dim + e_dim
        self.T = T
        self.h_dim = h_dim
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, D))
        self.manifold = ProductManifold(d_h=h_dim, d_e=e_dim, c=c)
        self.convs = nn.ModuleList()
        self.neurons = nn.ModuleList()
        for _ in range(L):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, D))))
            self.neurons.append(ManifoldLIFNeuron(
                manifold=self.manifold, ablation="full",
                surrogate_type="atan"))
        self.classifier = nn.Linear(D, n_class)

    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = self.proj(x)
        ph = self.manifold.hyp.expmap0(x[:, :self.h_dim])
        pe = x[:, self.h_dim:]
        pf = torch.cat([ph, pe], -1)
        xs = pf.unsqueeze(1).repeat(1, self.T, 1)
        layer_sr = []
        for cv, nr in zip(self.convs, self.neurons):
            mg = torch.stack([cv(xs[:, t, :], ei) for t in range(self.T)], dim=1)
            nr.reset_membrane()
            _, (ph, pe) = nr(mg, (ph, pe))
            layer_sr.append(nr.spike_rate_surrogate())  # differentiable
            pf = torch.cat([ph, pe], -1)
            xs = pf.unsqueeze(1).repeat(1, self.T, 1)
        sr = torch.stack(layer_sr).mean()
        return self.classifier(global_mean_pool(pf, b)), sr


# (key, dataset, kwargs, epochs, bs) — order = execution priority
CONFIGS = [
    # ── main table (tab:main OM column) ──
    ("MUTAG_OM",     "MUTAG",    dict(L=3, T=20),                50, 16),
    ("NCI1_OM",      "NCI1",     dict(L=2, T=5),                 10, 32),
    ("PROTEINS_OM",  "PROTEINS", dict(L=2, T=10),                25, 16),
    ("DD_OM",        "DD",       dict(L=2, T=10),                25, 16),
    # ── NCI1 ablation (vary one from default L=2, T=5, c=1.0, d_h=16) ──
    ("NCI1_abl_T10",  "NCI1",    dict(L=2, T=10),                10, 32),
    ("NCI1_abl_c0.5", "NCI1",    dict(L=2, T=5, c=0.5),          10, 32),
    ("NCI1_abl_c2.0", "NCI1",    dict(L=2, T=5, c=2.0),          10, 32),
    ("NCI1_abl_hd8",  "NCI1",    dict(L=2, T=5, h_dim=8, e_dim=8),   10, 32),
    ("NCI1_abl_hd32", "NCI1",    dict(L=2, T=5, h_dim=32, e_dim=32), 10, 32),
    ("NCI1_abl_L1",   "NCI1",    dict(L=1, T=5),                 10, 32),
    ("NCI1_abl_L3",   "NCI1",    dict(L=3, T=5),                 10, 32),
    ("NCI1_abl_T20",  "NCI1",    dict(L=2, T=20),                10, 32),
    # ── MUTAG time-step ablation (L=3 config; T=20 row == MUTAG_OM) ──
    ("MUTAG_abl_T5",  "MUTAG",   dict(L=3, T=5),                 50, 16),
    ("MUTAG_abl_T10", "MUTAG",   dict(L=3, T=10),                50, 16),
]
MAIN_KEYS = {"MUTAG_OM", "NCI1_OM", "PROTEINS_OM", "DD_OM"}


def train_fold(ds, ti, vi, bs, epochs, seed, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = OnManifoldGNN(ds.num_features, ds.num_classes, **kw).to(DEVICE)
    main_p, th_p = [], []
    for nm, p in m.named_parameters():
        (th_p if "u_th_learnable" in nm else main_p).append(p)
    opt = optim.Adam([
        {"params": main_p, "lr": 1e-3, "weight_decay": 5e-4},
        {"params": th_p, "lr": 1e-4, "weight_decay": 0}])
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-5)
    crit = nn.CrossEntropyLoss()
    tr = DataLoader([ds[i] for i in ti], bs, True)
    te = DataLoader([ds[i] for i in vi], bs, False)
    for _ in range(epochs):
        m.train()
        for d in tr:
            d = d.to(DEVICE)
            opt.zero_grad()
            o, sr = m(d)
            loss = crit(o, d.y) + LAMBDA_SR * (sr - R_TARGET) ** 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        sch.step()
    m.eval()
    c, t, srs, uths = 0, 0, [], []
    with torch.no_grad():
        for d in te:
            d = d.to(DEVICE)
            o, sr = m(d)
            c += (o.argmax(-1) == d.y).sum().item()
            t += d.num_graphs
            srs.append(float(sr))
    for nr in m.neurons:
        uths.append(float(nr.u_th_learnable.item()))
    return 100.0 * c / t, float(np.mean(srs)), uths


def run_job(key, ds, lbs, kw, epochs, bs):
    fp = os.path.join(OUT, f"{key}.json")
    total = NF * len(SEEDS)
    accs, srs, uths = [], [], []
    if os.path.exists(fp):
        d = json.load(open(fp))
        accs, srs, uths = d.get("accs", []), d.get("srs", []), d.get("uths", [])
        if len(accs) >= total:
            log(f"  {key}: {np.mean(accs):.1f} +/- {np.std(accs):.1f} (cached)")
            return d
    folds = list(StratifiedKFold(NF, shuffle=True, random_state=42)
                 .split(range(len(ds)), lbs))
    for idx in range(len(accs), total):
        seed = SEEDS[idx // NF]
        ti, vi = folds[idx % NF]
        t0 = time.time()
        a, sr, uth = train_fold(ds, ti, vi, bs, epochs, seed, **kw)
        accs.append(a); srs.append(sr); uths.append(uth)
        d = {"accs": accs, "srs": srs, "uths": uths,
             "mean": float(np.mean(accs)), "std": float(np.std(accs)),
             "config": {**kw, "epochs": epochs, "bs": bs},
             "lambda_sr": LAMBDA_SR, "r_target": R_TARGET}
        json.dump(d, open(fp, "w"), indent=2)
        log(f"    {key} s{seed} f{idx % NF + 1}: acc={a:.1f}% sr={sr:.3f} "
            f"u_th={uth[0]:.3f} ({idx + 1}/{total}, {time.time()-t0:.0f}s)")
    return d


def main(stage="all"):
    log(f"Device: {DEVICE}")
    keys = [c for c in CONFIGS
            if stage == "all" or (stage == "main") == (c[0] in MAIN_KEYS)]
    dcache = {}
    for key, ds_name, kw, epochs, bs in keys:
        if ds_name not in dcache:
            log(f"Loading {ds_name}...")
            ds = TUDataset(root=DT_ROOT, name=ds_name)
            dcache[ds_name] = (ds, [ds[i].y.item() for i in range(len(ds))])
            log(f"  {ds_name}: {len(ds)} graphs, {ds.num_features}f")
        ds, lbs = dcache[ds_name]
        log(f"=== {key} ({kw}, {epochs} epochs) ===")
        run_job(key, ds, lbs, kw, epochs, bs)
    log("=== ALL COMPLETE ===")
    status()


def status():
    print(f"\n{'key':20s} {'n':>3s} {'mean':>6s} {'std':>5s} {'sr':>5s}")
    for key, *_ in CONFIGS:
        fp = os.path.join(OUT, f"{key}.json")
        if os.path.exists(fp):
            d = json.load(open(fp))
            print(f"{key:20s} {len(d['accs']):3d} {d['mean']:6.1f} "
                  f"{d['std']:5.1f} {np.mean(d['srs']):5.3f}")
        else:
            print(f"{key:20s}   -")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "status":
        status()
    else:
        main(stage)
