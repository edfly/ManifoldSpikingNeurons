"""
v38 supplementary experiment (P3-1): OGB large-scale benchmark.

Dataset: ogbg-molhiv (41,127 graphs, binary HIV activity prediction).
  - Official scaffold split (80/10/10, deterministic) — the standard OGB
    protocol, no CV (labels are highly imbalanced: ~3.5% active).
  - Metric: ROC-AUC (OGB standard, via ogb Evaluator) + accuracy.
  - Test metric reported at the best-validation epoch.

Methods:
  - OM  : OnManifoldGNN (imported from experiment_v37, identical
          implementation) with L=2, T=5, h=e=16 — the NCI1-scale
          configuration, to test whether the manifold spiking mechanism
          scales to 41k graphs.
  - GIN : parameter-matched plain GIN baseline (same backbone, no
          neuron layer).

3 seeds (42/123/456). Outputs -> spikergnn_results/v38_ogb/<key>.json
(incremental + resume).

Usage:
  python experiment_v38_ogb.py            # all (OM + GIN)
  python experiment_v38_ogb.py om
  python experiment_v38_ogb.py gin
  python experiment_v38_ogb.py status
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
import sys, os, json, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.serialization
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# PyTorch >= 2.6 defaults torch.load(weights_only=True); PyG's processed
# dataset files need these globals allowlisted.
import torch_geometric.data.data as _pyg_data
import torch_geometric.data.storage as _pyg_storage
torch.serialization.add_safe_globals(
    [_pyg_data.DataEdgeAttr, _pyg_data.DataTensorAttr,
     _pyg_storage.GlobalStorage])

from experiment_v37 import OnManifoldGNN

OUT = _os.path.join(RESULTS_ROOT, "v38_ogb")
DATA_ROOT = "data/OGB"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

SEEDS = [42, 123, 456]
EPOCHS = 10
BS = 64
LAMBDA_SR = 0.5
R_TARGET = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GINBaseline(nn.Module):
    """Parameter-matched plain GIN (no neuron layer)."""

    def __init__(self, in_dim, n_class=1, h_dim=32, hidden=64, L=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, h_dim))
        self.convs = nn.ModuleList()
        for _ in range(L):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(h_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, h_dim))))
        self.classifier = nn.Linear(h_dim, n_class)

    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = self.proj(x)
        for cv in self.convs:
            x = cv(x, ei)
        return self.classifier(global_mean_pool(x, b))


def evaluate(model, loader, evaluator):
    model.eval()
    ys, ps, srs = [], [], []
    with torch.no_grad():
        for d in loader:
            d = d.to(DEVICE)
            out = model(d)
            if isinstance(out, tuple):
                out, sr = out
                srs.append(float(sr))
            logits = out.squeeze(-1)
            ys.append(d.y.squeeze(-1).float().cpu())
            ps.append(torch.sigmoid(logits).cpu())
    y = torch.cat(ys).unsqueeze(1).numpy()
    p = torch.cat(ps).unsqueeze(1).numpy()
    res = evaluator.eval({"y_true": y, "y_pred": p})
    acc = float(((p > 0.5).astype(float) == y).mean())
    return res["rocauc"], acc, (float(np.mean(srs)) if srs else None)


def train_run(make_model, split_idx, evaluator, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = make_model.dataset
    tr = DataLoader([dataset[i] for i in split_idx["train"]], BS, True)
    va = DataLoader([dataset[i] for i in split_idx["valid"]], BS, False)
    te = DataLoader([dataset[i] for i in split_idx["test"]], BS, False)

    m = make_model().to(DEVICE)
    main_p, th_p = [], []
    for nm, p in m.named_parameters():
        (th_p if "u_th_learnable" in nm else main_p).append(p)
    opt = optim.Adam([
        {"params": main_p, "lr": 1e-3, "weight_decay": 5e-4},
        {"params": th_p, "lr": 1e-4, "weight_decay": 0}])
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, 1e-5)
    crit = nn.BCEWithLogitsLoss()

    best_va, best_te, best_acc, best_sr = -1.0, -1.0, 0.0, None
    for ep in range(EPOCHS):
        m.train()
        t0 = time.time()
        for d in tr:
            d = d.to(DEVICE)
            opt.zero_grad()
            out = m(d)
            sr = None
            if isinstance(out, tuple):
                out, sr = out
            loss = crit(out.squeeze(-1), d.y.squeeze(-1).float())
            if sr is not None:
                loss = loss + LAMBDA_SR * (sr - R_TARGET) ** 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        sch.step()
        va_rocauc, _, _ = evaluate(m, va, evaluator)
        if va_rocauc > best_va:
            best_va = va_rocauc
            best_te, best_acc, best_sr = evaluate(m, te, evaluator)
        log(f"    ep{ep + 1}/{EPOCHS} va_rocauc={va_rocauc:.4f} "
            f"({time.time() - t0:.0f}s)")
    return best_va, best_te, best_acc, best_sr


def run_job(key, make_model):
    fp = os.path.join(OUT, f"{key}.json")
    if os.path.exists(fp):
        d = json.load(open(fp))
        if len(d.get("rocaucs", [])) >= len(SEEDS):
            log(f"  {key}: cached rocauc={np.mean(d['rocaucs']):.4f}")
            return d
    else:
        d = {"rocaucs": [], "accs": [], "srs": [],
             "seeds": [], "epochs": EPOCHS, "bs": BS}
    evaluator = Evaluator(name="ogbg-molhiv")
    split_idx = make_model.split_idx
    for seed in SEEDS:
        if seed in d["seeds"]:
            continue
        t0 = time.time()
        va, te, acc, sr = train_run(make_model, split_idx, evaluator, seed)
        d["rocaucs"].append(te)
        d["accs"].append(acc)
        d["srs"].append(sr)
        d["seeds"].append(seed)
        d["mean_rocauc"] = float(np.mean(d["rocaucs"]))
        d["std_rocauc"] = float(np.std(d["rocaucs"]))
        d["mean_acc"] = float(np.mean(d["accs"]))
        json.dump(d, open(fp, "w"), indent=2)
        log(f"  {key} s{seed}: test_rocauc={te:.4f} va={va:.4f} "
            f"acc={acc:.4f} sr={sr if sr is None else round(sr, 3)} "
            f"({time.time() - t0:.0f}s)")
    return d


class OMFactory:
    def __init__(self, dataset, split_idx):
        self.dataset = dataset
        self.split_idx = split_idx

    def __call__(self):
        return OnManifoldGNN(self.dataset.num_features, 1,
                             L=2, T=5, h_dim=16, e_dim=16)


class GINFactory:
    def __init__(self, dataset, split_idx):
        self.dataset = dataset
        self.split_idx = split_idx

    def __call__(self):
        return GINBaseline(self.dataset.num_features, 1)


def get_data():
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=DATA_ROOT)
    split_idx = dataset.get_idx_split()
    # molhiv node features are categorical int64 -> cast to float for Linear
    dataset._data.x = dataset._data.x.float()
    log(f"  molhiv: {len(dataset)} graphs, {dataset.num_features}f "
        f"train/val/test = {len(split_idx['train'])}/"
        f"{len(split_idx['valid'])}/{len(split_idx['test'])}")
    return dataset, split_idx


def main(stage="all"):
    log(f"Device: {DEVICE}")
    dataset, split_idx = get_data()
    if stage in ("all", "om"):
        log("=== molhiv_OM (L=2, T=5) ===")
        run_job("molhiv_OM", OMFactory(dataset, split_idx))
    if stage in ("all", "gin"):
        log("=== molhiv_GIN (L=2) ===")
        run_job("molhiv_GIN", GINFactory(dataset, split_idx))
    log("=== ALL COMPLETE ===")
    status()


def status():
    print(f"\n{'key':12s} {'n':>2s} {'rocauc':>7s} {'+/-':>6s} {'acc':>6s}")
    for key in ["molhiv_OM", "molhiv_GIN"]:
        fp = os.path.join(OUT, f"{key}.json")
        if os.path.exists(fp):
            d = json.load(open(fp))
            print(f"{key:12s} {len(d['rocaucs']):2d} "
                  f"{d['mean_rocauc']:7.4f} {d['std_rocauc']:6.4f} "
                  f"{d['mean_acc']:6.4f}")
        else:
            print(f"{key:12s}  -")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "status":
        status()
    else:
        main(stage)
