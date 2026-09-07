"""v38 supplementary experiment: hyperbolic (Riemannian) ANN baseline.

Motivation (peer review, domain-expert view):
  The paper claims that on-manifold computation preserves geometric structure,
  yet every baseline is Euclidean (GIN / GCN / SAGE); the only Riemannian
  comparison is MSG, which is a *spiking* method. A reviewer therefore cannot
  tell whether the observed gain comes from the manifold geometry or from the
  spiking mechanism itself.

Design: complete a 2x2 factorial control.

                 Euclidean          Manifold (hyperbolic)
      ANN        GIN (have)         HGCN  (this script)
      Spiking    SGC / MSG (have)   OM    (have)

  * HGCN vs GIN  isolates the effect of the geometry (both are ANN).
  * OM   vs HGCN isolates the effect of spiking (both are on-manifold).
  * OM   vs GIN  is the overall effect.

Implementation follows Chami et al. (2019), Hyperbolic Graph Convolutional
Neural Networks: hyperbolic linear layer (exp_0 W log_0), tangent-space
neighbour aggregation, tangent-space activation — all re-projected onto the
Poincare ball. Configuration is matched to OM (same L, d_h=16, c=1.0, hidden
64) so that the comparison isolates the mechanism.

Protocol identical to the other baselines (experiment_v38_5seeds.py):
50 epochs, bs=16, Adam 1e-3 / wd 5e-4, cosine to 1e-5, grad clip 5.0,
5 seeds x 3-fold stratified CV, incremental JSON, degenerate single-class
runs re-run once with seed+10000 and both values recorded.

Usage:
    python experiment_v38_hgcn.py              # all four datasets
    python experiment_v38_hgcn.py status
    python experiment_v38_hgcn.py smoke         # one fold, 2 epochs
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from sklearn.model_selection import StratifiedKFold

from spikergnn.manifolds import ProductManifold

# Hyperbolic width. HGC_DIM=32 gives a version whose representational width
# matches OM (16 hyperbolic + 16 Euclidean = 32), for a capacity-matched
# manifold-ANN vs manifold-spiking comparison.
HGC_DIM = int(os.environ.get("HGC_DIM", "16"))
OUT = os.environ.get("HGC_OUT", "spikergnn_results/v38_hgcn")
DT_ROOT = "data/TU"
os.makedirs(OUT, exist_ok=True)
NF = 3
SEEDS5 = [42, 123, 456, 789, 2024]
DATASETS = ["MUTAG", "NCI1", "PROTEINS", "DD"]
BASE_EPOCHS = 50
BASE_BS = 16

# Per-dataset training budget, matched to the OM protocol in
# experiment_v37.CONFIGS. This matters: OM uses 10 epochs on NCI1 and 25 on
# PROTEINS/DD (not 50), so HGCN must use the same budget for the OM-vs-HGCN
# comparison to isolate the mechanism rather than the training budget.
OM_BUDGET = {
    "MUTAG":    (50, 16),
    "NCI1":     (10, 32),
    "PROTEINS": (25, 16),
    "DD":       (25, 16),
}

# OM's per-dataset depth. HypGNN defaults to L=3, but OM uses L=2 on NCI1,
# PROTEINS and DD, so a depth-unaware comparison gives HGCN an extra layer on
# three of four datasets. Set HGC_MATCH_L=1 to match OM's depth exactly; the
# result is written to its own directory so the original sweep is untouched.
OM_DEPTH = {
    "MUTAG":    3,
    "NCI1":     2,
    "PROTEINS": 2,
    "DD":       2,
}
MATCH_L = os.environ.get("HGC_MATCH_L", "0") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── hyperbolic layers (Chami et al. 2019) ─────────────────────────────

class HypLinear(nn.Module):
    """Hyperbolic linear layer: exp_0( W log_0(x) + b )."""

    def __init__(self, manifold, in_dim, out_dim, bias=True):
        super().__init__()
        self.manifold = manifold
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        v = self.manifold.logmap0(x)
        v = self.linear(v)
        return self.manifold.projx(self.manifold.expmap0(v))


class HypAgg(MessagePassing):
    """Degree-normalised neighbour aggregation performed in the tangent
    space at the origin, then mapped back (as in HGCN)."""

    def __init__(self, manifold):
        super().__init__(aggr="add")
        self.manifold = manifold

    def forward(self, x, edge_index):
        ei, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = ei
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        v = self.manifold.logmap0(x)
        out = self.propagate(ei, x=v, norm=norm)
        return self.manifold.projx(self.manifold.expmap0(out))

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class HypBN(nn.Module):
    """Hyperbolic BatchNorm: normalise in the tangent space at the origin,
    then map back onto the ball.

    This mirrors what BatchNorm does for the GIN baseline. Without it,
    activations drift toward the ball boundary (|x| -> 1/sqrt(c)) where the
    Poincare conformal factor 2/(1 - c|x|^2) blows up: the representation
    saturates and the gradient vanishes — the hyperbolic analogue of the
    dead-ReLU collapse that the no-BN variant exhibits on NCI1.
    """

    def __init__(self, manifold, dim):
        super().__init__()
        self.manifold = manifold
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        v = self.manifold.logmap0(x)
        v = self.bn(v)
        return self.manifold.projx(self.manifold.expmap0(v))


class HypAct(nn.Module):
    """Activation applied in the tangent space at the origin."""

    def __init__(self, manifold, act=nn.ReLU()):
        super().__init__()
        self.manifold = manifold
        self.act = act

    def forward(self, x):
        v = self.manifold.logmap0(x)
        return self.manifold.projx(self.manifold.expmap0(self.act(v)))


class HypGNN(nn.Module):
    """Hyperbolic ANN graph classifier (HGCN-style), configuration matched
    to the on-manifold spiking model: same L, d_h, curvature c, hidden."""

    def __init__(self, in_dim, n_class, h_dim=16, hidden=64, L=3, c=1.0,
                 bn=True):
        super().__init__()
        self.h_dim = h_dim
        self.manifold = ProductManifold(d_h=h_dim, d_e=0, c=c).hyp
        self.proj_in = nn.Linear(in_dim, h_dim)
        self.layers = nn.ModuleList()
        for _ in range(L):
            self.layers.append(nn.ModuleList([
                HypAgg(self.manifold),
                HypLinear(self.manifold, h_dim, hidden),
                HypBN(self.manifold, hidden) if bn else nn.Identity(),
                HypAct(self.manifold),
                HypLinear(self.manifold, hidden, h_dim),
            ]))
        self.classifier = nn.Linear(h_dim, n_class)

    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = self.manifold.projx(self.manifold.expmap0(self.proj_in(x)))
        for agg, lin1, bn, act, lin2 in self.layers:
            x = agg(x, ei)
            x = lin1(x)
            x = bn(x)
            x = act(x)
            x = lin2(x)
            x = self.manifold.projx(x)
        # readout in the tangent space at the origin
        v = self.manifold.logmap0(x)
        return self.classifier(global_mean_pool(v, b))


# ── training (identical protocol to the other baselines) ───────────────

def train_fold(model, ds, ti, vi, bs, epochs, seed, return_preds=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-5)
    crit = nn.CrossEntropyLoss()
    tr = DataLoader([ds[i] for i in ti], bs, True)
    te = DataLoader([ds[i] for i in vi], bs, False)
    for _ in range(epochs):
        model.train()
        for d in tr:
            d = d.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(d), d.y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sch.step()
    model.eval()
    c, t = 0, 0
    pred_classes = set()
    with torch.no_grad():
        for d in te:
            d = d.to(DEVICE)
            pred = model(d).argmax(-1)
            c += (pred == d.y).sum().item()
            pred_classes.update(pred.cpu().tolist())
            t += d.num_graphs
    acc = 100.0 * c / t
    if return_preds:
        return acc, pred_classes
    return acc


def run_dataset(make_model, key, ds, lbs, epochs, bs, depth=None):
    fp = os.path.join(OUT, f"{key}.json")
    total = NF * len(SEEDS5)
    recs, degenerate = [], []
    if os.path.exists(fp):
        d = json.load(open(fp))
        recs = d.get("accs", [])
        degenerate = d.get("degenerate", [])
        if len(recs) >= total:
            log(f"  {key}: {np.mean(recs):.1f} +/- {np.std(recs):.1f} (cached)")
            return d
    folds = list(StratifiedKFold(NF, shuffle=True, random_state=42)
                 .split(range(len(ds)), lbs))
    for idx in range(len(recs), total):
        seed = SEEDS5[idx // NF]
        ti, vi = folds[idx % NF]
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make_model(ds)
        a, pcs = train_fold(model, ds, ti, vi, bs, epochs, seed, True)
        note = ""
        if len(pcs) == 1:
            torch.manual_seed(seed + 10000)
            np.random.seed(seed + 10000)
            model = make_model(ds)
            a2, pcs2 = train_fold(model, ds, ti, vi, bs, epochs,
                                  seed + 10000, True)
            degenerate.append({
                "idx": idx, "seed": seed, "fold": idx % NF + 1,
                "acc": a, "rerun_seed": seed + 10000, "rerun_acc": a2,
                "rerun_single_class": len(pcs2) == 1})
            a = a2
            note = f" DEGENERATE -> rerun s{seed + 10000} acc={a:.1f}"
        recs.append(a)
        d = {"accs": recs,
             "mean": float(np.mean(recs)), "std": float(np.std(recs)),
             "seeds": SEEDS5, "epochs": epochs, "bs": bs,
             "degenerate": degenerate,
             "model": "HypGNN (HGCN-style, Chami et al. 2019)",
             "manifold": f"Poincare ball, d_h={HGC_DIM}, c=1.0",
             "hgc_dim": HGC_DIM,
             "depth": depth,
             "depth_matched_to_OM": MATCH_L,
             "stability_policy": ("single-class degenerate runs re-run once "
                                  "with seed+10000")}
        json.dump(d, open(fp, "w"), indent=2)
        log(f"    {key} s{seed} f{idx % NF + 1}: acc={a:.1f}% "
            f"({idx + 1}/{total}, {time.time() - t0:.0f}s){note}")
    return d


def load_dataset(name):
    ds = TUDataset(root=DT_ROOT, name=name)
    return ds, [ds[i].y.item() for i in range(len(ds))]


def main():
    log(f"Device: {DEVICE}  (d_h={HGC_DIM}, depth-match={MATCH_L})")
    for ds_name in DATASETS:
        ds, lbs = load_dataset(ds_name)
        epochs, bs = OM_BUDGET[ds_name]
        depth = OM_DEPTH[ds_name] if MATCH_L else 3
        log(f"=== {ds_name}_HGCN ({epochs} epochs, bs={bs}, L={depth}; "
            f"matched to OM budget) ===")

        def mk(ds, _L=depth):
            return HypGNN(ds.num_features, ds.num_classes, h_dim=HGC_DIM, L=_L)

        run_dataset(mk, f"{ds_name}_HGCN", ds, lbs, epochs, bs, depth=depth)
    log("=== ALL COMPLETE ===")
    status()


def status():
    print(f"\n{'key':16s} {'n':>3s} {'mean':>6s} {'std':>5s}")
    for ds in DATASETS:
        fp = os.path.join(OUT, f"{ds}_HGCN.json")
        if os.path.exists(fp):
            d = json.load(open(fp))
            print(f"{ds + '_HGCN':16s} {len(d['accs']):3d} {d['mean']:6.1f} "
                  f"{d['std']:5.1f}")
        else:
            print(f"{ds + '_HGCN':16s}   -")


def smoke():
    """One fold, two epochs, to validate the hyperbolic ops before the run."""
    ds, lbs = load_dataset("MUTAG")
    folds = list(StratifiedKFold(NF, shuffle=True, random_state=42)
                 .split(range(len(ds)), lbs))
    ti, vi = folds[0]
    torch.manual_seed(42)
    np.random.seed(42)
    m = HypGNN(ds.num_features, ds.num_classes, h_dim=HGC_DIM)
    a = train_fold(m, ds, ti, vi, BASE_BS, 2, 42)
    log(f"SMOKE OK: MUTAG HGCN d_h={HGC_DIM} 2 epochs -> acc={a:.1f}% "
        f"(expect ~66-85%)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        status()
    elif cmd == "smoke":
        smoke()
    else:
        main()
