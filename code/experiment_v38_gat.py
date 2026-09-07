"""v38 supplementary experiment: GAT baseline (Euclidean ANN, attention).

Purpose: give the Euclidean/ANN cell of the 2x2 factorial a second reference
point, so the comparison does not rest on GIN alone.

                 Euclidean                   Manifold (hyperbolic)
      ANN        GIN (have), GAT (this)      HGCN (have)
      Spiking    SGC / MSG (have)            OM (have)

Configuration matched to the other baselines: hidden=64, L=3. GAT uses
heads=4 with 16 channels per head so the concatenated output is 64-d, i.e.
the same width as GIN's MLP. Training protocol is byte-identical to
experiment_v38_hgcn.py (50 epochs, bs=16, Adam 1e-3 / wd 5e-4, cosine to
1e-5, grad clip 5.0, 5 seeds x 3-fold stratified CV, incremental JSON,
degenerate single-class runs re-run once with seed+10000).

Usage:
    python experiment_v38_gat.py
    python experiment_v38_gat.py status
    python experiment_v38_gat.py smoke
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold

# reuse the identical training loop / bookkeeping as the HGCN run so the two
# supplementary baselines cannot diverge in protocol
from experiment_v38_hgcn import (
    train_fold, run_dataset, load_dataset, log, status as _status_unused,
    OUT as _UNUSED_OUT, DATASETS, SEEDS5, BASE_EPOCHS, BASE_BS, NF,
)

OUT = "spikergnn_results/v38_gat"
os.makedirs(OUT, exist_ok=True)


class GATGNN(nn.Module):
    """Euclidean ANN graph classifier with graph attention.

    heads=4 x 16 channels = 64-d output, matching the 64-d width of the GIN
    baseline so the comparison isolates the aggregation mechanism.

    bn=True mirrors the GIN baseline (BatchNorm after every aggregation, Xu
    et al. reference implementation). This is not cosmetic: the legacy no-BN
    protocol suffers deterministic dead-ReLU collapse on NCI1 (single-class
    50%), so omitting BN would confound the comparison with an implementation
    artefact rather than the mechanism under test.
    """

    def __init__(self, in_dim, n_class, hidden=64, L=3, heads=4, bn=True):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(L):
            self.convs.append(GATConv(hidden, hidden // heads,
                                      heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden) if bn else None)
        self.classifier = nn.Linear(hidden, n_class)

    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = self.proj(x)
        for cv, bn in zip(self.convs, self.bns):
            x = cv(x, ei)
            if bn is not None:
                x = bn(x)
            x = torch.relu(x)
        return self.classifier(global_mean_pool(x, b))


def run_dataset_here(make_model, key, ds, lbs, epochs, bs):
    """Same bookkeeping as the HGCN run but writing into the GAT directory."""
    import experiment_v38_hgcn as H
    fp = os.path.join(OUT, f"{key}.json")
    total = NF * len(SEEDS5)
    recs, degenerate = [], []
    if os.path.exists(fp):
        d = json.load(open(fp))
        recs = d.get("accs", [])
        degenerate = d.get("degenerate", [])
        if len(recs) >= total:
            log(f"  {key}: {np.mean(recs):.1f} +/- {np.std(recs):.1f} (cached)")
            return
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
             "model": "GATGNN (Veličković et al. 2018), heads=4 x 16",
             "manifold": "Euclidean",
             "stability_policy": ("single-class degenerate runs re-run once "
                                  "with seed+10000")}
        json.dump(d, open(fp, "w"), indent=2)
        log(f"    {key} s{seed} f{idx % NF + 1}: acc={a:.1f}% "
            f"({idx + 1}/{total}, {time.time() - t0:.0f}s){note}")


def status():
    print(f"\n{'key':16s} {'n':>3s} {'mean':>6s} {'std':>5s}")
    for ds in DATASETS:
        fp = os.path.join(OUT, f"{ds}_GAT.json")
        if os.path.exists(fp):
            d = json.load(open(fp))
            print(f"{ds + '_GAT':16s} {len(d['accs']):3d} {d['mean']:6.1f} "
                  f"{d['std']:5.1f}")
        else:
            print(f"{ds + '_GAT':16s}   -")


def smoke():
    ds, lbs = load_dataset("MUTAG")
    folds = list(StratifiedKFold(NF, shuffle=True, random_state=42)
                 .split(range(len(ds)), lbs))
    ti, vi = folds[0]
    torch.manual_seed(42)
    np.random.seed(42)
    m = GATGNN(ds.num_features, ds.num_classes)
    from experiment_v38_hgcn import train_fold as tf, DEVICE
    a = tf(m, ds, ti, vi, BASE_BS, 2, 42)
    log(f"SMOKE OK: MUTAG GAT 2 epochs -> acc={a:.1f}%")


def main():
    log(f"=== GAT baseline (Euclidean ANN, attention) ===")
    for ds_name in DATASETS:
        ds, lbs = load_dataset(ds_name)
        log(f"=== {ds_name}_GAT ({BASE_EPOCHS} epochs) ===")

        def mk(ds):
            return GATGNN(ds.num_features, ds.num_classes)

        run_dataset_here(mk, f"{ds_name}_GAT", ds, lbs, BASE_EPOCHS, BASE_BS)
    log("=== ALL COMPLETE ===")
    status()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        status()
    elif cmd == "smoke":
        smoke()
    else:
        main()
