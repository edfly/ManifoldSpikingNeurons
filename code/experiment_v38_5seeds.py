"""
v38 supplementary experiment (P2-1): extend the paired statistical
protocol from 3 seeds (n=9) to 5 seeds (n=15) per method.

Design:
  - Seeds: 42, 123, 456, 789, 2024 (first three = published v37 protocol).
  - Folds: StratifiedKFold(3, shuffle, random_state=42) — identical to
    every previous run, so paired tests stay valid and new samples are
    directly comparable.
  - OM (on-manifold spiking, v37 fixed neuron + L_sr): APPENDED to the
    existing v37 JSONs (deterministic resume; seeds 789/2024 only).
  - ANN baselines + NoSpikes (GIN/GCN/SAGE/NS): fresh 15-run rerun with
    the experiment_ext_fast protocol (Adam 1e-3/5e-4, cosine, clip 5.0,
    50 epochs, bs=16) that produced the published v30_ext numbers.
  - MSG: fresh 15-run rerun of the supervised adaptation of the user's
    MSGStarEncoder (Sun24 reimplementation; logmap0 readout + linear
    classifier). Spike counts recorded via IF-neuron hooks.
  - SGC (SpikeGCN, Zhu22): fresh 15-run rerun — GCN backbone with an IF
    activation, FIXED threshold 1.0, T=5 internal steps, decay 1-1/tau,
    hard reset, ATan surrogate (beta=2). Peak membrane magnitude and
    spike counts recorded for the fragility analysis.

Outputs -> spikergnn_results/v38_5seeds/<DS>_<METHOD>.json

Usage:
  python experiment_v38_5seeds.py om         # OM append (seeds 789, 2024)
  python experiment_v38_5seeds.py ann        # NS/GIN/GCN/SAGE fresh 15-run
  python experiment_v38_5seeds.py msg        # MSG fresh 15-run
  python experiment_v38_5seeds.py sgc        # SGC fresh 15-run
  python experiment_v38_5seeds.py calibrate  # GIN seed42/fold1 vs old JSON
  python experiment_v38_5seeds.py status
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
import sys, os, json, time, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv, SAGEConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold

import experiment_v37 as e37
from spikergnn.neurons import ATanSurrogate
from msg_baseline.models import MSGStarEncoder
from msg_baseline.manifolds import PoincareManifold

OUT = _os.path.join(RESULTS_ROOT, "v38_5seeds")
DT_ROOT = _os.environ.get("TU_ROOT", _os.path.join(_os.path.dirname(_ROOT), "data", "TU"))
os.makedirs(OUT, exist_ok=True)
NF = 3
SEEDS5 = [42, 123, 456, 789, 2024]
DATASETS = ["MUTAG", "NCI1", "PROTEINS", "DD"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OM configs (from experiment_v37 CONFIGS, main keys)
OM_CONFIGS = {
    "MUTAG":    dict(kw=dict(L=3, T=20),  epochs=50, bs=16),
    "NCI1":     dict(kw=dict(L=2, T=5),   epochs=10, bs=32),
    "PROTEINS": dict(kw=dict(L=2, T=10),  epochs=25, bs=16),
    "DD":       dict(kw=dict(L=2, T=10),  epochs=25, bs=16),
}
# Baseline protocol (experiment_ext_fast): 50 epochs, bs=16, all datasets
BASE_EPOCHS = 50
BASE_BS = 16


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════
class NoSpikesGNN(nn.Module):
    def __init__(self, in_dim, n_class, hidden=64, L=3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList()
        for _ in range(L):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden))))
        self.classifier = nn.Linear(hidden, n_class)

    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = self.proj(x)
        for cv in self.convs:
            x = torch.relu(cv(x, ei))
        return self.classifier(global_mean_pool(x, b))


def _ann_conv(kind, hidden):
    if kind == "GIN":
        return GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden)))
    if kind == "GCN":
        return GCNConv(hidden, hidden)
    return SAGEConv(hidden, hidden)


class AnnGNN(nn.Module):
    """Vanilla ANN GNN baseline (GIN / GCN / SAGE).

    GIN follows the reference implementation of Xu et al. (2019), which
    uses BatchNorm after every aggregation. The no-BN variant used by
    the legacy v30_ext protocol suffers deterministic dead-ReLU collapse
    on NCI1 in the current environment (4/5 seeds at single-class 50%);
    BN removes the instability and matches published GIN accuracy.
    GCN / SAGE follow their reference implementations (no BN).
    """

    def __init__(self, kind, in_dim, n_class, hidden=64, L=3):
        super().__init__()
        self.kind = kind
        self.proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([_ann_conv(kind, hidden) for _ in range(L)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden) for _ in range(L)]) if kind == "GIN" else None
        self.classifier = nn.Linear(hidden, n_class)

    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = torch.relu(self.proj(x))
        for i, cv in enumerate(self.convs):
            x = cv(x, ei)
            if self.bns is not None:
                x = self.bns[i](x)
            x = torch.relu(x)
        return self.classifier(global_mean_pool(x, b))


class MSGSupervised(nn.Module):
    """MSG (Sun24) adapted to supervised graph classification.

    MSGStarEncoder (the paper's verified reimplementation of the
    tangent-space MSNeuron pipeline) + logmap0 readout + linear head.
    Spike counts accumulated via forward hooks on the IF neurons.
    """
    def __init__(self, in_dim, n_class, hidden=32, L=3, c=1.0):
        super().__init__()
        self.enc = MSGStarEncoder(in_dim, hidden_dim=hidden, num_layers=L,
                                  manifold_c=c, epsilon=0.1, tau=20.0)
        self.manifold = self.enc.manifold
        self.classifier = nn.Linear(hidden, n_class)
        self._spike_buf = 0.0
        for layer in self.enc.layers:
            layer.if_neuron.register_forward_hook(self._if_hook())

    def _if_hook(self):
        def hook(module, inputs, output):
            s_seq = output[0]  # (N, 1, D) binary spikes in eval
            self._spike_buf += float(s_seq.sum().item())
        return hook

    def forward(self, d):
        z = self.enc(d.x, d.edge_index, d.batch)
        v = self.manifold.logmap0(z)
        return self.classifier(v)

    def count_spikes(self):
        """Total IF spikes since construction (test-time measurement)."""
        return int(self._spike_buf)


class SpikeGCN(nn.Module):
    """SGC baseline (Zhu22 SpikeGCN): GCN backbone with an IF activation.

    Fixed threshold 1.0, membrane decay 1-1/tau, hard reset, ATan
    surrogate (beta=2). Each GCN layer rate-encodes its output over T
    internal time steps. Tracks peak membrane magnitude and spike count.
    """
    def __init__(self, in_dim, n_class, hidden=64, L=3, T=5, tau=20.0):
        super().__init__()
        self.T = T
        self.decay = 1.0 - 1.0 / tau
        self.proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(L)])
        self.surrogate = ATanSurrogate(beta=2.0)
        self.classifier = nn.Linear(hidden, n_class)
        self.peak_membrane = 0.0
        self.spike_count = 0

    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = torch.relu(self.proj(x))
        for gcn in self.convs:
            h = gcn(x, ei)
            v = torch.zeros_like(h)
            spikes = []
            for _ in range(self.T):
                v = self.decay * v + h
                self.peak_membrane = max(self.peak_membrane,
                                         float(v.abs().max()))
                s = self.surrogate(v - 1.0)  # FIXED threshold 1.0
                self.spike_count += int(s.sum().item())
                spikes.append(s)
                v = v * (1 - s)  # hard reset
            x = torch.stack(spikes, dim=0).mean(dim=0)
        return self.classifier(global_mean_pool(x, b))


# ══════════════════════════════════════════════════════════════════
# Training loops
# ══════════════════════════════════════════════════════════════════
def train_generic(model, ds, ti, vi, bs, epochs, seed, return_preds=False):
    """Generic ANN/baseline training (experiment_ext_fast protocol)."""
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
            o = model(d)
            if isinstance(o, tuple):
                o = o[0]
            crit(o, d.y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sch.step()
    model.eval()
    c, t = 0, 0
    pred_classes = set()
    extra = {}
    with torch.no_grad():
        for d in te:
            d = d.to(DEVICE)
            o = model(d)
            if isinstance(o, tuple):
                o = o[0]
            pred = o.argmax(-1)
            c += (pred == d.y).sum().item()
            pred_classes.update(pred.cpu().tolist())
            t += d.num_graphs
    if isinstance(model, SpikeGCN):
        extra = {"peak_membrane": model.peak_membrane,
                 "spike_count": model.spike_count}
    if return_preds:
        return 100.0 * c / t, extra, pred_classes
    return 100.0 * c / t, extra


def run_baseline(make_model, key, ds, lbs, epochs, bs):
    """15-run (5 seeds x 3 folds) baseline with incremental saving.

    Stability policy (uniform for every method, disclosed in the paper):
    a run whose model predicts a SINGLE class on the entire test fold is
    a degenerate training collapse; it is re-run ONCE with seed+10000
    (fixed, deterministic offset). Both accuracies are kept in the JSON
    ("degenerate" list); the reported accs use the re-run value.
    """
    fp = os.path.join(OUT, f"{key}.json")
    total = NF * len(SEEDS5)
    recs = []
    if os.path.exists(fp):
        d = json.load(open(fp))
        recs = d.get("accs", [])
        if len(recs) >= total:
            log(f"  {key}: {np.mean(recs):.1f} +/- {np.std(recs):.1f} (cached)")
            return d
    else:
        d = None
    folds = list(StratifiedKFold(NF, shuffle=True, random_state=42)
                 .split(range(len(ds)), lbs))
    extras = d.get("extras", []) if d else []
    degenerate = d.get("degenerate", []) if d else []
    for idx in range(len(recs), total):
        seed = SEEDS5[idx // NF]
        ti, vi = folds[idx % NF]
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make_model(ds)
        a, extra, pcs = train_generic(model, ds, ti, vi, bs, epochs, seed,
                                      return_preds=True)
        note = ""
        if len(pcs) == 1:
            # degenerate single-class collapse -> one deterministic rerun
            torch.manual_seed(seed + 10000)
            np.random.seed(seed + 10000)
            model = make_model(ds)
            a2, extra2, pcs2 = train_generic(
                model, ds, ti, vi, bs, epochs, seed + 10000,
                return_preds=True)
            degenerate.append({
                "idx": idx, "seed": seed, "fold": idx % NF + 1,
                "acc": a, "rerun_seed": seed + 10000, "rerun_acc": a2,
                "rerun_single_class": len(pcs2) == 1})
            a, extra = a2, extra2
            note = (f" DEGENERATE(single-class) -> rerun s{seed + 10000} "
                    f"acc={a:.1f}")
        recs.append(a)
        extras.append(extra)
        d = {"accs": recs,
             "mean": float(np.mean(recs)), "std": float(np.std(recs)),
             "seeds": SEEDS5, "epochs": epochs, "bs": bs,
             "extras": extras, "degenerate": degenerate,
             "stability_policy": ("single-class degenerate runs re-run "
                                  "once with seed+10000"),
             "protocol_note": ("GIN baseline uses BatchNorm per the "
                               "reference implementation of Xu et al. "
                               "2019; GCN/SAGE/NS/MSG/SGC unchanged")}
        json.dump(d, open(fp, "w"), indent=2)
        log(f"    {key} s{seed} f{idx % NF + 1}: acc={a:.1f}% "
            f"({idx + 1}/{total}, {time.time()-t0:.0f}s) {extra}{note}")
    return d


def load_dataset(name):
    ds = TUDataset(root=DT_ROOT, name=name)
    return ds, [ds[i].y.item() for i in range(len(ds))]


# ══════════════════════════════════════════════════════════════════
# Stages
# ══════════════════════════════════════════════════════════════════
def stage_om():
    """Append seeds 789/2024 to the v37 OM results (deterministic resume)."""
    e37.OUT = OUT
    e37.SEEDS = SEEDS5
    for ds_name in DATASETS:
        src = _os.path.join(RESULTS_ROOT, "v37", f"{ds_name}_OM.json")
        dst = os.path.join(OUT, f"{ds_name}_OM.json")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            log(f"  copied {src} -> {dst}")
    # Run via experiment_v37's own machinery (resume from 9)
    for key, ds_name, kw, epochs, bs in e37.CONFIGS:
        if key not in e37.MAIN_KEYS:
            continue
        ds, lbs = load_dataset(ds_name)
        log(f"=== OM append {key} ({kw}, {epochs} epochs) ===")
        e37.run_job(key, ds, lbs, kw, epochs, bs)


def stage_ann():
    """Fresh 15-run NS / GIN / GCN / SAGE."""
    for ds_name in DATASETS:
        ds, lbs = load_dataset(ds_name)
        specs = [
            ("NS",   lambda ds: NoSpikesGNN(ds.num_features, ds.num_classes)),
            ("GIN",  lambda ds: AnnGNN("GIN", ds.num_features, ds.num_classes)),
            ("GCN",  lambda ds: AnnGNN("GCN", ds.num_features, ds.num_classes)),
            ("SAGE", lambda ds: AnnGNN("SAGE", ds.num_features, ds.num_classes)),
        ]
        for name, mk in specs:
            log(f"=== {ds_name}_{name} ({BASE_EPOCHS} epochs) ===")
            run_baseline(mk, f"{ds_name}_{name}", ds, lbs, BASE_EPOCHS, BASE_BS)


def stage_msg():
    """Fresh 15-run MSG (supervised adaptation of MSGStarEncoder)."""
    for ds_name in DATASETS:
        ds, lbs = load_dataset(ds_name)
        log(f"=== {ds_name}_MSG ({BASE_EPOCHS} epochs) ===")

        def mk(ds, _name=ds_name):
            return MSGSupervised(ds.num_features, ds.num_classes)

        run_baseline(mk, f"{ds_name}_MSG", ds, lbs, BASE_EPOCHS, BASE_BS)


def stage_sgc():
    """Fresh 15-run SGC (SpikeGCN)."""
    for ds_name in DATASETS:
        ds, lbs = load_dataset(ds_name)
        log(f"=== {ds_name}_SGC ({BASE_EPOCHS} epochs) ===")

        def mk(ds):
            return SpikeGCN(ds.num_features, ds.num_classes)

        run_baseline(mk, f"{ds_name}_SGC", ds, lbs, BASE_EPOCHS, BASE_BS)


def stage_calibrate():
    """Sanity check: GIN seed 42 fold 1 vs the published v30_ext value."""
    for ds_name in DATASETS:
        ds, lbs = load_dataset(ds_name)
        folds = list(StratifiedKFold(NF, shuffle=True, random_state=42)
                     .split(range(len(ds)), lbs))
        ti, vi = folds[0]
        torch.manual_seed(42)
        np.random.seed(42)
        m = AnnGNN("GIN", ds.num_features, ds.num_classes)
        a, _ = train_generic(m, ds, ti, vi, BASE_BS, BASE_EPOCHS, 42)
        old = json.load(open(_os.path.join(RESULTS_ROOT, "v30_ext", f"{ds_name}_GIN.json")))["accs"][0]
        log(f"  {ds_name}: new GIN s42 f1 = {a:.1f}% | old v30_ext = {old:.1f}% "
            f"| diff = {a - old:+.1f}")


def status():
    print(f"\n{'key':18s} {'n':>3s} {'mean':>6s} {'std':>5s}")
    for ds_name in DATASETS:
        for m in ["OM", "NS", "GIN", "GCN", "SAGE", "MSG", "SGC"]:
            fp = os.path.join(OUT, f"{ds_name}_{m}.json")
            if os.path.exists(fp):
                d = json.load(open(fp))
                print(f"{ds_name+'_'+m:18s} {len(d['accs']):3d} "
                      f"{d['mean']:6.1f} {d['std']:5.1f}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    log(f"Device: {DEVICE}")
    if stage == "status":
        status()
    elif stage == "calibrate":
        stage_calibrate()
    elif stage == "om":
        stage_om()
    elif stage == "ann":
        stage_ann()
    elif stage == "msg":
        stage_msg()
    elif stage == "sgc":
        stage_sgc()
    else:
        stage_om()
        stage_ann()
        stage_msg()
        stage_sgc()
    log("=== STAGE COMPLETE ===")
