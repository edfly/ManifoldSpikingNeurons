"""
v38 supplementary experiment (P2-7): product manifold with a spherical
component — H^{d_h} x S^{d_s-1} x R^{d_e} vs the paper's H^{d_h} x R^{d_e}.

Design:
  - SphereLIFNeuron mirrors spikergnn/neurons.py ManifoldLIFNeuron EXACTLY
    (same membrane dynamics, absolute input clamp 0.6, ATan surrogate,
    hard reset, geodesic spike interpolation, p_bar EMA, L_sr-ready
    surrogate rate) but carries a 3-component state (p_h, p_s, p_e).
    The spherical branch follows the same update pattern as the
    hyperbolic branch: logmap toward p_bar + expmap drift, then geodesic
    interpolation on spike.
  - Configs identical to the v37 main table (same L/T/epochs/bs) so the
    comparison isolates the geometry change.
  - 3 seeds (42/123/456) x 3-fold stratified CV (random_state=42, same
    folds as all previous runs).
  - d_h = d_s = d_e = 16 (ambient); total representation dim 48 vs 32
    for OM — this is the capacity-extended variant as suggested by the
    review report ("S^2 x H^d x R^d").

Outputs -> spikergnn_results/v38_s2/<key>.json (incremental + resume).

Usage:
  python experiment_v38_s2.py            # all 4 datasets
  python experiment_v38_s2.py status
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
import sys, os, json, time, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from spikergnn.neurons import ATanSurrogate
from spikergnn.manifolds import ProductManifold
from spikergnn.sphere import SphereManifold

OUT = _os.path.join(RESULTS_ROOT, "v38_s2")
DT_ROOT = _os.environ.get("TU_ROOT", _os.path.join(_os.path.dirname(_ROOT), "data", "TU"))
os.makedirs(OUT, exist_ok=True)
NF = 3
SEEDS = [42, 123, 456]
LAMBDA_SR = 0.5
R_TARGET = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SphereLIFNeuron(nn.Module):
    """ManifoldLIFNeuron with a 3-component state (H, S, R).

    Faithful copy of spikergnn.neurons.ManifoldLIFNeuron dynamics with a
    spherical component added between the hyperbolic and Euclidean parts.
    """

    def __init__(self, hyp, sph, d_h, d_s, d_e,
                 alpha_decay=0.9, surrogate_beta=2.0,
                 p_bar_decay=0.9, input_mag_cap=0.6):
        super().__init__()
        self.hyp = hyp
        self.sph = sph
        self.d_h, self.d_s, self.d_e = d_h, d_s, d_e
        self.d_total = d_h + d_s + d_e
        self.alpha = alpha_decay
        self.p_bar_decay = p_bar_decay
        self.input_mag_cap = input_mag_cap

        # Learnable parameters (same as ManifoldLIFNeuron)
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.W = nn.Linear(self.d_total, self.d_total, bias=False)
        self.input_ln = nn.LayerNorm(self.d_total)
        self.spike_alpha = nn.Parameter(torch.tensor(0.5))
        self.u_th_learnable = nn.Parameter(torch.tensor(1.0))
        self.surrogate = ATanSurrogate(beta=surrogate_beta)

        self._spike_count = 0
        self._total_neurons = 0
        self._spike_rate = 0.0
        self._surrogate_spike_sum = None
        self.register_buffer("_u_mag", None)

    def split(self, z):
        return z[..., :self.d_h], z[..., self.d_h:self.d_h + self.d_s], \
               z[..., self.d_h + self.d_s:]

    def merge(self, ph, ps, pe):
        return torch.cat([ph, ps, pe], dim=-1)

    def _get_u_mag(self, N, device):
        if self._u_mag is None or self._u_mag.shape[0] != N \
                or self._u_mag.device != device:
            self._u_mag = torch.zeros(N, device=device)
        return self._u_mag

    def forward(self, x_seq, p_init):
        N, T, d_total = x_seq.shape
        device = x_seq.device

        p_h, p_s, p_e = p_init
        p_h, p_s, p_e = p_h.clone(), p_s.clone(), p_e.clone()

        u_mag = self._get_u_mag(N, device)
        p_bar_h = p_h.detach().clone()
        p_bar_s = p_s.detach().clone()
        p_bar_e = p_e.detach().clone()

        spike_list = []
        beta_val = self.beta.clamp(min=0.01, max=1.0)
        gamma_val = self.gamma.clamp(min=0.01, max=2.0)

        for t in range(T):
            I_t = x_seq[:, t, :]

            # ── State evolution: tangent direction toward p_bar + input ──
            log_bar_h = self.hyp.logmap(p_bar_h, p_h)
            log_bar_s = self.sph.logmap(p_bar_s, p_s)
            log_bar_e = p_bar_e - p_e

            I_h, I_s, I_e = self.split(I_t)
            v_h = beta_val * log_bar_h + gamma_val * I_h
            v_s = beta_val * log_bar_s + gamma_val * I_s
            v_e = beta_val * log_bar_e + gamma_val * I_e

            p_new_h = self.hyp.projx(self.hyp.expmap(v_h, p_h))
            p_new_s = self.sph.projx(self.sph.expmap(v_s, p_s))
            p_new_e = p_e + v_e

            # ── Membrane potential (identical to v37 neuron) ──
            I_t_normed = self.input_ln(I_t)
            w_input = self.W(I_t_normed)
            w_input_mag = w_input.norm(dim=-1) / math.sqrt(self.d_total)
            w_input_mag = torch.clamp(w_input_mag, max=self.input_mag_cap)
            u_mag = self.alpha * u_mag + w_input_mag
            effective_threshold = self.u_th_learnable.clamp(min=0.05)
            u_mag = torch.clamp(u_mag, max=effective_threshold.detach() * 1.5)

            # ── Spike trigger (ATan surrogate, hard reset) ──
            s_out = self.surrogate(u_mag - effective_threshold)
            self._spike_count += int(s_out.sum().item())
            if self._surrogate_spike_sum is None:
                self._surrogate_spike_sum = s_out.sum()
            else:
                self._surrogate_spike_sum = self._surrogate_spike_sum + s_out.sum()
            u_mag = u_mag * (1 - s_out)

            # ── Geodesic spike interpolation per component ──
            spike_mask = s_out.unsqueeze(-1)
            spike_alpha = torch.sigmoid(self.spike_alpha).clamp(min=0.01, max=0.99)

            # Hyperbolic branch (as in ManifoldLIFNeuron)
            v_geo_h = self.hyp.logmap(p_new_h, p_h)
            p_h_spike = self.hyp.projx(
                self.hyp.expmap(spike_alpha * v_geo_h, p_h))
            p_h = spike_mask * p_h_spike + (1 - spike_mask) * p_h
            p_h = self.hyp.projx(p_h)

            # Spherical branch (mirrors the hyperbolic branch)
            v_geo_s = self.sph.logmap(p_new_s, p_s)
            p_s_spike = self.sph.projx(
                self.sph.expmap(spike_alpha * v_geo_s, p_s))
            p_s = spike_mask * p_s_spike + (1 - spike_mask) * p_s
            p_s = self.sph.projx(p_s)

            # Euclidean branch (linear blend)
            p_e = (1 - spike_alpha * spike_mask) * p_e + \
                  spike_alpha * spike_mask * p_new_e

            # p_bar EMA
            with torch.no_grad():
                p_bar_h = self.p_bar_decay * p_bar_h + \
                          (1 - self.p_bar_decay) * p_h.detach()
                p_bar_h = self.hyp.projx(p_bar_h)
                p_bar_s = self.sph.projx(
                    self.p_bar_decay * p_bar_s +
                    (1 - self.p_bar_decay) * p_s.detach())
                p_bar_e = self.p_bar_decay * p_bar_e + \
                          (1 - self.p_bar_decay) * p_e.detach()

            spike_list.append(spike_mask.expand_as(I_t))

        self._u_mag = u_mag.detach().clone()
        self._total_neurons += N * T
        self._spike_rate = self._spike_count / max(self._total_neurons, 1)
        spikes = torch.stack(spike_list, dim=1)
        return spikes, (p_h, p_s, p_e)

    def reset_membrane(self):
        self._u_mag = None
        self._spike_count = 0
        self._total_neurons = 0
        self._spike_rate = 0.0
        self._surrogate_spike_sum = None

    def spike_rate_surrogate(self):
        if self._total_neurons == 0 or self._surrogate_spike_sum is None:
            return torch.tensor(0.0, device=self.beta.device)
        return self._surrogate_spike_sum / self._total_neurons


class OnManifoldS2GNN(nn.Module):
    """GIN backbone + SphereLIFNeuron on H x S x R."""

    def __init__(self, in_dim, n_class, h_dim=16, s_dim=16, e_dim=16,
                 hidden=64, L=3, T=20, c=1.0):
        super().__init__()
        D = h_dim + s_dim + e_dim
        self.T = T
        self.d_h, self.d_s, self.d_e = h_dim, s_dim, e_dim
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, D))
        self.hyp = ProductManifold(d_h=h_dim, d_e=e_dim, c=c).hyp
        self.sph = SphereManifold(dim=s_dim)
        self.convs = nn.ModuleList()
        self.neurons = nn.ModuleList()
        for _ in range(L):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, D))))
            self.neurons.append(SphereLIFNeuron(
                self.hyp, self.sph, h_dim, s_dim, e_dim))
        self.classifier = nn.Linear(D, n_class)

    def forward(self, d):
        x, ei, b = d.x, d.edge_index, d.batch
        x = self.proj(x)
        ph = self.hyp.expmap0(x[:, :self.d_h])
        ps = self.sph.projx(x[:, self.d_h:self.d_h + self.d_s])
        pe = x[:, self.d_h + self.d_s:]
        pf = torch.cat([ph, ps, pe], -1)
        xs = pf.unsqueeze(1).repeat(1, self.T, 1)
        layer_sr = []
        for cv, nr in zip(self.convs, self.neurons):
            mg = torch.stack([cv(xs[:, t, :], ei) for t in range(self.T)], dim=1)
            nr.reset_membrane()
            _, (ph, ps, pe) = nr(mg, (ph, ps, pe))
            layer_sr.append(nr.spike_rate_surrogate())
            pf = torch.cat([ph, ps, pe], -1)
            xs = pf.unsqueeze(1).repeat(1, self.T, 1)
        sr = torch.stack(layer_sr).mean()
        return self.classifier(global_mean_pool(pf, b)), sr


# Same configs as the v37 main table
CONFIGS = [
    ("MUTAG_S2",    "MUTAG",    dict(L=3, T=20), 50, 16),
    ("NCI1_S2",     "NCI1",     dict(L=2, T=5),  10, 32),
    ("PROTEINS_S2", "PROTEINS", dict(L=2, T=10), 25, 16),
    ("DD_S2",       "DD",       dict(L=2, T=10), 25, 16),
]


def train_fold(ds, ti, vi, bs, epochs, seed, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = OnManifoldS2GNN(ds.num_features, ds.num_classes, **kw).to(DEVICE)
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
             "config": {**kw, "epochs": epochs, "bs": bs,
                        "s_dim": 16},
             "lambda_sr": LAMBDA_SR, "r_target": R_TARGET}
        json.dump(d, open(fp, "w"), indent=2)
        log(f"    {key} s{seed} f{idx % NF + 1}: acc={a:.1f}% sr={sr:.3f} "
            f"u_th={uth[0]:.3f} ({idx + 1}/{total}, {time.time()-t0:.0f}s)")
    return d


def main():
    log(f"Device: {DEVICE}")
    dcache = {}
    for key, ds_name, kw, epochs, bs in CONFIGS:
        if ds_name not in dcache:
            ds = TUDataset(root=DT_ROOT, name=ds_name)
            dcache[ds_name] = (ds, [ds[i].y.item() for i in range(len(ds))])
            log(f"  {ds_name}: {len(ds)} graphs, {ds.num_features}f")
        ds, lbs = dcache[ds_name]
        log(f"=== {key} ({kw}, {epochs} epochs) ===")
        run_job(key, ds, lbs, kw, epochs, bs)
    log("=== ALL COMPLETE ===")
    status()


def status():
    print(f"\n{'key':16s} {'n':>3s} {'mean':>6s} {'std':>5s} {'sr':>5s}")
    for key, *_ in CONFIGS:
        fp = os.path.join(OUT, f"{key}.json")
        if os.path.exists(fp):
            d = json.load(open(fp))
            print(f"{key:16s} {len(d['accs']):3d} {d['mean']:6.1f} "
                  f"{d['std']:5.1f} {np.mean(d['srs']):5.3f}")
        else:
            print(f"{key:16s}   -")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
    else:
        main()
