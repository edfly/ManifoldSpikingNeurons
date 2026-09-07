"""v38 supplementary: Euclidean degeneration of the on-manifold neuron.

The question this answers
-------------------------
OM beats the Euclidean spiking baselines (SGC/MSG) by a wide margin. But OM
differs from them in THREE ways at once:
  (1) the state lives on a curved manifold;
  (2) the update is a geodesic step with a sigmoid-gated geodesic spike
      interpolation instead of a scalar hard reset;
  (3) the spike-rate regulariser L_sr pins the firing rate.
So the observed gain could come from the geometry, or from (2)+(3) alone.

This script isolates (1) by running the *identical* model with the hyperbolic
component removed: ProductManifold(d_h=0, d_e=32) is exactly R^32 --
verified numerically (logmap(x,y) == y - x, expmap(x,v) == x + v, projx ==
identity, all to 0.00e+00). Hence with h_dim=0 the whole on-manifold
machinery degenerates as:

    geodesic                 -> straight line
    log/exp maps             -> vector subtraction / addition
    parallel transport       -> identity
    geodesic spike interp.   -> linear interpolation  p+ = p- + sigma(p_new - p-)
    sectional curvature K    -> 0

Everything else -- vector-valued state, leak-and-input update, the scalar
membrane magnitude u_mag, the ATan surrogate, the learnable threshold, the
sigmoid interpolation fraction sigma(alpha_s), and L_sr -- is untouched.

The resulting decomposition:
    SGC/MSG  ->  OM-E   : contribution of the update/regularisation mechanism
    OM-E     ->  OM     : contribution of the geometry (curvature) alone
    SGC/MSG  ->  OM     : total

Dimension is matched to OM (32 = 16 hyperbolic + 16 Euclidean) and the
per-dataset training budget follows experiment_v37.CONFIGS.

Usage:
    python experiment_v38_euclid.py
    python experiment_v38_euclid.py status
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset

import experiment_v37 as e37

OUT = "spikergnn_results/v38_euclid"
os.makedirs(OUT, exist_ok=True)
NF = 3
SEEDS5 = [42, 123, 456, 789, 2024]
DEVICE = e37.DEVICE
log = e37.log

# Per-dataset budget matched to experiment_v37.CONFIGS (same as OM).
# h_dim=0 removes curvature; e_dim=32 matches OM's total width 16+16.
CONFIGS = [
    # key,               dataset,    model kwargs,                    ep,  bs
    ("MUTAG_EUCLID",     "MUTAG",    dict(L=3, T=20, h_dim=0, e_dim=32), 50, 16),
    ("NCI1_EUCLID",      "NCI1",     dict(L=2, T=5,  h_dim=0, e_dim=32), 10, 32),
    ("PROTEINS_EUCLID",  "PROTEINS", dict(L=2, T=10, h_dim=0, e_dim=32), 25, 16),
    ("DD_EUCLID",        "DD",       dict(L=2, T=10, h_dim=0, e_dim=32), 25, 16),
]


def run_job(key, ds, lbs, kw, epochs, bs):
    fp = os.path.join(OUT, f"{key}.json")
    total = NF * len(SEEDS5)
    accs, srs, uths = [], [], []
    if os.path.exists(fp):
        d = json.load(open(fp))
        accs = d.get("accs", []); srs = d.get("srs", []); uths = d.get("uths", [])
        if len(accs) >= total:
            log(f"  {key}: {np.mean(accs):.1f} +/- {np.std(accs):.1f} (cached)")
            return d
    folds = list(StratifiedKFold(NF, shuffle=True, random_state=42)
                 .split(range(len(ds)), lbs))
    for idx in range(len(accs), total):
        seed = SEEDS5[idx // NF]
        ti, vi = folds[idx % NF]
        t0 = time.time()
        a, sr, uth = e37.train_fold(ds, ti, vi, bs, epochs, seed, **kw)
        accs.append(a); srs.append(sr); uths.append(uth)
        d = {"accs": accs, "srs": srs, "uths": uths,
             "mean": float(np.mean(accs)), "std": float(np.std(accs)),
             "config": {**kw, "epochs": epochs, "bs": bs},
             "lambda_sr": e37.LAMBDA_SR, "r_target": e37.R_TARGET,
             "note": ("Euclidean degeneration of OM: h_dim=0 => R^32, "
                      "K=0. Same update, spike interpolation and L_sr as OM; "
                      "only the curvature is removed.")}
        json.dump(d, open(fp, "w"), indent=2)
        log(f"    {key} s{seed} f{idx % NF + 1}: acc={a:.1f}% sr={sr:.3f} "
            f"u_th={uth[0]:.3f} ({idx + 1}/{total}, {time.time() - t0:.0f}s)")
    return d


def status():
    print(f"\n{'key':18s} {'n':>3s} {'mean':>6s} {'std':>5s} {'sr':>5s}")
    for key, *_ in CONFIGS:
        fp = os.path.join(OUT, f"{key}.json")
        if os.path.exists(fp):
            d = json.load(open(fp))
            sr = np.mean(d.get("srs", [0]))
            print(f"{key:18s} {len(d['accs']):3d} {d['mean']:6.1f} "
                  f"{d['std']:5.1f} {sr:5.3f}")
        else:
            print(f"{key:18s}   -")


def main():
    log(f"Device: {DEVICE}")
    log("=== Euclidean degeneration of the on-manifold neuron (K=0) ===")
    for key, ds_name, kw, epochs, bs in CONFIGS:
        ds = TUDataset(root=e37.DT_ROOT, name=ds_name)
        lbs = [ds[i].y.item() for i in range(len(ds))]
        log(f"=== {key} ({epochs} epochs, bs={bs}, h_dim=0, e_dim=32) ===")
        run_job(key, ds, lbs, kw, epochs, bs)
    log("=== ALL COMPLETE ===")
    status()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        status()
    else:
        main()
