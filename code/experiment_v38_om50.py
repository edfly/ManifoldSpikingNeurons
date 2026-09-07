"""v38 supplementary: OM under a MATCHED training budget (50 epochs).

Why this exists
---------------
The main table compares OM against the ANN baselines under *unequal* training
budgets, because experiment_v37 gave each dataset its own budget while the
baseline runner (experiment_v38_5seeds.py) used a uniform 50 epochs:

    MUTAG     OM 50 ep / bs16   vs baselines 50 ep / bs16   (equal)
    NCI1      OM 10 ep / bs32   vs baselines 50 ep / bs16   (OM: 1/5)
    PROTEINS  OM 25 ep / bs16   vs baselines 50 ep / bs16   (OM: 1/2)
    DD        OM 25 ep / bs16   vs baselines 50 ep / bs16   (OM: 1/2)

The bias runs AGAINST our method, but it must be disclosed and quantified.
This script reruns OM at 50 epochs on the three affected datasets so the
comparison can be reported both ways. MUTAG already used 50 epochs and is
included only for completeness (it is read from the cached JSON).

Everything else (model, optimiser, L_sr, seeds, folds, stability policy) is
identical to experiment_v37 so that the only changed variable is the budget.

Usage:
    python experiment_v38_om50.py
    python experiment_v38_om50.py status
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset

import experiment_v37 as e37

OUT = "spikergnn_results/v38_om50"
os.makedirs(OUT, exist_ok=True)
NF = 3
SEEDS5 = [42, 123, 456, 789, 2024]
DEVICE = e37.DEVICE

# same architecture/optimisation as v37, budget raised to the baseline's
MATCHED = [
    # key,              dataset,   model kwargs,        epochs, bs
    ("NCI1_OM50",      "NCI1",     dict(L=2, T=5),      50, 32),
    ("PROTEINS_OM50",  "PROTEINS", dict(L=2, T=10),     50, 16),
    ("DD_OM50",        "DD",       dict(L=2, T=10),     50, 16),
]

log = e37.log


def run_job(key, ds, lbs, kw, epochs, bs):
    fp = os.path.join(OUT, f"{key}.json")
    total = NF * len(SEEDS5)
    accs, srs, uths = [], [], []
    if os.path.exists(fp):
        d = json.load(open(fp))
        accs = d.get("accs", [])
        srs = d.get("srs", [])
        uths = d.get("uths", [])
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
             "note": ("OM under the ANN baselines' budget (50 epochs); "
                      "identical model/schedule to experiment_v37")}
        json.dump(d, open(fp, "w"), indent=2)
        log(f"    {key} s{seed} f{idx % NF + 1}: acc={a:.1f}% sr={sr:.3f} "
            f"u_th={uth[0]:.3f} ({idx + 1}/{total}, {time.time() - t0:.0f}s)")
    return d


def status():
    print(f"\n{'key':16s} {'n':>3s} {'mean':>6s} {'std':>5s} {'sr':>5s}")
    for key, *_ in MATCHED:
        fp = os.path.join(OUT, f"{key}.json")
        if os.path.exists(fp):
            d = json.load(open(fp))
            sr = np.mean(d.get("srs", [0]))
            print(f"{key:16s} {len(d['accs']):3d} {d['mean']:6.1f} "
                  f"{d['std']:5.1f} {sr:5.3f}")
        else:
            print(f"{key:16s}   -")


def main():
    log(f"Device: {DEVICE}")
    log("=== OM under matched budget (50 epochs) ===")
    for key, ds_name, kw, epochs, bs in MATCHED:
        ds = TUDataset(root=e37.DT_ROOT, name=ds_name)
        lbs = [ds[i].y.item() for i in range(len(ds))]
        log(f"=== {key} ({epochs} epochs, bs={bs}) ===")
        run_job(key, ds, lbs, kw, epochs, bs)
    log("=== ALL COMPLETE ===")
    status()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        status()
    else:
        main()
