"""Verify the paper's hyperparameter table against the ACTUAL runtime values.

Motivation: the v38 audit found that the paper declared ATan surrogate
steepness beta_surr=2 while the on-manifold neuron actually ran with beta=5.0
(the ManifoldLIFNeuron default, never overridden by OnManifoldGNN). Static
reading of the source would likely have missed it again, so this script reads
the values by *instantiating* the models and *reflecting* on them, then
compares against the numbers literally printed in Appendix B of the paper.

Usage:
    python verify_hyperparams.py            # check and report
    python verify_hyperparams.py --emit     # also print a LaTeX table fragment
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
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEX = PAPER_TEX

# ── runtime reflection: what the code ACTUALLY uses ──────────────────────


def runtime_values():
    """Instantiate every model and read the effective hyperparameter values."""
    import experiment_v37 as e37              # OnManifoldGNN + LAMBDA_SR
    import experiment_v38_5seeds as e5        # SpikeGCN, MSGSupervised
    import experiment_v38_s2 as es2           # SphereLIFNeuron (supplementary)

    vals = {}

    # ---- on-manifold neuron (main method) ----
    m = e37.OnManifoldGNN(in_dim=7, n_class=2, h_dim=16, e_dim=16,
                          hidden=64, L=2, T=5, c=1.0)
    nr = m.neurons[0]
    vals["Membrane decay alpha"] = float(nr.alpha)
    vals["Leak beta (init)"] = float(nr.beta.detach())
    vals["Input gain gamma (init)"] = float(nr.gamma.detach())
    vals["Threshold u_th (init)"] = float(nr.u_th_learnable.detach())
    vals["Spike interpolation sigma"] = float(torch.sigmoid(nr.spike_alpha).detach())
    vals["Reference EMA decay"] = float(nr.p_bar_decay)
    vals["Input magnitude cap"] = float(nr.input_mag_cap)
    vals["Surrogate steepness (OM)"] = float(getattr(nr.surrogate, "beta", float("nan")))
    vals["lambda_sr"] = float(e37.LAMBDA_SR)
    vals["r_target"] = float(e37.R_TARGET)

    # ---- SGC (SpikeGCN) baseline ----
    s = e5.SpikeGCN(in_dim=7, n_class=2)
    vals["SGC T"] = float(s.T)
    vals["SGC tau"] = round(1.0 / (1.0 - float(s.decay)), 6)
    vals["Surrogate steepness (SGC)"] = float(getattr(s.surrogate, "beta", float("nan")))

    # ---- MSG baseline ----
    g = e5.MSGSupervised(in_dim=7, n_class=2)
    enc = g.enc
    layer = enc.layers[0]
    vals["MSG hidden"] = float(enc.hidden_dim)
    vals["MSG layers"] = float(enc.num_layers)
    vals["MSG epsilon"] = float(layer.epsilon)
    vals["MSG tau"] = round(1.0 / (1.0 - float(layer.if_neuron.decay)), 6)
    # T=8 is hard-coded inside MSNeuronLayer.forward
    import inspect
    src = inspect.getsource(type(layer).forward)
    mt = re.search(r"T\s*=\s*(\d+)", src)
    vals["MSG T"] = float(mt.group(1)) if mt else float("nan")

    return vals


# ── paper side: parse Appendix B table ───────────────────────────────────


def paper_values():
    """Extract (param-name -> set of numeric literals) from tab:hyper."""
    text = open(TEX, encoding="utf-8").read()
    blk = re.search(r"\\label\{tab:hyper\}.*?\\bottomrule", text, re.S)
    if not blk:
        return {}
    out = {}
    for line in blk.group(0).split("\n"):
        m = re.match(r"\s*(?:\\multirow\{(\d+)\}\{[^}]*\}\{([^}]*)\}\s*)?"
                     r"&\s*(.+?)\s*&\s*(.+?)\s*\\\\", line)
        if not m:
            continue
        name = re.sub(r"\\+", "", m.group(3))
        name = re.sub(r"[\$\{\}\\]", "", name).strip()
        nums = set()
        for tok in re.findall(r"\d+\.?\d*", m.group(4)):
            try:
                nums.add(float(tok))
            except ValueError:
                pass
        out[name] = nums
    return out


# ── comparison ───────────────────────────────────────────────────────────

# runtime key -> list of substrings identifying the corresponding paper row
MATCH = {
    "Membrane decay alpha": ["Membrane decay"],
    "Leak beta (init)": ["Leak"],
    "Input gain gamma (init)": ["Input gain"],
    "Threshold u_th (init)": ["Threshold"],
    "Spike interpolation sigma": ["Spike interpolation"],
    "Reference EMA decay": ["Reference-point EMA"],
    "Input magnitude cap": ["Input magnitude cap"],
    "Surrogate steepness (OM)": ["Surrogate steepness"],
    "lambda_sr": ["mathcalL_sr"],
    "r_target": ["mathcalL_sr"],
    "SGC T": ["SGC (SpikeGCN)"],
    "SGC tau": ["SGC (SpikeGCN)"],
    "Surrogate steepness (SGC)": ["SGC (SpikeGCN)"],
    "MSG hidden": ["MSG"],
    "MSG layers": ["MSG"],
    "MSG epsilon": ["MSG"],
    "MSG tau": ["MSG"],
    "MSG T": ["MSG"],
}

# relative tolerance: the paper reports rounded values (e.g. sigma(0.5)=0.62
# vs runtime 0.6225), so allow 1% rather than exact equality.
RTOL = 0.01


def main():
    rt = runtime_values()
    pv = paper_values()
    print("=" * 70)
    print("HYPERPARAMETER DRIFT CHECK  (runtime code  vs  Appendix B of paper)")
    print("=" * 70)
    print(f"{'parameter':32s} {'runtime':>10s}  {'in paper?':>9s}  paper row")
    print("-" * 70)
    problems = []
    for key, val in rt.items():
        cands = MATCH.get(key, [key])
        row_name, row_nums = None, set()
        for c in cands:
            for name, nums in pv.items():
                if c.lower() in name.lower():
                    row_name, row_nums = name, nums
                    break
            if row_name:
                break
        ok = row_name is not None and any(
            abs(val - n) <= max(RTOL * max(abs(val), 1e-9), 1e-6)
            for n in row_nums)
        flag = "OK" if ok else "**DRIFT**"
        print(f"{key:32s} {val:10.4f}  {flag:>9s}  "
              f"{(row_name or '(row not found)')[:40]}")
        if not ok:
            problems.append((key, val, row_name, sorted(row_nums)))

    print("-" * 70)
    if problems:
        print(f"\n{len(problems)} DRIFT(S) DETECTED:")
        for key, val, row, nums in problems:
            print(f"  - {key}: runtime={val}, paper row '{row}' "
                  f"declares {nums if nums else '(none)'}")
        print("\nAction: update Appendix B (tab:hyper) to match the code, or "
              "change the code and rerun the affected experiments.")
        return 1
    print("\nALL HYPERPARAMETERS MATCH — Appendix B is consistent with the code.")

    if "--emit" in sys.argv:
        print("\n% ---- LaTeX fragment (regenerate Appendix B rows) ----")
        print(r"\begin{tabular}{@{}llc@{}}")
        print(r"\toprule")
        print(r"\textbf{Parameter} & \textbf{Runtime value} \\")
        print(r"\midrule")
        for key, val in rt.items():
            print(f"{key} & {val:g} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
