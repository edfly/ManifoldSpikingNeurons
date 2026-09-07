"""Figure for Section 5.7: the mechanism-matched factorial.

Plots the four cells of Design B (GIN, HGCN-32, OM-E, OM) on each dataset and
annotates the OM - OM-E gap, which is the clean estimate of what the curvature
contributes under spiking. The visual point of the figure is that the last two
bars are nearly the same height.

Usage:
    python make_figure_factorial_v38.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "spikergnn_results/v38_5seeds"
RH32 = "spikergnn_results/v38_hgcn32"
REUC = "spikergnn_results/v38_euclid"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

DATASETS = ["MUTAG", "NCI1", "PROTEINS", "DD"]

# Euclidean = warm greys, manifold = saturated; ANN = blue, spiking = red.
CELLS = [
    ("GIN", "GIN\n(Euc, ANN)", "#9BB7D4", lambda ds: f"{R}/{ds}_GIN.json"),
    ("HGCN-32", "HGCN-32\n(Man, ANN)", "#1F5C9C",
     lambda ds: f"{RH32}/{ds}_HGCN.json"),
    ("OM-E", "OM-E\n(Euc, spiking)", "#E8A0A8",
     lambda ds: f"{REUC}/{ds}_EUCLID.json"),
    ("OM", "OM\n(Man, spiking)", "#A4262C", lambda ds: f"{R}/{ds}_OM.json"),
]


def accs(path):
    with open(path) as f:
        return np.array(json.load(f)["accs"], dtype=float)


fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9))
for ax, ds in zip(axes, DATASETS):
    vals = [accs(fn(ds)) for _, _, _, fn in CELLS]
    means = [v.mean() for v in vals]
    sds = [v.std(ddof=0) for v in vals]
    x = np.arange(len(CELLS))
    ax.bar(x, means, yerr=sds, color=[c for _, _, c, _ in CELLS], width=0.68,
           capsize=3, edgecolor="white", linewidth=0.6,
           error_kw={"elinewidth": 0.9, "ecolor": "#444444"})
    for i, v in enumerate(means):
        ax.text(i, v + 1.4, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
    # annotate the mechanism-matched contrast (OM - OM-E)
    gap = means[3] - means[2]
    top = max(means[2], means[3]) + max(sds[2], sds[3])
    ax.plot([2, 3], [top + 4.2, top + 4.2], color="#333333", lw=1.0)
    ax.text(2.5, top + 5.2, f"OM$-$OM-E = {gap:+.1f}", ha="center",
            va="bottom", fontsize=9, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _, _ in CELLS], fontsize=9)
    ax.set_title(ds, fontsize=12, fontweight="bold", pad=6)
    ax.set_ylim(0, max(means) + max(sds) + 16)
    ax.set_ylabel("Accuracy (%)" if ds == DATASETS[0] else "", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

fig.suptitle("Design B (mechanism-matched): the last two bars differ by at "
             "most 1.3 points", fontsize=11.5, y=1.03)
fig.tight_layout()
out = f"{FIG}/factorial_designB.png"
fig.savefig(out, bbox_inches="tight", dpi=300)
print(f"wrote {out}")
