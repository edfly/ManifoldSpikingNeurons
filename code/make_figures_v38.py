"""Regenerate v38 data figures from the 15-sample (5 seeds x 3 folds) JSONs.

Figures: main_results_bars, om_vs_ns, energy_accuracy_pareto.
(All methods now read from spikergnn_results/v38_5seeds; the ablation figure
is unchanged because the ablation sweep was run at 3 seeds and not rerun.)
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
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.linewidth": 0.8, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

R = _os.path.join(RESULTS_ROOT, "v38_5seeds")
FIG = FIGURES_ROOT
os.makedirs(FIG, exist_ok=True)

DATASETS = ["MUTAG", "NCI1", "PROTEINS", "DD"]
METHOD_KEYS = ["OM", "NS", "MSG", "SGC", "GIN", "GCN", "SAGE"]
METHOD_LABELS = ["OM", "No-Spikes", "MSG", "SGC", "GIN", "GCN", "SAGE"]
COLORS = {
    "OM":  "#A4262C", "NS": "#D77480", "MSG": "#C46A1B", "SGC": "#E5A23A",
    "GIN": "#1F5C9C", "GCN": "#3F86BC", "SAGE": "#7BB3DC",
}
BEST = {"MUTAG": "GIN", "NCI1": "GIN", "PROTEINS": "GCN", "DD": "NS"}


def accs(ds, mk):
    return np.array(json.load(open(f"{R}/{ds}_{mk}.json"))["accs"])


def sr_om(ds):
    return float(np.mean(json.load(open(f"{R}/{ds}_OM.json"))["srs"]))


# ---------------------------------------------------------------- main bars
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
for ax, ds in zip(axes.flat, DATASETS):
    means = [float(accs(ds, mk).mean()) for mk in METHOD_KEYS]
    x = np.arange(len(METHOD_KEYS))
    ax.bar(x, means, color=[COLORS[mk] for mk in METHOD_KEYS],
           width=0.72, edgecolor="white", linewidth=0.5)
    for i, (mk, mlabel, v) in enumerate(zip(METHOD_KEYS, METHOD_LABELS, means)):
        weight = "bold" if mlabel == BEST[ds] else "normal"
        ax.text(i, v + 1.0, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9.5, fontweight=weight)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10.5)
    ax.set_title(ds, fontsize=12, fontweight="bold", pad=6)
    ax.set_ylim(0, max(means) + 12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
fig.suptitle("Graph Classification Accuracy: On-Manifold vs. Baselines (5 seeds x 3 folds)",
             fontsize=14, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIG}/main_results_bars.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved main_results_bars.png")

# ---------------------------------------------------------------- om vs ns
fig, ax = plt.subplots(figsize=(5.6, 3.9))
x = np.arange(len(DATASETS))
om_m = [accs(ds, "OM").mean() for ds in DATASETS]
ns_m = [accs(ds, "NS").mean() for ds in DATASETS]
om_e = [accs(ds, "OM").std(ddof=0) for ds in DATASETS]
ns_e = [accs(ds, "NS").std(ddof=0) for ds in DATASETS]
ax.bar(x - 0.19, om_m, yerr=om_e, width=0.36, color="#A4262C",
       edgecolor="white", linewidth=0.5, capsize=3, label="OM (on-manifold)")
ax.bar(x + 0.19, ns_m, yerr=ns_e, width=0.36, color="#D77480",
       edgecolor="white", linewidth=0.5, capsize=3, label="No-Spikes (Euclidean)")
for xi, (a, b) in enumerate(zip(om_m, ns_m)):
    ax.text(xi - 0.19, a + 1.2, f"{a:.1f}", ha="center", va="bottom", fontsize=9)
    ax.text(xi + 0.19, b + 1.2, f"{b:.1f}", ha="center", va="bottom", fontsize=9)
    d = a - b
    ax.text(xi, max(a, b) + 5.5, f"$\\Delta$={d:+.1f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold",
            color="#1F5C9C" if d > 0.5 else "#444444")
ax.set_xticks(x)
ax.set_xticklabels(DATASETS, fontsize=10.5)
ax.set_ylabel("Accuracy (%)", fontsize=10.5)
ax.set_ylim(45, 85)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.set_title("On-Manifold vs. No-Spikes accuracy", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG}/om_vs_ns.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved om_vs_ns.png")

# ------------------------------------------------- energy-accuracy pareto
fig, ax = plt.subplots(figsize=(6.4, 4.4))
om_acc = [accs(ds, "OM").mean() for ds in DATASETS]
om_en = [sr_om(ds) * 0.9 / 4.6 * 100 for ds in DATASETS]
ann_acc = [accs(ds, "GIN").mean() for ds in DATASETS]
msg_acc = [accs(ds, "MSG").mean() for ds in DATASETS]
mk = ["o", "s", "D", "^"]
for i, ds in enumerate(DATASETS):
    ax.scatter(om_en[i], om_acc[i], marker=mk[i], s=70, color="#A4262C",
               edgecolors="white", linewidths=0.8, zorder=5)
    ax.scatter(100, ann_acc[i], marker=mk[i], s=55, facecolors="none",
               edgecolors="#1F5C9C", linewidths=1.4, zorder=4)
    ax.scatter(om_en[i], msg_acc[i], marker="x", s=55, color="#C46A1B",
               linewidths=1.6, zorder=3)
    ax.annotate(ds, (om_en[i], om_acc[i]), textcoords="offset points",
                xytext=(8, 4), fontsize=9, color="#A4262C")
    ax.annotate(ds, (100, ann_acc[i]), textcoords="offset points",
                xytext=(-8, 4), fontsize=9, color="#1F5C9C", ha="right")
    ax.annotate(ds, (om_en[i], msg_acc[i]), textcoords="offset points",
                xytext=(8, -3), fontsize=9, color="#C46A1B")
ax.scatter([], [], marker="o", s=70, color="#A4262C", label="OM (SNN)")
ax.scatter([], [], marker="o", s=55, facecolors="none", edgecolors="#1F5C9C",
           label="GIN (ANN, 100% energy)")
ax.scatter([], [], marker="x", s=55, color="#C46A1B",
           label="MSG (non-functional)")
ax.set_xlabel("Relative energy $E_{\\mathrm{SNN}}/E_{\\mathrm{ANN}}$ (%)", fontsize=10.5)
ax.set_ylabel("Accuracy (%)", fontsize=10.5)
ax.set_xlim(-4, 118)
lo = min(msg_acc) - 4
ax.set_ylim(lo, max(ann_acc) + 6)
ax.grid(alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.set_title("Energy--accuracy trade-off (45 nm model)", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG}/energy_accuracy_pareto.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved energy_accuracy_pareto.png")

print("\nOM 15-sample means:", {ds: f"{accs(ds,'OM').mean():.1f}" for ds in DATASETS})
print("OM sr:", {ds: f"{sr_om(ds):.3f}" for ds in DATASETS})
