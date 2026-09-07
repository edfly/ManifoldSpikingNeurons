"""# =====================================================================
#  DEPRECATED.  The TikZ source in graphical_abstract.tex (at the
#  repository root) is the canonical graphical abstract.  This
#  matplotlib generator produced an earlier raster draft on
#  2024-09-03 and is kept only for one-way comparison; it is no
#  longer regenerated and no longer matches the submission asset.
# =====================================================================

Graphical abstract for the v38 manuscript.

Design notes (submitted to Neurocomputing):
  * No title inside the image -- the submission system collects the title and
    the caption separately, so an in-figure title would duplicate it.
  * Layout is a fixed three-block grid built with add_axes, not a free-form
    scatter of annotations. Every text element sits at an explicit coordinate
    and no two elements share a y-band, which is what keeps labels from
    colliding.
  * Generous gaps: blocks are separated by >=3% of the canvas and bar labels
    are placed above the bar top with headroom reserved in the y-limits.

Usage:
    python make_graphical_abstract_v38.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

R = "spikergnn_results"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

DS = ["MUTAG", "NCI1", "PROTEINS", "DD"]


def mean_acc(path):
    with open(path) as f:
        return float(np.mean(json.load(f)["accs"]))


OM = np.array([mean_acc(f"{R}/v38_5seeds/{d}_OM.json") for d in DS])
OME = np.array([mean_acc(f"{R}/v38_euclid/{d}_EUCLID.json") for d in DS])
MSG = np.array([mean_acc(f"{R}/v38_5seeds/{d}_MSG.json") for d in DS])
SGC = np.array([mean_acc(f"{R}/v38_5seeds/{d}_SGC.json") for d in DS])

C_OM, C_OME = "#A4262C", "#E8A0A8"
C_MSG, C_SGC = "#C46A1B", "#E5A23A"
C_BOX, C_EDGE = "#EEF3F8", "#1F5C9C"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

fig = plt.figure(figsize=(12.6, 5.0))

# ------------------------------------------------------------ block 1: method
ax1 = fig.add_axes([0.045, 0.16, 0.50, 0.70])
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis("off")

steps = [
    ("Input", "graph"),
    ("Product", r"manifold $\mathbb{H}^{d}\!\times\!\mathbb{R}^{d}$"),
    ("On-manifold", "LIF, $T$ steps"),
    ("Spike-gated", "readout"),
]
bw, gap = 0.205, 0.060           # box width, arrow gap  -> 4*bw + 3*gap = 1.0
for i, (l1, l2) in enumerate(steps):
    x0 = 0.0 + i * (bw + gap)
    ax1.add_patch(mpatches.FancyBboxPatch(
        (x0, 0.34), bw, 0.32, boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor=C_BOX, edgecolor=C_EDGE, linewidth=1.3, zorder=2))
    ax1.text(x0 + bw / 2, 0.575, l1, ha="center", va="center",
             fontsize=10.5, zorder=3)
    ax1.text(x0 + bw / 2, 0.445, l2, ha="center", va="center",
             fontsize=9.5, zorder=3)
    if i < 3:
        ax1.annotate("", xy=(x0 + bw + gap - 0.008, 0.50),
                     xytext=(x0 + bw + 0.008, 0.50),
                     arrowprops=dict(arrowstyle="-|>", color=C_EDGE,
                                     linewidth=1.5), zorder=2)

# mechanism caption sits in its own band well below the boxes
ax1.text(0.5, 0.145, "spike-encoded state never leaves the manifold:",
         ha="center", va="center", fontsize=10, style="italic")
ax1.text(0.5, 0.045,
         r"$\exp/\log$ maps  $\cdot$  learnable threshold  $\cdot$  "
         r"$L_{sr}$ rate regulariser",
         ha="center", va="center", fontsize=10)

# ------------------------------------------------------- block 2: vs baselines
ax2 = fig.add_axes([0.615, 0.545, 0.355, 0.335])
labels = ["OM", "MSG", "SGC"]
means = [OM.mean(), MSG.mean(), SGC.mean()]
sds = [OM.std(ddof=0), MSG.std(ddof=0), SGC.std(ddof=0)]
bars = ax2.bar(labels, means, yerr=sds, color=[C_OM, C_MSG, C_SGC], width=0.60,
               capsize=3, edgecolor="white", linewidth=0.8,
               error_kw=dict(elinewidth=0.9, ecolor="#444444"))
for b, v in zip(bars, means):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.1f}",
             ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax2.set_ylim(0, 84)
ax2.set_ylabel("Accuracy (%)", fontsize=9.5)
ax2.set_title("Four-benchmark mean - OM vs. MSG:  +10.7",
              fontsize=10.5, pad=6, color=C_OM, fontweight="bold")
ax2.tick_params(labelsize=9.5)
ax2.grid(axis="y", alpha=0.22, linestyle=":")
ax2.set_axisbelow(True)
# The highlight goes INSIDE the axes. Placed outside at y>1 it reached up into
# the panel above and collided with that panel's x tick labels.


# ------------------------------------------------- block 3: mechanism vs curve
ax3 = fig.add_axes([0.615, 0.115, 0.355, 0.335])
labels2 = ["OM", "OM-E"]
means2 = [OM.mean(), OME.mean()]
sds2 = [OM.std(ddof=0), OME.std(ddof=0)]
bars2 = ax3.bar(labels2, means2, yerr=sds2, color=[C_OM, C_OME], width=0.45,
                capsize=3, edgecolor="white", linewidth=0.8,
                error_kw=dict(elinewidth=0.9, ecolor="#444444"))
for b, v in zip(bars2, means2):
    ax3.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.1f}",
             ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax3.set_ylim(0, 84)
ax3.set_ylabel("Accuracy (%)", fontsize=9.5)
ax3.set_title("Curvature removed - OM vs. OM-E:  -0.3 (n.s.)",
              fontsize=10.5, pad=6, color=C_OM, fontweight="bold")
ax3.tick_params(labelsize=9.5)
ax3.grid(axis="y", alpha=0.22, linestyle=":")
ax3.set_axisbelow(True)


out = f"{FIG}/graphical_abstract.png"
fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")

# ------------------------------------------------- collision check (no overlap)
# "Looks fine" is not a check. Render the figure, take the bounding box of every
# text object, and verify that no two boxes overlap by more than a hairline.
# This catches the usual failure: a value label drifting into a title, or a
# mechanism caption colliding with the boxes above it.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
import matplotlib.text as mtext

texts = [t for t in fig.findobj(mtext.Text) if t.get_text().strip()]
boxes = [(t.get_text().replace("\n", " ")[:42], t.get_window_extent(renderer))
         for t in texts]

TOL = 2.0  # px; sub-pixel touching from hinting is not a real collision
bad = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        (ta, ba), (tb, bb) = boxes[i], boxes[j]
        ox = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
        oy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
        if ox > TOL and oy > TOL:
            bad.append((ta, tb, ox, oy))

if bad:
    print(f"  [FAIL] {len(bad)} overlapping text pair(s):")
    for ta, tb, ox, oy in bad:
        print(f"    {ox:.1f}x{oy:.1f} px  '{ta}' <> '{tb}'")
else:
    print(f"  [OK] no text overlap among {len(boxes)} text elements")

# also confirm everything landed inside the saved image
from PIL import Image
im = Image.open(out)
print(f"  image: {im.size[0]}x{im.size[1]} px, "
      f"{os.path.getsize(out)/1024:.0f} KB")

print(f"  OM={OM.mean():.2f}  OM-E={OME.mean():.2f}  "
      f"MSG={MSG.mean():.2f}  SGC={SGC.mean():.2f}")
print(f"  OM-MSG={OM.mean()-MSG.mean():+.2f}  OM-OME={OM.mean()-OME.mean():+.2f}")
