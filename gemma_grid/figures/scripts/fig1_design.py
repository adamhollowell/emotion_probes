"""GGE1 draft v3 — Figure 1 (§2.1): the grid, as a design schematic.

Round 3 (MEMO_TO_FIGURES_FIG1_ROUND3_100T.md): stripped per Adam.
REMOVED: the corpus box, the "rounded ⅔ of depth" annotation, the anchor
line, and the connecting line through the marks (a rising line reads as a
trend, which this figure must not suggest). Kept, and nothing else: three
column pairs at TRUE heights (26/42/46 layers, layer 0 at the bottom),
base solid / instruction-tuned dashed, size + layer count above each,
analysis-layer marks labelled L17/L28/L31, the layer-0 baseline, the
checkpoint identifiers below, and the tuning legend. Compact for a
two-column layout. The caption (drafting side) carries the two-thirds
rule and the anchor identification.

Carries NO measured quantity — checkpoint ids, layer counts, and analysis
layers only (§6.4 / release manifest). Size colors are the set-wide
color-equals-size encoding of Figures 2 to 4.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gge1_style as G
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

G.apply()

CKPT = {
    "gemma-2-2b-base": "google/gemma-2-2b",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "gemma-2-9b-base": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-2-27b-base": "google/gemma-2-27b",
    "gemma-2-27b-it": "google/gemma-2-27b-it",
}
SIZES = ["2b", "9b", "27b"]

fig, ax = plt.subplots(figsize=(3.8, 2.55))  # widened so the ids print legibly (fix 3)
# axes fill the canvas: with default subplot margins only ~78% of the
# figure width carries the 9 data units, which silently shrank the
# inter-column spacing and made the id rows collide at print sizes
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, 9.0)
ax.set_ylim(0, 6.9)
ax.axis("off")

BASE_Y = 1.85            # y of layer 0
K = 0.082                # height units per layer -> 27B column = 3.77
CX = {"2b": 1.60, "9b": 4.50, "27b": 7.40}   # column centers
BW, GAP = 0.62, 0.18     # bar width, gap within the pair

for s in SIZES:
    m = G.CELL_META[f"gemma-2-{s}-base"]
    h = m["n_layers"] * K
    col = G.SIZE_COLORS[s]
    xb = CX[s] - GAP / 2 - BW          # base bar (solid edge)
    xi = CX[s] + GAP / 2               # IT bar (dashed edge)
    for x0, ls in ((xb, "-"), (xi, "--")):
        ax.add_patch(Rectangle((x0, BASE_Y), BW, h, facecolor="white",
                               edgecolor=col, linewidth=1.1, linestyle=ls,
                               zorder=3))
    # analysis-layer mark across the pair, labelled
    ym = BASE_Y + (m["layer"] / m["n_layers"]) * h
    ax.plot([xb - 0.09, xi + BW + 0.09], [ym, ym], color=col,
            linewidth=1.7, zorder=5, solid_capstyle="butt")
    ax.text(xi + BW + 0.16, ym, f"L{m['layer']}", ha="left", va="center",
            fontsize=6.0, color=col)
    # size + layer count above the column
    ax.text(CX[s], BASE_Y + h + 0.42, s.upper(), ha="center", va="bottom",
            fontsize=8.5, color=col, fontweight="semibold")
    ax.text(CX[s], BASE_Y + h + 0.10, f"{m['n_layers']} layers",
            ha="center", va="bottom", fontsize=5.8, color="#777777")

# layer-0 baseline
ax.plot([0.45, 8.75], [BASE_Y, BASE_Y], color="#666666", linewidth=0.8,
        zorder=1)
ax.text(0.40, BASE_Y - 0.12, "layer 0", ha="left", va="top", fontsize=5.6,
        color="#666666")

# checkpoint identifiers below (tuning keyed by legend and -it suffix)
for s in SIZES:
    col = G.SIZE_COLORS[s]
    ax.text(CX[s], 1.20, CKPT[f"gemma-2-{s}-base"],
            ha="center", va="center", fontsize=6.0, color=col,
            family="monospace")
    ax.text(CX[s], 0.84, CKPT[f"gemma-2-{s}-it"],
            ha="center", va="center", fontsize=6.0, color=col,
            family="monospace")

# legend: tuning edges (matches Figures 2 to 4) + the mark's meaning
ax.plot([0.95, 1.40], [0.28, 0.28], color="#666666", linewidth=1.0, ls="-")
ax.text(1.50, 0.28, "base", ha="left", va="center", fontsize=5.6,
        color="#666666")
ax.plot([2.50, 2.95], [0.28, 0.28], color="#666666", linewidth=1.0, ls="--")
ax.text(3.05, 0.28, "tuned", ha="left", va="center",
        fontsize=5.6, color="#666666")  # paper term (fix 4)
ax.plot([4.55, 5.00], [0.28, 0.28], color="#666666", linewidth=1.7,
        solid_capstyle="butt")
ax.text(5.10, 0.28, "analysis layer", ha="left", va="center",
        fontsize=5.6, color="#666666")

G.save(fig, "gge1_v3_fig1_design",
       {"checkpoints": CKPT,
        "n_layers": {c: G.CELL_META[c]["n_layers"] for c in G.CELLS},
        "analysis_layers": {c: G.CELL_META[c]["layer"] for c in G.CELLS},
        "note": "round-3 stripped design (memo 2026-09-09): columns at "
                "true heights, labelled analysis marks, baseline, ids, "
                "legend — nothing else. Corpus box, two-thirds "
                "annotation, anchor line, and connecting line removed; "
                "the caption carries the two-thirds rule and the anchor "
                "identification. Compact 3.4in for two-column layout."},
       ["§6.4 / release manifest facts only (checkpoint ids, layer "
        "counts, analysis layers); verified against CELL_META, gated "
        "against lens_summary.json n_layers and the stageA layer column"])
print("marks:", {s: f"L{G.CELL_META[f'gemma-2-{s}-base']['layer']} of "
                    f"{G.CELL_META[f'gemma-2-{s}-base']['n_layers']}"
                 for s in SIZES})
