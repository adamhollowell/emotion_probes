"""GGE1 draft v2 — Figure 4 (§4.2 + §4.3): depth curves with seams.

Median own-word rank by fractional depth, six cells, log y, chance line;
each cell's consolidation seam (§4.3) marked on its own curve so the
27B-only coincidence of seam and readability onset is visible.

Seam values: seam_gap_metric_v1.json, with G6.4's canonical-basis
adoption of L23 for BOTH 27B cells (coordinator Flag-A ruling: the
artifact's ledger_reference_mismatch is documented supersession
bookkeeping; resample CI [23, 24] is a caption obligation).

Sources: geometry_out/<cell>/lens_summary.json,
         geometry_out/seam_gap_metric_v1.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gge1_style as G
import matplotlib.pyplot as plt

G.apply()
CHANCE = 128000  # V/2, V = 256,000

curves, seams = {}, {}
sg = json.load(open(os.path.join(G.T100, "seam_gap_metric_v1.json")))["cells"]
for c in G.CELLS:
    d = json.load(open(os.path.join(G.T100, c, "lens_summary.json")))
    nl = d["n_layers"]
    ranks = {p["layer"]: p["median_own_rank"]
             for p in d["per_layer_median_own_rank"]}
    seam_layer = sg[c]["consolidation_layer"]
    if c == "gemma-2-27b-it":
        seam_layer = 23  # G6.4 canonical adoption (supersession of L24)
    curves[c] = {"frac_depth": [l / nl for l in sorted(ranks)],
                 "median_own_rank": [ranks[l] for l in sorted(ranks)],
                 "n_layers": nl,
                 "onset_layer": d["onset_layer_median_lt_10k"]}
    seams[c] = {"layer": seam_layer, "frac": seam_layer / nl,
                "rank_at_seam": ranks[seam_layer]}

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.axhline(CHANCE, color=G.GREY, linestyle=":", linewidth=0.9, zorder=1)
ax.text(0.008, CHANCE * 1.28, "chance (V/2 = 128,000)", fontsize=G.ANNOT_FS,
        color=G.ANNOT_GREY, va="bottom")
# readability criterion (onset = first layer with median rank < 10,000):
# dotted like the chance line so reference lines never read as tuned
# curves (which are dashed); labeled in-figure so the caption need not
ONSET_RANK = 10000
ax.axhline(ONSET_RANK, color=G.GREY, linestyle=":", linewidth=0.9, zorder=1)
ax.text(0.995, ONSET_RANK * 1.18, "readability threshold (rank 10,000)",
        fontsize=G.ANNOT_FS, color=G.ANNOT_GREY, va="bottom", ha="right")
for c in G.CELLS:
    st = G.cell_style(c)
    ax.plot(curves[c]["frac_depth"], curves[c]["median_own_rank"],
            color=st["color"], linestyle=st["linestyle"], linewidth=1.0,
            marker=st["marker"], markersize=2.4,
            markerfacecolor=st["markerfacecolor"],
            markeredgecolor=st["markeredgecolor"], markeredgewidth=0.7,
            label=G.cell_label(c), zorder=3)
    # consolidation seam: diamond on the curve at the seam layer (REVIEW
    # _100T_ROUND1 fix: vertical dashes merged where seams coincide at 2B
    # and 27B). Fill follows the global tuning rule (base filled, IT open)
    # so coincident pairs read as two marks; white halo for contrast.
    m = G.CELL_META[c]
    ax.plot(seams[c]["frac"], seams[c]["rank_at_seam"], marker="D",
            markersize=7.5, markerfacecolor="white",
            markeredgecolor="white", markeredgewidth=2.0, zorder=4)
    ax.plot(seams[c]["frac"], seams[c]["rank_at_seam"], marker="D",
            markersize=5.0,
            markerfacecolor=(st["color"] if m["tuning"] == "base" else "white"),
            markeredgecolor=st["color"], markeredgewidth=1.1, zorder=5)

# one neutral explainer for the seam markers (encoding, not a claim);
# fill follows the same base/IT rule as the curve markers
ax.plot([], [], marker="D", markersize=4.6, linestyle="none",
        markerfacecolor="#555555", markeredgecolor="#555555",
        label="consolidation seam")
ax.set_yscale("log")
ax.set_xlim(0, 1.0)
ax.set_xlabel("Fractional depth (layer / total layers)")
ax.set_ylabel("Median own-word self-rank")
handles, labels = ax.get_legend_handles_labels()
fig.subplots_adjust(bottom=0.24)
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.005),
           ncol=4, frameon=False, fontsize=6.5, columnspacing=0.9,
           handlelength=1.6, handletextpad=0.4)
G.save(fig, "gge1_v3_fig4_depth_seams",
       {"curves": curves, "seams": seams, "chance": CHANCE,
        "note": "27b-it seam L23 per G6.4 canonical adoption "
                "(artifact ledger_reference_mismatch = documented "
                "supersession); resample CI [23,24] both 27B cells — "
                "caption obligation"},
       ["geometry_out/<cell>/lens_summary.json",
        "geometry_out/seam_gap_metric_v1.json",
        "Ledger G6.4 (L23 adoption, both 27B cells)"])
print("seams:", {c: seams[c]["layer"] for c in G.CELLS})
print("onsets:", {c: curves[c]["onset_layer"] for c in G.CELLS})
