"""GGE1 draft v2 — Figure 2 (§2.4): transport summary.

Per COORDINATOR_RULINGS ruling 1 (option a, ratified): no per-pair rank
vectors exist on disk, so the figure shows the sixteen per-pair transport
summaries (top-1 accuracy; median held-out rank is 1.0 for every pair)
against the frozen 10,000-shuffle permutation floor, with the primary pair
(2B base -> 27B base, the grid's largest scale gap) distinguished.

Sources (read, never typed):
  geometry_out/pmap_transport.json        (per-pair summaries)
  geometry_out/pmap_transport_floor.json  (floor distribution, p)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gge1_style as G
import matplotlib.pyplot as plt

G.apply()

d = json.load(open(os.path.join(G.T100, "pmap_transport.json")))
fl = json.load(open(os.path.join(G.T100, "pmap_transport_floor.json")))
pairs = d["pairs"]
PRIMARY = fl["primary_pair"]  # gemma-2-2b-base->gemma-2-27b-base


def pretty(cell):
    # "tuned", not "IT" (figures match the paper's term, 2026-10-20)
    m = G.CELL_META[cell]
    return f"{m['size'].upper()} {'base' if m['tuning'] == 'base' else 'tuned'}"


def pair_label(key):
    a, b = key.split("->")
    return f"{pretty(a)} → {pretty(b)}"


# stable display order: artifact order, primary pair first
keys = list(pairs)
keys.remove(PRIMARY)
keys = [PRIMARY] + keys

# 2026-10-25 legibility pass (outside layout review, preview p. 8): the
# figure carries ~100 labels, so type is raised to body-adjacent sizes
# and the frame grown in height only; width must stay <= 5.5 in
# (397 pt) or LaTeX scales it down. Annotation grey darkened for paper.
LABEL_FS = 8.0
ANNOT_FS = 7.5
ANNOT_COL = "#4D4D4D"
fig, ax = plt.subplots(figsize=(5.15, 4.4))
n = len(keys)
ys = list(range(n, 0, -1))

# permutation floor band (shuffled top-1 distribution, min..max)
sh = fl["shuffled_top1_distribution"]
ax.axvspan(sh["min"], sh["max"], color=G.BAND_GREY, zorder=0)
# below the bottom row's gridline so no connector cuts through the text
ax.text(sh["max"] + 0.012, 0.28,
        "permutation floor (10,000 shuffles)",
        fontsize=ANNOT_FS, color=ANNOT_COL, va="center", ha="left")

# make the floor-to-ceiling gap load-bearing (Adam, 2026-09-06): a labeled
# double-headed span through the empty mid-plot, factual endpoints only
top1_min = min(v["top1_acc"] for v in pairs.values())
gap_y = 8.5
ax.annotate("", xy=(sh["max"], gap_y), xytext=(top1_min - 0.006, gap_y),
            arrowprops=dict(arrowstyle="<|-|>", color="#6E6E6E",
                            linewidth=1.0, mutation_scale=10))
ax.text((sh["max"] + top1_min) / 2, gap_y + 0.45,
        (f"shuffled maps ≤ {sh['max']:.3f}   ·   "
         f"fitted maps ≥ {top1_min:.3f}").replace("0.", "."),
        fontsize=ANNOT_FS, color=ANNOT_COL, ha="center", va="bottom")

for key, y in zip(keys, ys):
    v = pairs[key]
    primary = key == PRIMARY
    ax.plot([0, v["top1_acc"]], [y, y], color="#E3E3E3", linewidth=0.7,
            zorder=1)
    ax.plot(v["top1_acc"], y,
            marker="^" if primary else "o",
            markersize=6.5 if primary else 4.8,
            color=G.SIZE_COLORS["27b"] if primary else "#5B7A94",
            zorder=3)

ax.set_yticks(ys)
ax.set_yticklabels([pair_label(k) for k in keys], fontsize=LABEL_FS,
                   color="#333333")
# primary row label slightly emphasized
ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_xlim(0, 1.04)
ax.set_ylim(-0.35, n + 0.7)
ax.set_xlabel("Held-out retrieval, top-1 accuracy", fontsize=8.5)
ax.tick_params(axis="x", labelsize=7.5)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

# right-side facts, neutral wording (claims live in the caption)
ax.text(1.02, n + 0.45, "median held-out rank = 1.0, all pairs",
        fontsize=ANNOT_FS, color=ANNOT_COL, ha="right", va="bottom")

G.save(fig, "gge1_v3_fig2_transport",
       {"pairs": {k: {kk: pairs[k][kk] for kk in
                      ("median_rank", "top1_acc", "top5_acc", "median_cosine")}
                  for k in keys},
        "primary_pair": PRIMARY,
        "floor": {"shuffled_top1_distribution": sh,
                  "p_add_one": fl["p_add_one"], "floor_count": fl["floor_count"],
                  "n_perm": fl["frozen"]["n_perm"]},
        "note": "per-pair permutation p recorded DEFERRED in the artifact; "
                "floor graded on the primary pair only (ruling 1)"},
       ["geometry_out/pmap_transport.json",
        "geometry_out/pmap_transport_floor.json"])
print("16 pairs, medians:", sorted({v['median_rank'] for v in pairs.values()}),
      "| top1 min:", min(v['top1_acc'] for v in pairs.values()))
