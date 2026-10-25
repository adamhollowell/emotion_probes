"""GGE1 draft v2 — Figure 3 (§3.2): the spread.

(a) Leading-component variance share by size, both chains.
(b) Effective dimensionality: participation ratio (rises) and TwoNN
    intrinsic dimension (flat) by size, both chains, one axis — the
    opposing-trend reading the memo asks for.

Framing per ruling 4: PR/ID artifact is Phase-5 SUSTAINED (sha
ff483ee9f858, ledger SR.7 descriptive support); its internal "check owed"
status string is stale and the artifact is NOT edited (sha-pinned).

Sources: geometry_out/stageA_cross_cell_table.csv,
         geometry_out/op1_spectrum_measures_v1.json
"""
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gge1_style as G
import matplotlib.pyplot as plt

G.apply()

SIZES = ["2b", "9b", "27b"]
X = [0, 1, 2]
XL = ["2B", "9B", "27B"]
NEUTRAL_LINE = "#7A8A99"

rows = {r["cell"]: r for r in csv.DictReader(
    open(os.path.join(G.T100, "stageA_cross_cell_table.csv")))}
var_pc1 = {c: float(rows[c]["var_pc1"]) for c in G.CELLS}

spec_path = os.path.join(G.T100, "op1_spectrum_measures_v1.json")
sha = hashlib.sha256(open(spec_path, "rb").read()).hexdigest()[:12]
assert sha == "ff483ee9f858", f"spectrum artifact sha changed: {sha}"
sp = json.load(open(spec_path))["cells"]
pr = {c: sp[c]["cleaned"]["participation_ratio"] for c in G.CELLS}
idm = {c: sp[c]["cleaned"]["twonn_id"] for c in G.CELLS}

chain = lambda d, t: [d[f"gemma-2-{s}-{t}"] for s in SIZES]


def series(ax, d, tuning, color=NEUTRAL_LINE, alpha=1.0):
    y = chain(d, tuning)
    ax.plot(X, y, linestyle=G.TUNING_LS[tuning], color=color,
            linewidth=1.2, alpha=alpha, zorder=2)
    for xi, yi, s in zip(X, y, SIZES):
        st = G.cell_style(f"gemma-2-{s}-{tuning}")
        ax.plot(xi, yi, marker=G.SIZE_MARKERS[s], markersize=4.2,
                markerfacecolor=st["markerfacecolor"],
                markeredgecolor=st["markeredgecolor"],
                markeredgewidth=0.8, alpha=alpha, zorder=3)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.5, 2.7))

# (a) variance share
for t in ["base", "it"]:
    series(axA, var_pc1, t)
axA.set_xticks(X); axA.set_xticklabels(XL)
axA.set_xlim(-0.25, 2.25)
axA.set_xlabel("Model size")
axA.set_ylabel("PC1 variance share (%)")
from matplotlib.lines import Line2D
axA.legend(handles=[
    Line2D([], [], color=NEUTRAL_LINE, ls="-", label="base"),
    Line2D([], [], color=NEUTRAL_LINE, ls="--", label="instruction-tuned")],
    loc="lower left", fontsize=6.5, frameon=False)
G.panel_label(axA, "(a)")

# (b) PR rising, ID flat — one axis
for t in ["base", "it"]:
    series(axB, pr, t)
    series(axB, idm, t, color="#B0B0B0", alpha=0.9)
axB.set_xticks(X); axB.set_xticklabels(XL)
axB.set_xlim(-0.25, 2.25)
axB.set_ylim(3.4, 9.2)
axB.set_xlabel("Model size")
axB.set_ylabel("Effective dimensions")
axB.text(2.18, max(chain(pr, "base")[-1], chain(pr, "it")[-1]) - 0.55,
         "participation ratio", fontsize=G.ANNOT_FS, color="#555555",
         ha="right", va="top")
axB.text(2.18, chain(idm, "base")[-1] + 0.25,
         "intrinsic dimension (TwoNN)", fontsize=G.ANNOT_FS,
         color=G.ANNOT_GREY, ha="right", va="bottom")
G.panel_label(axB, "(b)")

fig.tight_layout()
G.save(fig, "gge1_v3_fig3_spread",
       {"var_pc1": var_pc1, "participation_ratio": pr, "twonn_id": idm,
        "spectrum_artifact_sha256_12": sha,
        "framing": "SR.7 descriptive support; Phase-5 SUSTAINED "
                   "2026-09-02; artifact status string stale by ruling 4"},
       ["geometry_out/stageA_cross_cell_table.csv",
        "geometry_out/op1_spectrum_measures_v1.json (ff483ee9f858)"])
print("base var_pc1:", chain(var_pc1, "base"), "| base PR:", chain(pr, "base"))
