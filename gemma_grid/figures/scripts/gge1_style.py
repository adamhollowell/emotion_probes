"""
gge1_style.py — shared style layer for all GGE1 figures.

Implements gemma_grid/drafting/figures/STYLE_DECISIONS.md (locked 2026-08-04)
as revised by review/REVIEW_ROUND1.md (print-worthiness pass, 2026-08-04):
  - color = size (locked steel-blue hues), linestyle = tuning (base solid,
    IT dashed), filled markers = base, open markers = IT
  - NEW (round 1, S3): marker SHAPE is a redundant size channel for
    grayscale robustness — 2B = circle, 9B = square, 27B = triangle
  - NEW (round 1, S1): figures are designed at final print size
    (~5.5 in text width) with true publication point sizes; apply()
    sets the font hierarchy (8 pt axis titles, regular weight; 7 pt ticks
    and legends; 8 pt medium near-black panel titles)
  - REVISED (round 3): NO in-figure exploratory marking of any kind —
    the author's 17-item shippability checklist bans watermarks/tags;
    exploratory status is caption-only. This supersedes the round-1
    "(exploratory)" tag and the original handoff's in-figure mandate
    (author override, 2026-08-04, recorded in STYLE_DECISIONS.md).
  - caption-only: no in-figure figure titles, section refs, or verdicts
  - every save emits 300 dpi PNG + vector PDF + <name>_data.json sidecar

All scripts run from the project root (so `import figstyle` works) and write
into gemma_grid/drafting/figures/.
"""
from __future__ import annotations
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle

FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
T100 = os.path.join(PROJECT_ROOT, "gemma_grid", "geometry_out")  # canonical basis, draft v2

# --- six-cell encoding (locked decision #1 + round-1 S3) --------------------
SIZE_COLORS = {          # anchored on E1 steel blue for 2B, darker with size
    "2b":  "#5B8DB8",
    "9b":  "#3D6F9E",
    "27b": "#1F4E79",
}
SIZE_MARKERS = {         # redundant grayscale channel (REVIEW_ROUND1 S3):
    "2b":  "o",          # 2B  = circle
    "9b":  "s",          # 9B  = square
    "27b": "^",          # 27B = triangle
}
TUNING_LS = {"base": "-", "it": "--"}

CELLS = ["gemma-2-2b-base", "gemma-2-2b-it",
         "gemma-2-9b-base", "gemma-2-9b-it",
         "gemma-2-27b-base", "gemma-2-27b-it"]

CELL_META = {
    "gemma-2-2b-base":  dict(size="2b",  tuning="base", layer=17, n_layers=26),
    "gemma-2-2b-it":    dict(size="2b",  tuning="it",   layer=17, n_layers=26),
    "gemma-2-9b-base":  dict(size="9b",  tuning="base", layer=28, n_layers=42),
    "gemma-2-9b-it":    dict(size="9b",  tuning="it",   layer=28, n_layers=42),
    "gemma-2-27b-base": dict(size="27b", tuning="base", layer=31, n_layers=46),
    "gemma-2-27b-it":   dict(size="27b", tuning="it",   layer=31, n_layers=46),
}
# n_layers verified against each cell's lens_summary.json (26/26/42/42/46/46).

GREY = "#888888"
LIGHT_GREY = "#CCCCCC"
BAND_GREY = "#E6E6E6"
ANNOT_GREY = "#777777"   # in-plot reference annotations (round-1 S1 spec)
ANNOT_FS = 6.5           # size for those annotations, points
ACCENT_RUST = "#B5502A"  # optB moral-cluster accent (REVIEW_ROUND1 fix 8)


def apply():
    """Global theme: figstyle base, then round-1 print-size overrides.

    Figures are designed AT final print size (figsize = target inches,
    dpi=300 at save), so every size below is a true publication point size
    (REVIEW_ROUND1 S1).
    """
    figstyle.apply()
    plt.rcParams.update({
        "font.size": 8,
        # panel titles: 8 pt medium near-black — not rust, not bold (S1/S3)
        "axes.titlesize": 8,
        "axes.titleweight": "medium",
        "axes.titlecolor": "#333333",
        "axes.titlepad": 4,
        # axis titles: 8 pt REGULAR weight, black (S1: kill all bold)
        "axes.labelsize": 8,
        "axes.labelweight": "normal",
        "axes.labelcolor": "black",
        # ticks and legends
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        # finer furniture at print scale
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        # embed TrueType fonts in PDF/PS for LaTeX inclusion
        # (round 3, checklist item 17; matplotlib default is Type 3)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def cell_style(cell: str) -> dict:
    m = CELL_META[cell]
    return dict(
        color=SIZE_COLORS[m["size"]],
        linestyle=TUNING_LS[m["tuning"]],
        marker=SIZE_MARKERS[m["size"]],
        markerfacecolor=(SIZE_COLORS[m["size"]] if m["tuning"] == "base" else "white"),
        markeredgecolor=SIZE_COLORS[m["size"]],
    )


def cell_label(cell: str) -> str:
    # "tuned", not "IT": figures match the paper's term (v3 caption review,
    # 2026-10-20)
    m = CELL_META[cell]
    return f"{m['size'].upper()} {'base' if m['tuning'] == 'base' else 'tuned'}"


def exploratory_tag(*args, **kwargs):
    """REMOVED (round 3). The author's shippability checklist bans in-figure
    exploratory watermarks/tags; status is caption-only. Raises so the tag
    cannot be silently reintroduced."""
    raise RuntimeError(
        "In-figure exploratory tags were removed by the round-3 ruling "
        "(shippability checklist items 2/16). State exploratory status in "
        "the caption instead.")


def panel_label(ax, letter, x=-0.14, y=1.02):
    """Bold (a)/(b) label outside the axes, top-left (REVIEW_ROUND1 S4).
    Only for figures with heterogeneous panels; homogeneous small
    multiples use panel titles instead."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="black",
            ha="left", va="bottom")


def save(fig, name: str, data: dict, sources: list[str]):
    """PNG 300dpi + vector PDF + _data.json sidecar (ground rule #5)."""
    png = os.path.join(FIG_DIR, f"{name}.png")
    pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    sidecar = {"figure": name,
               "source_artifacts": sources,
               "style": "STYLE_DECISIONS.md (locked 2026-08-04) + "
                        "review/REVIEW_ROUND1.md revisions",
               "plotted_values": data}
    with open(os.path.join(FIG_DIR, f"{name}_data.json"), "w") as f:
        json.dump(sidecar, f, indent=1)
    print(f"saved {name}: PNG+PDF+data.json")
