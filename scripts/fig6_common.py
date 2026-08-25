#!/usr/bin/env python3
"""
Shared construction for Figure 3 (the UMAP cluster scatter).

Both the interactive label editor (render_parallel_fig6_interactive.py) and the
static producer (render_parallel_fig6_umap.py) build the base figure through this
module, so the ONLY thing that can differ between the hand-placed labels and what
the repo re-renders is the set of word-label positions. Everything else — the
frozen UMAP coordinates, the cluster palette, the legend/key, the title, the
axis limits, the font/density convention — comes from here, once.

Run everything from the PROJECT ROOT so the relative paths resolve to the
verified main-copy artifacts (not the stale staged coords).
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import figstyle

# ---- shipped artifact paths (relative to project root) --------------------
# canonical rebuild (2026-08-17): ONE authoritative coordinate file. UMAP is
# non-deterministic, so the coords are computed exactly once, frozen here, and every
# render reads this file. Every other fig6*coords*.csv / fig6_label_layout*.json in
# the tree is a superseded artifact of the earlier attempt and is NOT read by this
# pipeline. Never recompute these coordinates (the producer's --recompute refuses to
# overwrite an existing AUTH file for exactly this reason).
COORDS_CSV = "geometry_out/fig3_coords_AUTH.csv"
# native canonical partition (certified seed-0 KMeans on 6de2e293 vectors; ledger 2026-08-17).
# NOT the reused prior-basis L31_kmeans_partition.csv. read via read_partition_map() (JSON or CSV).
PARTITION_CSV = "geometry_out/s4_clusters.json"
LAYOUT_JSON = "geometry_out/fig3_label_layout_AUTH.json"


def read_partition_map(path: str = None) -> dict:
    """emotion -> cluster_id, from either an s4_clusters JSON (families/members, id = list
    index) or a legacy CSV (emotion,cluster). Single source for the recompute + membership gate."""
    import json as _json
    from pathlib import Path as _Path
    path = path or PARTITION_CSV
    if str(path).endswith(".json"):
        fams = _json.loads(_Path(path).read_text())["families"]
        return {m: ci for ci, fam in enumerate(fams) for m in fam["members"]}
    import csv as _csv
    out = {}
    for r in _csv.DictReader(open(path)):
        col = "cluster" if "cluster" in r else [k for k in r if k != "emotion"][0]
        out[r["emotion"]] = int(r[col])
    return out

# ---- fixed figure geometry (identical in editor and producer) -------------
FIGSIZE = (10.0, 8.8)
TITLE = "UMAP of Gemma 2 27B Emotion Probe Clusters"
AXIS_PAD = 0.6          # data-unit padding around the point cloud for xlim/ylim
DPI = 180
SUBPLOTS_ADJUST = dict(left=0.03, right=0.97, top=0.94, bottom=0.03)

# valence-ordered palette (idx 0 = most positive ... 9 = most negative)
PALETTE = ["#8B1A1A", "#E8735A", "#F4A460", "#F0D060", "#B8C840",
           "#80C080", "#40A080", "#30A0B0", "#4080C0", "#2C3E7A"]

# emotions that carry a printed word-label (prototypes + boundary points).
# Seeds the editor's initial label set; the final render is driven by whatever
# labels the layout file actually contains.
DENSE_WORDS = ["exuberant", "elated", "blissful", "optimistic", "inspired", "rejuvenated",
               "valiant", "grateful", "compassionate", "patient", "relaxed", "at ease",
               "exasperated", "impatient", "angry", "furious", "smug", "greedy", "obstinate",
               "paranoid", "suspicious", "disoriented", "melancholy", "grief-stricken",
               "sorry", "ashamed", "bored", "listless", "droopy", "sleepy", "nervous",
               "shaken", "afraid", "terrified", "surprised", "bewildered"]

# shrinkB=0 -> the leader reaches the exact target coordinate (the dot's center),
# not its inner edge.
LEADER_KW = dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=0, shrinkB=0)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_coords(path: str = COORDS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def check_membership(df: pd.DataFrame, partition_csv: str = PARTITION_CSV) -> None:
    """Cheap integrity gate (no sklearn / no recompute): the coords' cluster
    membership must equal the canonical partition, up to cluster renumbering."""
    part = defaultdict(set)
    for e, c in read_partition_map(partition_csv).items():
        part[c].add(e)
    coord = defaultdict(set)
    for r in df.itertuples():
        coord[int(r.cluster_idx)].add(r.emotion)
    a = sorted(sorted(s) for s in part.values())
    b = sorted(sorted(s) for s in coord.values())
    if a != b:
        raise SystemExit("FATAL: coords membership does not match %s" % partition_csv)


def axis_limits(df: pd.DataFrame):
    return (float(df.x.min() - AXIS_PAD), float(df.x.max() + AXIS_PAD),
            float(df.y.min() - AXIS_PAD), float(df.y.max() + AXIS_PAD))


def build_base(df: pd.DataFrame, scale: float = 1.0):
    """The identical base figure: scatter + legend/key + title + fixed limits.
    Returns (fig, ax, DENS, point_pos) where point_pos maps emotion -> (x, y).
    No word labels are drawn here.

    scale uniformly shrinks the whole figure — figsize, fonts, and markers
    together — so a smaller value opens a smaller ON-SCREEN window that fits the
    display while keeping the text-to-data geometry (and therefore label
    placement) identical. Done at normal dpi, so macOS hit-testing/picking is
    unaffected. The producer always renders at scale=1.0; the returned
    point_label_fontsize is already scaled for the caller to draw labels with."""
    figstyle.apply()                                  # Nunito + house theme
    DENS = dict(figstyle.apply_parallel_density())    # copy: we scale a few fields

    fig, ax = plt.subplots(figsize=(FIGSIZE[0] * scale, FIGSIZE[1] * scale), facecolor="white")
    ax.set_facecolor("white")

    name_for = {int(r.cluster_idx): r.cluster_name for r in df.itertuples()}
    for i in range(10):
        sub = df[df.cluster_idx == i]
        ax.scatter(sub.x, sub.y, color=PALETTE[i], s=DENS["marker_s"] * scale * scale,
                   alpha=0.85, edgecolors="none", label=f"{name_for[i]} ({len(sub)})")

    ax.set_title(TITLE, color=figstyle.TITLE_RUST, fontsize=DENS["title_fontsize"] * scale,
                 pad=12, weight="normal")
    ax.axis("off")
    x0, x1, y0, y1 = axis_limits(df)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    # layout: the warm/positive families now occupy the upper-left, so the key
    # moves to the lower-left (empty on this data shape) to avoid overlapping points.
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              edgecolor="#E0E0E0", fontsize=14 * scale, markerscale=1.1)
    fig.subplots_adjust(**SUBPLOTS_ADJUST)

    DENS["point_label_fontsize"] *= scale   # so live labels match the uniform scale
    point_pos = {r.emotion: (float(r.x), float(r.y)) for r in df.itertuples()}
    return fig, ax, DENS, point_pos


def layout_from_artists(df: pd.DataFrame, texts, point_pos) -> dict:
    """Capture the current label artists into a serializable layout dict."""
    x0, x1, y0, y1 = axis_limits(df)
    labels = {}
    for t in texts:
        w = t.get_text()
        cx, cy = t.get_position()
        ox, oy = point_pos[w]
        labels[w] = {"x": float(cx), "y": float(cy), "ha": t.get_ha(),
                     "va": t.get_va(), "target_x": float(ox), "target_y": float(oy)}
    return {
        "figure": "Figure 3 (UMAP)",
        "coords_csv": COORDS_CSV,
        "coords_sha256": sha256_file(COORDS_CSV),
        "figsize": list(FIGSIZE),
        "xlim": [x0, x1], "ylim": [y0, y1],
        "labels": labels,
    }


def save_layout(layout: dict, path: str = LAYOUT_JSON) -> None:
    Path(path).write_text(json.dumps(layout, indent=1))


def load_layout(path: str = LAYOUT_JSON):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def place_labels(ax, layout: dict, point_pos, fontsize: float) -> None:
    """Draw word labels + leader lines from a layout dict. Deterministic:
    no adjustText, positions taken verbatim from the layout."""
    for w, L in layout["labels"].items():
        ox = L.get("target_x", point_pos[w][0])
        oy = L.get("target_y", point_pos[w][1])
        ax.annotate("", xy=(ox, oy), xytext=(L["x"], L["y"]), arrowprops=dict(**LEADER_KW))
        ax.text(L["x"], L["y"], w, fontsize=fontsize, ha=L.get("ha", "center"),
                va=L.get("va", "center"), color="black")


def render_final(df: pd.DataFrame, layout: dict, out_png: str, dpi: int = DPI):
    """Build the base figure + place labels from the layout + save. This is the
    ONE code path that produces the canonical PNG; the editor calls it on save so
    its preview is byte-identical to what the producer emits."""
    fig, ax, DENS, point_pos = build_base(df)
    # honor the layout's own limits if present (keeps old layouts reproducible)
    if "xlim" in layout and "ylim" in layout:
        ax.set_xlim(*layout["xlim"])
        ax.set_ylim(*layout["ylim"])
    place_labels(ax, layout, point_pos, DENS["point_label_fontsize"])
    fig.savefig(out_png, dpi=dpi, facecolor="white", bbox_inches="tight",
                pad_inches=DENS["savefig_pad_inches"])
    plt.close(fig)
    return out_png
