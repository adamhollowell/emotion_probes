"""
figstyle.py
-----------
Reusable styling + plotting helpers to make your Gemma emotion-vector figures
visually match Anthropic's originals (transformer-circuits.pub/2026/emotions/),
so they can sit side-by-side for comparison.

Design notes / what was matched against the originals:
  - Diverging heatmaps (cosine similarity): matplotlib 'RdBu_r' (dark red +,
    white 0, dark blue -), with SYMMETRIC color limits. Read the original's
    colorbar limits and pass them in (commonly +/-0.1 for activation-cosine,
    +/-1.0 for probe-probe cosine).
  - Sequential heatmap (cross-layer, Fig 9): 'viridis'.
  - Bars / scatter points: a single muted steel blue.
  - Title color: rust / terracotta. The constant below is a CLOSE match;
    for an exact side-by-side, sample the real hex off an original PNG with a
    color picker and overwrite TITLE_RUST.
  - Font: Anthropic uses a rounded geometric sans. We try a list of close
    matches (Nunito / Quicksand / Comfortaa) and fall back silently to the
    default if none are installed. This is the only element that is a visual
    approximation rather than a confirmed value.

Usage:
    import figstyle
    figstyle.apply()                       # set the global theme once
    fig, ax = figstyle.diverging_heatmap(M, rows, cols, vlim=0.1)
    fig, ax = figstyle.ranked_bar(values, labels, ylabel="PC1 (27% var)")
    fig, ax = figstyle.labeled_scatter(x, y, labels)
    html = figstyle.shade_tokens(tokens, activations)   # -> HTML string
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, font_manager
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Color + font constants
# ---------------------------------------------------------------------------

TITLE_RUST = "#C0492C"   # close match; sample an original for the exact value
STEEL_BLUE = "#5B8DB8"   # bars / scatter points
GRID_GREY  = "#CCCCCC"

# Preferred rounded-sans fonts, in order. First one installed wins.
_PREFERRED_FONTS = ["Nunito", "Quicksand", "Comfortaa", "Varela Round",
                    "DejaVu Sans"]  # DejaVu always present as final fallback


def _pick_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _PREFERRED_FONTS:
        if name in installed:
            return name
    return "DejaVu Sans"


def apply():
    """Set the global matplotlib theme. Call once before plotting."""
    font = _pick_font()
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": font,
        "axes.titlecolor": TITLE_RUST,
        "axes.titlesize": 16,
        "axes.titleweight": "normal",
        "axes.titlepad": 14,
        "axes.labelsize": 13,
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "image.cmap": "RdBu_r",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return font


def apply_parallel_density(ncols=2, kind="default"):
    """Density convention for the parallel-panel figures (the E1 Figs 2/4/5/6/7 that mirror
    the original study's scatter/heatmap panels): fill the canvas, dominant axis labels /
    tick labels / annotations, and ~2x markers — so a side-by-side with the original reads
    as the same figure family. Call after apply(). Returns recommended per-artist sizes and
    a margin dict; each parallel figure applies these to its explicit fontsize/s args, its
    subplots_adjust, and savefig(pad_inches=...), so the convention lives in ONE place
    instead of per-figure literals. Calibrated by eye against the original's Figure 8.

    kind="heatmap": the enlarged title is too aggressive over a single square heatmap with a
    two-line header, so the title reverts to ~pre-density size while ticks, axis labels,
    colorbar text, and margins keep the enlarged convention. Used by Figs 2 and 7.
    """
    d = {
        "title_fontsize": 22,          # suptitle / panel title
        "axis_label_fontsize": 20,     # x/y axis labels (visually dominant)
        "tick_labelsize": 15,          # tick numbers
        "annotation_fontsize": 22,     # in-panel callouts (e.g. "r = 0.71")
        "point_label_fontsize": 13,    # per-point text labels (kept modest to limit collisions)
        "marker_s": 150,               # scatter size, ~2x the labeled_scatter default s=70
        "margins": dict(left=0.06, right=0.985, top=0.90, bottom=0.13, wspace=0.16),
        "savefig_pad_inches": 0.05,    # snug crop under savefig.bbox="tight"
    }
    if kind == "heatmap":
        d["title_fontsize"] = 16       # revert to ~pre-density; keep enlarged ticks/labels/cbar/margins
    return d


# ---------------------------------------------------------------------------
# Diverging heatmap  (Fig 5 cosine matrix, Fig 2 probe x scenario, etc.)
# ---------------------------------------------------------------------------

def diverging_heatmap(matrix, row_labels=None, col_labels=None, *,
                      vlim=None, title=None, cbar_label="Cosine Similarity",
                      xlabel=None, ylabel=None, figsize=None,
                      tick_every=1, ax=None):
    """
    Render a symmetric diverging heatmap in Anthropic's style.

    matrix      : 2D array
    vlim        : float; color limits are (-vlim, +vlim). If None, uses
                  the max abs value of the matrix (rounded up a touch).
                  *** For side-by-side comparison, pass the SAME vlim the
                  original uses (read it off its colorbar). ***
    tick_every  : show every Nth tick label (originals often label a subset)
    """
    matrix = np.asarray(matrix)
    if vlim is None:
        vlim = float(np.nanmax(np.abs(matrix)))
        vlim = np.ceil(vlim * 100) / 100  # round up to 2 dp

    if ax is None:
        if figsize is None:
            figsize = (8, 7)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                   aspect="auto", interpolation="nearest")

    n_rows, n_cols = matrix.shape
    if col_labels is not None:
        idx = range(0, n_cols, tick_every)
        ax.set_xticks(list(idx))
        ax.set_xticklabels([col_labels[i] for i in idx], rotation=45, ha="right")
    if row_labels is not None:
        idx = range(0, n_rows, tick_every)
        ax.set_yticks(list(idx))
        ax.set_yticklabels([row_labels[i] for i in idx])

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=270, labelpad=18, weight="normal")
    cbar.outline.set_visible(False)
    return fig, ax


def cluster_order(matrix, method="average"):
    """
    Hierarchical-clustering leaf order for a SIMILARITY matrix (e.g. cosine).
    Returns an index array to reorder rows AND columns symmetrically (Fig 5).
    Pin `method` and report it; 'average' on cosine-distance is a sane default.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    sim = np.asarray(matrix, dtype=float)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0            # enforce symmetry
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    return leaves_list(Z)


def sequential_heatmap(matrix, row_labels=None, col_labels=None, *,
                       vmin=None, vmax=None, title=None,
                       cbar_label="Cosine Similarity", cmap="viridis",
                       xlabel="Layer", ylabel="Layer", figsize=None, ax=None):
    """Sequential (viridis) heatmap for the cross-layer figure (Fig 9)."""
    matrix = np.asarray(matrix)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (8, 7))
    else:
        fig = ax.figure
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=270, labelpad=18, weight="normal")
    cbar.outline.set_visible(False)
    return fig, ax


# ---------------------------------------------------------------------------
# Ranked bar chart  (Fig 7 PC projections)
# ---------------------------------------------------------------------------

def ranked_bar(values, labels, *, ylabel=None, title=None,
               label_every=None, figsize=None, ax=None, color=STEEL_BLUE):
    """
    Sorted bar chart with a zero line, labelling a subset of x-ticks
    (originals show only some). values/labels are reordered by value here.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    v = values[order]
    labs = [labels[i] for i in order]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (14, 4))
    else:
        fig = ax.figure

    ax.bar(range(len(v)), v, color=color, width=0.9)
    ax.axhline(0, color="black", linewidth=0.8)

    if label_every is None:
        label_every = max(1, len(v) // 30)   # ~30 labels across
    idx = list(range(0, len(v), label_every))
    ax.set_xticks(idx)
    ax.set_xticklabels([labs[i] for i in idx], rotation=45, ha="right")
    ax.set_xlim(-1, len(v))
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


# ---------------------------------------------------------------------------
# Labeled scatter  (Fig 57 circumplex, Fig 8 PCA-vs-norms)
# ---------------------------------------------------------------------------

def labeled_scatter(x, y, labels=None, *, label_subset=None, title=None,
                    xlabel=None, ylabel=None, trendline=False, figsize=None,
                    ax=None, color=STEEL_BLUE, annotate_kw=None):
    """
    Scatter with optional point labels and an optional dashed trendline.
    label_subset : list of labels (or indices) to annotate; None = annotate all
                   if `labels` given (use a subset for crowded plots like Fig 57).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (9, 8))
    else:
        fig = ax.figure

    ax.scatter(x, y, s=70, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)

    if labels is not None:
        akw = dict(fontsize=10, color="#333333")
        if annotate_kw:
            akw.update(annotate_kw)
        if label_subset is None:
            to_label = range(len(labels))
        elif all(isinstance(s, str) for s in label_subset):
            to_label = [i for i, l in enumerate(labels) if l in set(label_subset)]
        else:
            to_label = label_subset
        for i in to_label:
            ax.annotate(labels[i], (x[i], y[i]),
                        xytext=(4, 4), textcoords="offset points", **akw)

    if trendline:
        m, b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, m * xs + b, "--", color="#555555", linewidth=1.5)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return fig, ax


# ---------------------------------------------------------------------------
# Per-token text shading  (Fig 1, 40-51, 60, and the minimal-pair figs)
# ---------------------------------------------------------------------------

def shade_tokens(tokens, activations, *, vlim=None, title=None,
                 fontsize=15, max_width_px=1100):
    """
    Build a self-contained HTML string that renders `tokens` as monospace text,
    each token background-shaded by its `activations` value on the SAME RdBu_r
    scale used by the heatmaps (dark red = +, white = 0, dark blue = -).

    This matches the per-token figures far more cleanly than matplotlib can.
    Open the returned HTML in a browser and screenshot for side-by-side use,
    or write it to a file.

    tokens      : list[str]
    activations : list/array[float], same length
    vlim        : symmetric color limit; None -> max abs activation
    """
    tokens = list(tokens)
    acts = np.asarray(activations, dtype=float)
    assert len(tokens) == len(acts), "tokens and activations must align"
    if vlim is None:
        vlim = float(np.nanmax(np.abs(acts))) or 1.0

    cmap = matplotlib.colormaps["RdBu_r"]
    norm = Normalize(vmin=-vlim, vmax=vlim)

    spans = []
    for tok, a in zip(tokens, acts):
        r, g, b, _ = cmap(norm(a))
        R, G, B = int(r * 255), int(g * 255), int(b * 255)
        # luminance -> white text on dark cells for readability
        lum = 0.299 * R + 0.587 * G + 0.114 * B
        fg = "#ffffff" if lum < 128 else "#000000"
        safe = (tok.replace("&", "&amp;").replace("<", "&lt;")
                   .replace(">", "&gt;").replace(" ", "&nbsp;"))
        spans.append(
            f'<span style="background-color:rgb({R},{G},{B});color:{fg};'
            f'padding:1px 0;border-radius:2px;">{safe}</span>'
        )

    title_html = (f'<div style="color:{TITLE_RUST};font-weight:bold;'
                  f'font-size:18px;margin-bottom:8px;">{title}</div>'
                  if title else "")
    body = "".join(spans)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>
  body {{ background:#fff; }}
  .wrap {{ font-family: 'DejaVu Sans Mono', Menlo, Consolas, monospace;
           font-size:{fontsize}px; line-height:1.9; max-width:{max_width_px}px;
           white-space:pre-wrap; word-break:break-word; }}
</style></head><body>
{title_html}<div class="wrap">{body}</div>
</body></html>"""


def write_html(html, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


# ---------------------------------------------------------------------------
# self-test with synthetic data (run `python figstyle.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    apply()
    rng = np.random.default_rng(0)
    M = rng.normal(size=(20, 20)); M = (M + M.T) / 2
    order = cluster_order(np.tanh(M))
    fig, _ = diverging_heatmap(np.tanh(M)[order][:, order], vlim=1.0,
                               title="self-test heatmap")
    fig.savefig("/tmp/_figstyle_heatmap.png"); plt.close(fig)

    vals = rng.normal(size=40)
    fig, _ = ranked_bar(vals, [f"e{i}" for i in range(40)],
                        ylabel="PC1 (27% var)", title="self-test bars")
    fig.savefig("/tmp/_figstyle_bars.png"); plt.close(fig)

    fig, _ = labeled_scatter(rng.normal(size=15), rng.normal(size=15),
                             [f"e{i}" for i in range(15)], title="self-test scatter")
    fig.savefig("/tmp/_figstyle_scatter.png"); plt.close(fig)

    html = shade_tokens(["The", "cat", "sat"], [0.8, -0.3, 0.1], title="self-test")
    write_html(html, "/tmp/_figstyle_tokens.html")
    print("OK: wrote /tmp/_figstyle_{heatmap,bars,scatter}.png and _tokens.html")
