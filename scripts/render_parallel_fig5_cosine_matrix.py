"""
render_parallel_fig5_cosine_matrix.py
------------------------------
Polished parallel of Anthropic Figure 5: pairwise cosine similarity between all
171 emotion probes, hierarchically clustered.

Pure vector geometry -> runs on a laptop from the emotion vectors. No GPU/API, no
k-means partition, no UMAP, no random seed: given the same vectors it reproduces.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/render_parallel_fig5_cosine_matrix.py
    python3 repo/github_repo/scripts/render_parallel_fig5_cosine_matrix.py --out-prefix PATH

canonical rebuild 2026-08-17: this figure is intentionally PARTITION-FREE — the block
structure emerges from the average-linkage ordering alone. Do not annotate it with
the k=10 families.
"""

import hashlib
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import figstyle

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
PATH        = "data/vectors/emotion_vectors_171.pt"
LAYER       = 31
USE_CLEANED = True
VLIM        = 1.0

# Gate on BYTES, not the filename: the vectors are identified by their sha256, never by
# the name `emotion_vectors_171.pt`. The hash below is the only safe identifier.
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_got_sha = _sha256_file(PATH)
if _got_sha != EXPECTED_VEC_SHA:
    sys.exit("FATAL: %s sha256 %s != expected canonical basis %s (wrong basis or stray "
             "working copy)" % (PATH, _got_sha, EXPECTED_VEC_SHA))
print("[basis] %s sha256 %s == canonical basis (verified)" % (PATH, _got_sha[:12]))

blob = torch.load(PATH, map_location="cpu", weights_only=False)

if isinstance(blob, dict):
    key = "vectors" if USE_CLEANED else "vectors_raw"
    vecs = blob[key]
    labels = blob.get("emotions") or blob.get("labels") or blob.get("names")
else:
    vecs = blob
    labels = None

vecs = np.asarray(
    vecs.float().cpu() if torch.is_tensor(vecs) else vecs,
    dtype=np.float32,
)

if vecs.ndim == 3:
    V = vecs[:, LAYER, :]
elif vecs.ndim == 2:
    V = vecs
else:
    raise ValueError(f"unexpected vector shape {vecs.shape}")

if labels is None:
    labels = [f"e{i}" for i in range(V.shape[0])]
    print(
        "WARNING: no emotion labels found in file; using placeholders. "
        "Set `labels` from your metadata for real tick labels."
    )

labels = [str(label) for label in labels]

# ---------------------------------------------------------------------------
# COMPUTE: cosine matrix + clustered order
# ---------------------------------------------------------------------------
V = V.astype(np.float64)
Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

# Avoid platform BLAS warnings from matrix multiplication on this Mac/Python build.
cosine = np.einsum("ik,jk->ij", Vn, Vn)
cosine = np.clip(cosine, -1.0, 1.0)

# ---------------------------------------------------------------------------
# SANITY GATES — cheap, and they catch a wrong basis / wrong layer immediately.
# The three spot pairs are the confirmed canonical neighbour cosines (they also appear
# in geometry_out/s4_clusters.json's `neighbors` block).
# ---------------------------------------------------------------------------
assert cosine.shape == (171, 171), "cosine matrix is %s, expected (171, 171)" % (cosine.shape,)
_diag = np.diag(cosine)
assert np.allclose(_diag, 1.0, atol=1e-6), "diagonal not ~1.0 (min %.6f)" % _diag.min()
assert cosine.min() >= -1.0 and cosine.max() <= 1.0, "values outside [-1, 1]"

_idx = {str(l).lower(): i for i, l in enumerate(labels)}
_SPOT = [("afraid", "scared", 0.987), ("joyful", "happy", 0.977),
         ("compassionate", "empathetic", 0.965)]
for _a, _b, _want in _SPOT:
    _got = float(cosine[_idx[_a], _idx[_b]])
    print("[spot] %-14s <-> %-12s cos = %.4f (expect ~%.3f)" % (_a, _b, _got, _want))
    if abs(_got - _want) > 0.002:
        sys.exit("FATAL: %s<->%s cosine %.4f != expected canonical value %.3f — wrong "
                 "vectors or wrong layer." % (_a, _b, _got, _want))
print("[gates] 171x171, diag~1.0, range in [-1,1], 3/3 spot cosines OK")

order = np.asarray(figstyle.cluster_order(cosine, method="average"))
# leaf order oriented so the positive block leads, matching the original study's Figure 5 layout; orientation is arbitrary under average linkage
_pos_anchors = {"joyful", "happy", "cheerful", "optimistic"}
_labels_lower = [str(l).lower() for l in labels]
_anchor_ranks = [rank for rank, orig_i in enumerate(order) if _labels_lower[orig_i] in _pos_anchors]
if _anchor_ranks and np.mean(_anchor_ranks) > (len(order) - 1) / 2.0:
    order = order[::-1]
cosine_ord = cosine[order][:, order]
labels_ord = [labels[i].capitalize() for i in order]

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------
figstyle.apply()
# Parallel-panel density convention (font/margin subset for this heatmap; no markers).
DENS = figstyle.apply_parallel_density(kind="heatmap")  # heatmap title kept ~pre-density

fig, ax = plt.subplots(figsize=(9.2, 8.4), facecolor="white")
ax.set_facecolor("white")

im = ax.imshow(
    cosine_ord,
    cmap="RdBu_r",
    vmin=-VLIM,
    vmax=VLIM,
    interpolation="nearest",
    aspect="equal",
)

# Anthropic labels only a readable subset, not all 171.
tick_every = 6
ticks = np.arange(0, len(labels_ord), tick_every)
tick_labels = [labels_ord[i] for i in ticks]

ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.set_xticklabels(
    tick_labels,
    rotation=50,
    ha="right",
    rotation_mode="anchor",
    fontsize=12,  # dense heatmap axis ticks: heatmap exception to the convention tick size
    color="#666666",
    fontweight="normal",
)

ax.set_yticklabels(
    tick_labels,
    fontsize=12,  # dense heatmap axis ticks: heatmap exception to the convention tick size
    color="#666666",
    fontweight="normal",
)

ax.tick_params(axis="both", colors="#666666", length=4.5, width=0.8)

# Anthropic-like black outline around the heatmap/data block.
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(0.9)

ax.set_xlabel(
    "Emotion Probe",
    fontsize=DENS["axis_label_fontsize"],
    fontweight="normal",
    color="black",
    labelpad=12,
)

ax.set_ylabel(
    "Emotion Probe",
    fontsize=DENS["axis_label_fontsize"],
    fontweight="normal",
    color="black",
    labelpad=12,
)

ax.set_title(
    "Emotion Probe Similarity Gemma 2 27B\n(hierarchically clustered, 171 emotions)",
    color=figstyle.TITLE_RUST,
    fontsize=DENS["title_fontsize"],
    fontweight="normal",
    linespacing=1.18,
    pad=16,
)

# Shorter colorbar, closer to Anthropic's proportions.
cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.025,
    shrink=0.88,
)

cbar.set_label(
    "Cosine Similarity",
    rotation=270,
    labelpad=23,
    fontsize=DENS["axis_label_fontsize"],
    fontweight="normal",
    color="black",
)

cbar.ax.tick_params(
    labelsize=DENS["tick_labelsize"],
    colors="#666666",
    length=4.5,
    width=0.8,
)

for tick in cbar.ax.get_yticklabels():
    tick.set_fontweight("normal")
    tick.set_color("#666666")

# Spacing tuned for a readable subset of row/column labels.
fig.subplots_adjust(
    left=0.16,
    right=0.88,
    top=0.84,
    bottom=0.22,
)

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
# Working path under docs/e1_figures/. NOT locked: per the figure-lock freeze, no
# canonical figure is promoted to locked_figures_and_tables/ until its bytes are
# independently confirmed. --out-prefix renders elsewhere (used for the
# byte-reproducibility re-render) without disturbing the working copy.
import os

_prefix = "docs/e1_figures/E1_Figure_2_cosine_matrix"
if "--out-prefix" in sys.argv:
    _prefix = sys.argv[sys.argv.index("--out-prefix") + 1]

out_png = _prefix + ".png"
out_pdf = _prefix + ".pdf"
os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

# Pin the PDF's CreationDate. Matplotlib stamps the current time by default, which
# would make the PDF differ on every run for reasons that have nothing to do with the
# figure. Pinning it lets the PDF be byte-compared like the PNG. Content unaffected.
_PDF_META = {"CreationDate": None}

fig.savefig(out_png, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=DENS["savefig_pad_inches"])
fig.savefig(out_pdf, facecolor="white", bbox_inches="tight", pad_inches=DENS["savefig_pad_inches"],
            metadata=_PDF_META)
plt.close(fig)

print(f"wrote {out_png}")
print(f"wrote {out_pdf}")
print("  png sha256 %s" % _sha256_file(out_png))
print("  pdf sha256 %s" % _sha256_file(out_pdf))
print(
    "layer",
    LAYER,
    "| linkage average on cosine-distance | cleaned=",
    USE_CLEANED,
)