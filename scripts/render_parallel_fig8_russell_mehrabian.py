#!/usr/bin/env python3
"""
Canonical audited rebuild of Figure 8:
Gemma 2 27B emotion-vector PCA projections vs.
Russell & Mehrabian (1977) human Pleasure/Arousal ratings.

Purpose:
- Rebuild Figure 8 from source files.
- Use the Anthropic-parallel method:
    global PCA over all 171 Gemma emotion vectors,
    then slice to the 45 Russell & Mehrabian overlap terms.
- Save the figure, plotted point-level data, summary JSON, and audit TXT.
- Assert that printed r values match correlations recomputed from exported points.

Expected display (canonical basis):
- PC1 vs Human Pleasure: r ≈ 0.72
- PC2 vs Human Arousal:  r ≈ 0.67
"""

import os
import re
import csv
import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path.cwd()

# Gate on BYTES, not the filename: the vectors are identified by their sha256, never by
# the name `emotion_vectors_171.pt`. The hash below is the only safe identifier.
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
DEFAULT_VECTOR_PATH = Path("data/vectors/emotion_vectors_171.pt")

# Basis label is DERIVED from the observed hash, never asserted by the caller, so a
# mislabeled provenance stamp cannot be produced by pointing this at the wrong file.
BASIS_BY_SHA = {
    "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b": "canonical",
}

# usage: [VECTORS_PATH] [--out-prefix PREFIX]
_args = sys.argv[1:]
_out_prefix = None
if "--out-prefix" in _args:
    _i = _args.index("--out-prefix")
    _out_prefix = _args[_i + 1]
    del _args[_i:_i + 2]
VECTOR_PATH = Path(_args[0]) if _args else DEFAULT_VECTOR_PATH

RATINGS_PATH = PROJECT_ROOT / "data" / "ratings" / "russell_mehrabian_1977.csv"
OUT_DIR = PROJECT_ROOT / "geometry_out"
OUT_DIR.mkdir(exist_ok=True)

LAYER = 31
VECTOR_KEY = "vectors"  # cleaned vectors — never vectors_raw

STEM = "fig8_pca_vs_russell_mehrabian"

# Confirmed canonical values, from geometry_out/s5_supplement.json
# `rm` (canonical-stamped, vectors_sha256 6de2e293…) and its `pc_var_pct`. A wrong-basis
# load is caught here rather than shipping a figure with the wrong printed r.
# A wrong-basis load reads 0.7102 / 0.6302 and variance 26.07 / 13.35 / 11.87.
EXPECTED_R = {
    "pc1_pleasure": 0.7203121031143593,
    "pc2_arousal": 0.6675247148150895,
    "cross_pc1_arousal": -0.12519324201094784,
    "cross_pc2_pleasure": -0.014592799663455518,
}
R_TOL = 1e-6
EXPECTED_EVR = {"pc1": 26.73, "pc2": 13.38, "pc3": 11.60}
EVR_TOL = 0.05
EXPECTED_DISPLAY = (0.72, 0.67)

# Working paths under docs/e1_figures/. NOT locked: the canonical figure-lock freeze holds
# until a figure's bytes are independently confirmed. --out-prefix redirects every
# output (used for the byte-reproducibility re-render).
if _out_prefix:
    PNG_OUT = Path(_out_prefix + ".png")
    PDF_OUT = Path(_out_prefix + ".pdf")
    POINTS_OUT = Path(_out_prefix + "_points.csv")
    SUMMARY_OUT = Path(_out_prefix + "_summary.json")
    AUDIT_OUT = Path(_out_prefix + "_audit.txt")
else:
    FIG_DIR = PROJECT_ROOT / "docs" / "e1_figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_OUT = FIG_DIR / "E1_Figure_5_human_norms.png"
    PDF_OUT = FIG_DIR / "E1_Figure_5_human_norms.pdf"
    POINTS_OUT = OUT_DIR / f"{STEM}_points.csv"
    SUMMARY_OUT = OUT_DIR / f"{STEM}_summary.json"
    AUDIT_OUT = OUT_DIR / f"{STEM}_audit.txt"

# Matplotlib stamps the current time into the PDF by default, which defeats a byte
# comparison across re-renders. Pin it (same treatment as Fig 2).
_PDF_META = {"CreationDate": None}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path) -> str:
    """Path relative to PROJECT_ROOT when possible, so the emitted artifacts are
    machine-independent and the two tree copies are byte-identical."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)


def norm_term(x: str) -> str:
    """Normalize terms for exact matching."""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def corr(a, b) -> float:
    """Pearson correlation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.corrcoef(a, b)[0, 1])


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=0)


# ---------------------------------------------------------------------
# Load Russell & Mehrabian ratings
# ---------------------------------------------------------------------

if not RATINGS_PATH.exists():
    raise FileNotFoundError(f"Missing ratings file: {RATINGS_PATH}")

ratings_raw = pd.read_csv(RATINGS_PATH)

required_cols = {"Term", "Pleasure Mean", "Arousal Mean"}
missing_cols = required_cols - set(ratings_raw.columns)
if missing_cols:
    raise ValueError(f"Ratings file missing required columns: {sorted(missing_cols)}")

ratings = ratings_raw.copy()
ratings["term_norm"] = ratings["Term"].map(norm_term)

if ratings["term_norm"].duplicated().any():
    dupes = ratings.loc[ratings["term_norm"].duplicated(), "term_norm"].tolist()
    raise ValueError(f"Duplicate normalized terms in ratings file: {dupes}")

ratings_lookup = ratings.set_index("term_norm")


# ---------------------------------------------------------------------
# Load Gemma vectors
# ---------------------------------------------------------------------

if not VECTOR_PATH.exists():
    raise FileNotFoundError(f"Missing vector file: {VECTOR_PATH}")

VECTOR_SHA = sha256_file(VECTOR_PATH)
if VECTOR_SHA != EXPECTED_VEC_SHA:
    sys.exit(
        f"FATAL: {VECTOR_PATH} sha256 {VECTOR_SHA} != expected canonical basis "
        f"{EXPECTED_VEC_SHA} (wrong basis or stray working copy; this file resolves "
        f"to basis {BASIS_BY_SHA.get(VECTOR_SHA, 'UNKNOWN')})"
    )
BASIS = BASIS_BY_SHA[VECTOR_SHA]
print(f"[basis] {rel(VECTOR_PATH)} sha256 {VECTOR_SHA[:12]}… == {BASIS} basis (verified)")

RATINGS_SHA = sha256_file(RATINGS_PATH)
print(f"[norms] {rel(RATINGS_PATH)} sha256 {RATINGS_SHA[:12]}…")

blob = torch.load(VECTOR_PATH, map_location="cpu", weights_only=False)

for key in ["emotions", VECTOR_KEY]:
    if key not in blob:
        raise KeyError(f"Vector blob missing key: {key}")

emotions = [norm_term(e) for e in list(blob["emotions"])]
V_all = np.asarray(blob[VECTOR_KEY].float())

if V_all.ndim != 3:
    raise ValueError(f"Expected vectors shape (n_emotions, n_layers, d_model), got {V_all.shape}")

if LAYER >= V_all.shape[1]:
    raise ValueError(f"Requested layer {LAYER}, but vectors only have {V_all.shape[1]} layers")

V = V_all[:, LAYER, :]

if len(emotions) != V.shape[0]:
    raise ValueError(f"Emotion label count {len(emotions)} does not match vector count {V.shape[0]}")

if len(set(emotions)) != len(emotions):
    raise ValueError("Duplicate normalized Gemma emotion labels found")


# ---------------------------------------------------------------------
# Global PCA over all 171 emotion vectors
# ---------------------------------------------------------------------

V_centered = V - V.mean(axis=0, keepdims=True)
U, S, VT = np.linalg.svd(V_centered, full_matrices=False)
coords_raw = U[:, :3] * S[:3]

explained_variance = (S ** 2) / np.sum(S ** 2)
evr = explained_variance[:3]

pc1_raw = coords_raw[:, 0].copy()
pc2_raw = coords_raw[:, 1].copy()
pc3_raw = coords_raw[:, 2].copy()


# ---------------------------------------------------------------------
# Exact overlap with Russell & Mehrabian
# ---------------------------------------------------------------------

valid_indices = [i for i, e in enumerate(emotions) if e in ratings_lookup.index]
valid_terms = [emotions[i] for i in valid_indices]

if len(valid_terms) != 45:
    raise AssertionError(f"Expected 45 exact overlap terms, found {len(valid_terms)}")

# Keep the vector-file order for plotting and correlation.
pleasure = np.array([ratings_lookup.loc[e, "Pleasure Mean"] for e in valid_terms], dtype=float)
arousal = np.array([ratings_lookup.loc[e, "Arousal Mean"] for e in valid_terms], dtype=float)


# ---------------------------------------------------------------------
# Orient PC signs by human axes for Figure 8 readability
# ---------------------------------------------------------------------
# PC signs are arbitrary. For this validation figure, orient PC1 so it
# correlates positively with Pleasure and PC2 so it correlates positively
# with Arousal.

if corr(pleasure, pc1_raw[valid_indices]) < 0:
    pc1_raw *= -1

if corr(arousal, pc2_raw[valid_indices]) < 0:
    pc2_raw *= -1

pc1_plot = zscore(pc1_raw)
pc2_plot = zscore(pc2_raw)
pc3_plot = zscore(pc3_raw)


# ---------------------------------------------------------------------
# Build point-level dataframe
# ---------------------------------------------------------------------

points = pd.DataFrame({
    "emotion": valid_terms,
    "pleasure": pleasure,
    "arousal": arousal,
    "pc1_raw_oriented": pc1_raw[valid_indices],
    "pc2_raw_oriented": pc2_raw[valid_indices],
    "pc3_raw": pc3_raw[valid_indices],
    "pc1_z_plot": pc1_plot[valid_indices],
    "pc2_z_plot": pc2_plot[valid_indices],
    "pc3_z_plot": pc3_plot[valid_indices],
})

# Main correlations from plotted coordinates.
r_pc1_pleasure = corr(points["pleasure"], points["pc1_z_plot"])
r_pc2_arousal = corr(points["arousal"], points["pc2_z_plot"])

# Same correlations from raw oriented PC coordinates.
r_pc1_pleasure_raw = corr(points["pleasure"], points["pc1_raw_oriented"])
r_pc2_arousal_raw = corr(points["arousal"], points["pc2_raw_oriented"])

# Cross terms.
r_pc1_arousal = corr(points["arousal"], points["pc1_z_plot"])
r_pc2_pleasure = corr(points["pleasure"], points["pc2_z_plot"])

# Integrity assertions.
assert abs(r_pc1_pleasure - r_pc1_pleasure_raw) < 1e-12
assert abs(r_pc2_arousal - r_pc2_arousal_raw) < 1e-12
assert round(r_pc1_pleasure, 2) == EXPECTED_DISPLAY[0], (
    f"displayed PC1/Pleasure r {round(r_pc1_pleasure, 2)} != {EXPECTED_DISPLAY[0]}")
assert round(r_pc2_arousal, 2) == EXPECTED_DISPLAY[1], (
    f"displayed PC2/Arousal r {round(r_pc2_arousal, 2)} != {EXPECTED_DISPLAY[1]}")

# Full-precision agreement with the confirmed canonical supplement, not just the rounded
# display: 0.72/0.67 alone would not distinguish this basis from a near neighbour.
_observed_r = {
    "pc1_pleasure": r_pc1_pleasure,
    "pc2_arousal": r_pc2_arousal,
    "cross_pc1_arousal": r_pc1_arousal,
    "cross_pc2_pleasure": r_pc2_pleasure,
}
for _k, _want in EXPECTED_R.items():
    _got = _observed_r[_k]
    assert abs(_got - _want) < R_TOL, (
        f"{_k} = {_got:.10f} != confirmed canonical {_want:.10f} (tol {R_TOL})")

# Same PCA object as Figs 4/7 — its variance shares pin the basis independently of
# the human norms. A wrong-basis load reads 26.07 / 13.35 / 11.87.
_observed_evr = {"pc1": evr[0] * 100, "pc2": evr[1] * 100, "pc3": evr[2] * 100}
for _k, _want in EXPECTED_EVR.items():
    _got = _observed_evr[_k]
    assert abs(_got - _want) < EVR_TOL, (
        f"PCA variance {_k} = {_got:.4f}% != confirmed canonical {_want}% (tol {EVR_TOL})")

points.to_csv(POINTS_OUT, index=False)


# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------

LABELS_PC1 = {
    "lonely", "hostile", "upset", "tense", "terrified", "anxious",
    "alert", "aroused", "contemptuous", "relaxed", "hopeful", "happy"
}

LABELS_PC2 = {
    "enraged", "contemptuous", "disdainful", "elated", "astonished",
    "distressed", "bored", "weary", "listless", "depressed",
    "regretful", "inspired"
}

figstyle.apply()
# Density convention for the parallel-panel figures: fill the canvas, dominant axis
# labels/ticks/annotations, ~2x markers — matching the original study's Figure 8.
DENS = figstyle.apply_parallel_density(ncols=2)
fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.2))  # ~2.11:1, near the original's aspect

fig.suptitle(
    "Probe PCA Correlates with Human Ratings (Gemma 2 27B)",
    color=figstyle.TITLE_RUST,
    fontsize=DENS["title_fontsize"],
    fontweight="normal",
    y=0.98,
)


def render_panel(ax, df, x_col, y_col, xlabel, ylabel, labels, r_value, comp):
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    words = df["emotion"].tolist()

    # Scatter (labels added manually below). Marker size from the density convention.
    ax.scatter(xs, ys, s=DENS["marker_s"], color=figstyle.STEEL_BLUE, alpha=0.75,
               edgecolor="white", linewidth=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Regression line computed from exactly the plotted points.
    coef = np.polyfit(xs, ys, 1)
    poly = np.poly1d(coef)
    x_trend = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(
        x_trend,
        poly(x_trend),
        color="#9a9a9a",
        linestyle="--",
        linewidth=3,
        dash_capstyle="round",
    )

    y_range_offset = (ys.max() - ys.min()) * 0.015

    for i, word in enumerate(words):
        if word not in labels:
            continue

        x_val = xs[i]
        y_val = ys[i]

        dx, dy = 0.02, y_range_offset
        ha, va = "left", "bottom"

        if x_val > 0.75:
            dx = -0.02
            ha = "right"

        # Manual label nudges preserved from the historical production script.
        if word == "upset" and comp == 1:
            dx, dy = -0.02, -y_range_offset * 3.5
            ha = "right"

        if word == "hopeful" and comp == 1:
            dx = -0.02
            ha = "right"

        if word == "distressed" and comp == 2:
            dx, dy = 0.02, -y_range_offset * 1.5

        if word == "contemptuous" and comp == 2:
            dx = -0.02
            ha = "right"

        ax.text(
            x_val + dx,
            y_val + dy,
            word,
            fontsize=DENS["point_label_fontsize"],
            color="#222222",
            ha=ha,
            va=va,
            zorder=4,
        )

    ax.set_xlim([-1.0, 1.0])
    ax.set_xticks([-0.5, 0.0, 0.5])

    if comp == 1:
        ax.set_ylim([-2.5, 2.8])
        ax.set_yticks([-2, -1, 0, 1, 2])
    else:
        ax.set_ylim([-2.2, 2.2])
        ax.set_yticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")
    ax.xaxis.label.set_size(DENS["axis_label_fontsize"])
    ax.yaxis.label.set_size(DENS["axis_label_fontsize"])
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")

    ax.tick_params(axis="both", colors="#333333", labelsize=DENS["tick_labelsize"], width=0.8)
    ax.text(
        0.04,
        0.93,
        f"r = {r_value:.2f}",
        transform=ax.transAxes,
        fontsize=DENS["annotation_fontsize"],
        color="#222222",
    )


pc1_var_label = f"PC1 ({evr[0] * 100:.0f}% var)"
pc2_var_label = f"PC2 ({evr[1] * 100:.0f}% var)"

render_panel(
    axes[0],
    points,
    x_col="pleasure",
    y_col="pc1_z_plot",
    xlabel="Human Pleasure",
    ylabel=pc1_var_label,
    labels=LABELS_PC1,
    r_value=r_pc1_pleasure,
    comp=1,
)

render_panel(
    axes[1],
    points,
    x_col="arousal",
    y_col="pc2_z_plot",
    xlabel="Human Arousal",
    ylabel=pc2_var_label,
    labels=LABELS_PC2,
    r_value=r_pc2_arousal,
    comp=2,
)

# The axis limits below are literals inherited from the historical production script
# and were tuned on an earlier basis. Verify — do not assume — that no plotted point
# falls outside them on this basis; a silently clipped point would drop an emotion
# from the figure while the correlations still passed.
CLIP_CHECKS = [
    ("PC1 panel", axes[0], points["pleasure"], points["pc1_z_plot"]),
    ("PC2 panel", axes[1], points["arousal"], points["pc2_z_plot"]),
]
clip_report = []
for _name, _ax, _xs, _ys in CLIP_CHECKS:
    (_x0, _x1), (_y0, _y1) = _ax.get_xlim(), _ax.get_ylim()
    _out = [
        (e, float(x), float(y))
        for e, x, y in zip(points["emotion"], _xs, _ys)
        if not (_x0 <= x <= _x1 and _y0 <= y <= _y1)
    ]
    assert not _out, f"{_name}: points outside axis limits (would be clipped): {_out}"
    clip_report.append(
        f"- {_name}: all 45 points inside xlim [{_x0:g}, {_x1:g}] / ylim [{_y0:g}, {_y1:g}]"
    )

plt.subplots_adjust(**DENS["margins"])

fig.savefig(PNG_OUT, dpi=180, facecolor="white", bbox_inches="tight",
            pad_inches=DENS["savefig_pad_inches"])
fig.savefig(PDF_OUT, facecolor="white", bbox_inches="tight",
            pad_inches=DENS["savefig_pad_inches"], metadata=_PDF_META)
plt.close(fig)


# ---------------------------------------------------------------------
# Summary and audit
# ---------------------------------------------------------------------

summary = {
    "figure": "Figure 8 - Russell & Mehrabian (1977) parallel",
    "figure": "Figure 5",
    "model": "Gemma 2 27B",
    "layer": LAYER,
    "basis": BASIS,
    "input_manifest": {
        "vectors": {"path": rel(VECTOR_PATH), "sha256": VECTOR_SHA},
        "ratings": {"path": rel(RATINGS_PATH), "sha256": RATINGS_SHA},
    },
    "vector_file": rel(VECTOR_PATH),
    "vector_sha256": VECTOR_SHA,
    "vector_key": VECTOR_KEY,
    "ratings_source": rel(RATINGS_PATH),
    "ratings_sha256": RATINGS_SHA,
    "pca_fit_scope": "all 171 Gemma emotion vectors",
    "overlap_rule": "exact normalized term intersection",
    "n_total_emotions": len(emotions),
    "n_overlap": len(valid_terms),
    "pc_orientation": {
        "pc1": "oriented to positive Pearson correlation with Russell & Mehrabian Pleasure",
        "pc2": "oriented to positive Pearson correlation with Russell & Mehrabian Arousal",
    },
    "correlations": {
        "pc1_pleasure_r_from_plotted_points": r_pc1_pleasure,
        "pc2_arousal_r_from_plotted_points": r_pc2_arousal,
        "pc1_pleasure_r_from_raw_oriented_pc": r_pc1_pleasure_raw,
        "pc2_arousal_r_from_raw_oriented_pc": r_pc2_arousal_raw,
        "pc1_arousal_cross_r": r_pc1_arousal,
        "pc2_pleasure_cross_r": r_pc2_pleasure,
    },
    "displayed_r": {
        "pc1_pleasure": round(r_pc1_pleasure, 2),
        "pc2_arousal": round(r_pc2_arousal, 2),
    },
    "pca_explained_variance_percent": {
        "pc1": float(evr[0] * 100),
        "pc2": float(evr[1] * 100),
        "pc3": float(evr[2] * 100),
    },
    "expected_canonical": {
        "correlations": EXPECTED_R,
        "correlation_tolerance": R_TOL,
        "pca_explained_variance_percent": EXPECTED_EVR,
        "variance_tolerance": EVR_TOL,
        "displayed_r": {"pc1_pleasure": EXPECTED_DISPLAY[0],
                        "pc2_arousal": EXPECTED_DISPLAY[1]},
        "source": "geometry_out/s5_supplement.json (rm, pc_var_pct)",
    },
    "self_check_passed": True,
    "outputs": {
        "png": rel(PNG_OUT),
        "pdf": rel(PDF_OUT),
        "points_csv": rel(POINTS_OUT),
        "summary_json": rel(SUMMARY_OUT),
        "audit_txt": rel(AUDIT_OUT),
    },
    "overlap_terms": valid_terms,
}

with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

audit_lines = [
    "Figure 8 canonical audited rebuild (Figure 5)",
    "=" * 48,
    f"basis: {BASIS} (derived from vector sha256, not asserted)",
    f"vector_file: {rel(VECTOR_PATH)}",
    f"vector_sha256: {VECTOR_SHA}",
    f"vector_key: {VECTOR_KEY}",
    f"ratings_source: {rel(RATINGS_PATH)}",
    f"ratings_sha256: {RATINGS_SHA}",
    f"layer: {LAYER}",
    f"pca_fit_n: {len(emotions)}",
    f"overlap_n: {len(valid_terms)}",
    "",
    f"PC1 vs Human Pleasure r from plotted points: {r_pc1_pleasure:.6f}",
    f"PC1 vs Human Pleasure r from raw oriented PC: {r_pc1_pleasure_raw:.6f}",
    f"PC2 vs Human Arousal r from plotted points:  {r_pc2_arousal:.6f}",
    f"PC2 vs Human Arousal r from raw oriented PC:  {r_pc2_arousal_raw:.6f}",
    "",
    f"Cross PC1 vs Human Arousal r:  {r_pc1_arousal:.6f}",
    f"Cross PC2 vs Human Pleasure r: {r_pc2_pleasure:.6f}",
    "",
    f"PC1 variance percent: {evr[0] * 100:.6f}",
    f"PC2 variance percent: {evr[1] * 100:.6f}",
    f"PC3 variance percent: {evr[2] * 100:.6f}",
    "",
    f"Displayed PC1/Pleasure r: {r_pc1_pleasure:.2f}",
    f"Displayed PC2/Arousal r:  {r_pc2_arousal:.2f}",
    "",
    "Integrity checks:",
    f"- basis sha256 == canonical ({EXPECTED_VEC_SHA[:12]}…): {VECTOR_SHA == EXPECTED_VEC_SHA}",
    f"- points rows == 45: {len(points) == 45}",
    f"- raw/plotted PC1 correlation match: {abs(r_pc1_pleasure - r_pc1_pleasure_raw) < 1e-12}",
    f"- raw/plotted PC2 correlation match: {abs(r_pc2_arousal - r_pc2_arousal_raw) < 1e-12}",
    f"- displayed values match canonical target {EXPECTED_DISPLAY[0]} / {EXPECTED_DISPLAY[1]}: "
    f"{round(r_pc1_pleasure, 2) == EXPECTED_DISPLAY[0] and round(r_pc2_arousal, 2) == EXPECTED_DISPLAY[1]}",
    f"- full-precision correlations match s5_supplement (tol {R_TOL}): "
    f"{all(abs(_observed_r[k] - v) < R_TOL for k, v in EXPECTED_R.items())}",
    f"- PCA variance matches canonical {EXPECTED_EVR['pc1']} / {EXPECTED_EVR['pc2']} / "
    f"{EXPECTED_EVR['pc3']} (tol {EVR_TOL}): "
    f"{all(abs(_observed_evr[k] - v) < EVR_TOL for k, v in EXPECTED_EVR.items())}",
    *clip_report,
    "",
    "Outputs:",
    f"- {rel(PNG_OUT)}",
    f"- {rel(PDF_OUT)}",
    f"- {rel(POINTS_OUT)}",
    f"- {rel(SUMMARY_OUT)}",
    f"- {rel(AUDIT_OUT)}",
]

with open(AUDIT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(audit_lines) + "\n")

print("\n".join(audit_lines))
