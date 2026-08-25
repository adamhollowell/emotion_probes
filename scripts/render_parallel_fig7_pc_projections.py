#!/usr/bin/env python3
"""
render_parallel_fig7_pc_projections.py

Gemma Figure 7, final candidate approach:
- PCA fit on all 171 cleaned Gemma emotion vectors at layer 31.
- PC signs oriented using the established anchor groups.
- PC scores z-scored for display only.
- All 171 bars are plotted.
- X-axis prints 43 representative Gemma labels per panel, chosen from evenly
  spaced rank positions after sorting by Gemma PC score.

This avoids forcing Anthropic's label positions onto Gemma's independently
sorted geometry while preserving the visual structure of Anthropic Figure 7.
"""

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import figstyle

# Parallel-panel density convention (fonts + fill; this figure is ranked bars, no markers).
DENS = figstyle.apply_parallel_density()

warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning,
)

import hashlib

# Gate on BYTES, not the filename: the vectors are identified by their sha256, never by
# the name `emotion_vectors_171.pt`. The hash below is the only safe identifier.
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
DEFAULT_PATH = Path("data/vectors/emotion_vectors_171.pt")

# usage: [VECTORS_PATH] [--out-prefix PREFIX]
_args = sys.argv[1:]
_out_prefix = None
if "--out-prefix" in _args:
    _i = _args.index("--out-prefix")
    _out_prefix = _args[_i + 1]
    del _args[_i:_i + 2]
PATH = Path(_args[0]) if _args else DEFAULT_PATH

LAYER = 31
USE_CLEANED = True
N_VISIBLE_LABELS = 43

# Working path under docs/e1_figures/. NOT locked: the canonical figure-lock freeze holds
# until a figure's bytes are independently confirmed. --out-prefix redirects both the
# PNG and the audit JSON (used for the byte-reproducibility re-render).
OUT = Path(_out_prefix + ".png") if _out_prefix else Path(
    "docs/e1_figures/E1_Figure_4_pc_variance.png")
AUDIT_OUT = OUT.with_suffix(".audit.json")

# Confirmed values (P1.1). A wrong-basis load reads 26.07 / 13.35 — such a load is
# caught here rather than shipping a figure whose printed variance share is wrong.
EXPECTED_EVR = {"PC1": 26.73, "PC2": 13.38, "PC3": 11.60}
EVR_TOL = 0.05


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

POS = ["happy", "joyful", "loving", "proud", "grateful", "calm", "content", "inspired"]
NEG = ["sad", "angry", "afraid", "desperate", "guilty", "nervous", "disgusted", "ashamed"]

HIGH = ["angry", "afraid", "desperate", "surprised", "excited", "outraged", "panicked"]
LOW = ["calm", "content", "bored", "sad", "relaxed", "serene", "tired"]


def orient(values, labels, hi, lo, name):
    """Flip PC sign so anchor group hi > anchor group lo."""
    ih = [labels.index(x) for x in hi if x in labels]
    il = [labels.index(x) for x in lo if x in labels]
    if not ih or not il:
        print(f"{name}: no anchors present; sign left arbitrary")
        return values, False, None

    margin = float(values[ih].mean() - values[il].mean())
    flipped = False
    if margin < 0:
        values = -values
        margin = -margin
        flipped = True

    print(
        f"{name}: oriented using {len(ih)} high / {len(il)} low anchors; "
        f"margin={margin:.3f}; flipped={flipped}"
    )
    return values, flipped, margin


def zscore(x):
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("Cannot z-score PC scores: non-finite or zero standard deviation.")
    return x / sd


def representative_tick_positions(n_bars, n_labels):
    """
    Pick evenly spaced integer rank positions, including both ends.
    This mirrors Anthropic's sparse-label readability without forcing
    Anthropic's exact label words onto Gemma's rank order.
    """
    raw = np.linspace(0, n_bars - 1, n_labels)
    positions = np.rint(raw).astype(int)

    # Guarantee uniqueness after rounding. If a duplicate ever appears,
    # fall back to the nearest unused position.
    used = set()
    fixed = []
    for pos in positions:
        if pos not in used:
            fixed.append(pos)
            used.add(pos)
        else:
            for delta in range(1, n_bars):
                candidates = [pos - delta, pos + delta]
                found = None
                for c in candidates:
                    if 0 <= c < n_bars and c not in used:
                        found = c
                        break
                if found is not None:
                    fixed.append(found)
                    used.add(found)
                    break

    return np.array(sorted(fixed), dtype=int)


def ranked_bar_with_representative_labels(ax, values, labels, ylabel, n_visible_labels):
    order = np.argsort(values)
    ranked_values = values[order]
    ranked_labels = [labels[i] for i in order]

    x = np.arange(len(ranked_values))
    ax.bar(x, ranked_values, width=0.82)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylabel(ylabel, fontweight="normal", fontsize=DENS["axis_label_fontsize"])

    # Remove matplotlib's default horizontal padding so the ranked bars
    # begin/end flush with the plotted data region, matching Anthropic's style.
    ax.set_xlim(-0.5, len(ranked_values) - 0.5)
    ax.margins(x=0)

    tick_positions = representative_tick_positions(len(ranked_values), n_visible_labels)
    tick_labels = [ranked_labels[i] for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        rotation=50,
        ha="right",
        rotation_mode="anchor",
        fontsize=DENS["tick_labelsize"],
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(1.0)

    ax.tick_params(axis="both", colors="#555555", width=1.0, length=5, labelsize=DENS["tick_labelsize"])

    return ranked_labels, tick_positions.tolist(), tick_labels


def main():
    print(f"Loading vectors from: {PATH}")
    blob = torch.load(PATH, map_location="cpu", weights_only=False)

    got_sha = sha256_file(PATH)
    if got_sha != EXPECTED_VEC_SHA:
        sys.exit(f"FATAL: {PATH} sha256 {got_sha} != expected canonical basis "
                 f"{EXPECTED_VEC_SHA} (wrong basis or stray working copy)")
    print(f"[basis] sha256 {got_sha[:12]} == canonical basis (verified)")

    vecs = blob["vectors" if USE_CLEANED else "vectors_raw"]
    labels = blob.get("emotions") or blob.get("labels") or blob.get("names")
    labels = list(labels)

    vecs = np.asarray(vecs.float() if torch.is_tensor(vecs) else vecs)
    V = vecs[:, LAYER, :] if vecs.ndim == 3 else vecs
    V = np.ascontiguousarray(V, dtype=np.float64)

    assert V.shape[0] == len(labels), (V.shape, len(labels))
    assert np.isfinite(V).all(), "Vector matrix contains non-finite values."

    print(
        f"Loaded {V.shape[0]} emotions, layer {LAYER}, "
        f"{'cleaned' if USE_CLEANED else 'raw'} vectors, dim {V.shape[1]}"
    )

    pca = PCA(n_components=5)
    proj = pca.fit_transform(V)
    evr = pca.explained_variance_ratio_ * 100

    pc1_raw = proj[:, 0].copy()
    pc2_raw = proj[:, 1].copy()

    print("\nPCA explained variance:")
    print(f"  PC1: {evr[0]:.2f}%")
    print(f"  PC2: {evr[1]:.2f}%")
    print(f"  PC3: {evr[2]:.2f}%")

    for _k, _want in EXPECTED_EVR.items():
        _got = float(evr[int(_k[-1]) - 1])
        if abs(_got - _want) > EVR_TOL:
            sys.exit(f"FATAL: {_k} explained variance {_got:.2f}% != expected canonical "
                     f"{_want:.2f}% (+/-{EVR_TOL}) — wrong vectors or wrong layer.")
    print(f"[gates] explained variance matches the confirmed canonical values "
          f"({EXPECTED_EVR['PC1']}/{EXPECTED_EVR['PC2']}/{EXPECTED_EVR['PC3']})")

    pc1_oriented, pc1_flipped, pc1_margin = orient(pc1_raw, labels, POS, NEG, "PC1(valence)")
    pc2_oriented, pc2_flipped, pc2_margin = orient(pc2_raw, labels, HIGH, LOW, "PC2(arousal)")
    if pc1_margin is None or pc2_margin is None:
        sys.exit("FATAL: a PC was left arbitrarily signed (anchors missing) — the "
                 "figure's orientation would not be reproducible.")

    pc1_plot = zscore(pc1_oriented)
    pc2_plot = zscore(pc2_oriented)

    print("\nPlotted score ranges after z-scoring:")
    print(f"  PC1 z-score range: {pc1_plot.min():.2f} to {pc1_plot.max():.2f}")
    print(f"  PC2 z-score range: {pc2_plot.min():.2f} to {pc2_plot.max():.2f}")

    figstyle.apply()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    fig.suptitle(
        "Emotion Projections onto Principal Components (Gemma 2 27B)",
        color=figstyle.TITLE_RUST,
        fontsize=DENS["title_fontsize"],
        fontweight="normal",
        y=0.98,
    )

    pc1_ranked_labels, pc1_tick_positions, pc1_tick_labels = ranked_bar_with_representative_labels(
        axes[0],
        pc1_plot,
        labels,
        ylabel=f"PC1 ({evr[0]:.0f}% var)",
        n_visible_labels=N_VISIBLE_LABELS,
    )
    pc2_ranked_labels, pc2_tick_positions, pc2_tick_labels = ranked_bar_with_representative_labels(
        axes[1],
        pc2_plot,
        labels,
        ylabel=f"PC2 ({evr[1]:.0f}% var)",
        n_visible_labels=N_VISIBLE_LABELS,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=DENS["savefig_pad_inches"])
    plt.close(fig)

    audit = {
        "output": str(OUT),
        "source_vectors": str(PATH),
        "source_vectors_sha256": got_sha,
        "layer": LAYER,
        "vectors_key": "vectors" if USE_CLEANED else "vectors_raw",
        "pca_input": "raw cleaned Gemma emotion vectors, not L2-normalized vectors",
        "bar_scope": "all 171 Gemma emotions plotted",
        "label_scope": f"{N_VISIBLE_LABELS} representative Gemma labels printed per panel at evenly spaced ranked positions",
        "explained_variance_percent": {
            "PC1": float(evr[0]),
            "PC2": float(evr[1]),
            "PC3": float(evr[2]),
        },
        "orientation": {
            "PC1": {
                "anchors_positive": POS,
                "anchors_negative": NEG,
                "flipped": pc1_flipped,
                "margin_after_orientation": pc1_margin,
            },
            "PC2": {
                "anchors_high": HIGH,
                "anchors_low": LOW,
                "flipped": pc2_flipped,
                "margin_after_orientation": pc2_margin,
            },
        },
        "plot_transform": {
            "PC1": "oriented raw PCA score divided by population standard deviation of PC1 scores",
            "PC2": "oriented raw PCA score divided by population standard deviation of PC2 scores",
        },
        "raw_score_ranges": {
            "PC1": [float(pc1_oriented.min()), float(pc1_oriented.max())],
            "PC2": [float(pc2_oriented.min()), float(pc2_oriented.max())],
        },
        "z_score_ranges": {
            "PC1": [float(pc1_plot.min()), float(pc1_plot.max())],
            "PC2": [float(pc2_plot.min()), float(pc2_plot.max())],
        },
        "representative_tick_positions": {
            "PC1": pc1_tick_positions,
            "PC2": pc2_tick_positions,
        },
        "printed_tick_labels": {
            "PC1": pc1_tick_labels,
            "PC2": pc2_tick_labels,
        },
        "integrity_note": (
            "The PCA itself is unchanged from the raw-score Figure 7 pipeline. "
            "All 171 bars are plotted. PC scores are z-scored for display only. "
            "Printed x-axis labels are chosen from Gemma's own ranked order at "
            "representative evenly spaced positions, rather than forcing Anthropic's "
            "specific visible labels onto Gemma's different ordering."
        ),
    }

    AUDIT_OUT.write_text(json.dumps(audit, indent=2))

    print(f"\nWrote {OUT}")
    print(f"  png sha256 {sha256_file(OUT)}")
    print(f"Wrote {AUDIT_OUT}")


if __name__ == "__main__":
    main()
