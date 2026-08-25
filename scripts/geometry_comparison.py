#!/usr/bin/env python3
"""
geometry_comparison.py  --  Gemma-vs-Anthropic geometry comparison
representational-geometry comparison (Part-2 replication).

Runs entirely offline against `emotion_vectors_171.pt` (laptop job; no GPU, no model
forward pass). Reproduces each Part-2 geometric finding from the paper and
prints a Gemma-vs-Anthropic verdict:

  Paper anchor (171 emotions, mid-late layer):
    * PC1 = valence,  ~26% variance, orders fear/panic -> joy/optimism
    * PC2 = arousal,  ~15% variance, serene/reflective -> angry/playful
    * PC1 <-> human valence  r = 0.81  | PC2 <-> human arousal  r = 0.66
      (Mehrabian/PAD norms, restricted to the 45 overlapping emotions)
    * k-means (k=10) recovers interpretable, valence-ordered families
      (joy/excitement, sadness/grief/melancholy, fear/anxiety, anger, ...)
    * cosine geometry is stable across layers, esp. early-mid -> late

What this script computes from emotion_vectors_171.pt alone:
    1. PCA / affective circumplex   (Fig 7, Fig 57)
    2. k-means k=10 families         (Fig 6, Table 12)
    3. hierarchical cosine clustering (Fig 5)        -> dendrogram
    4. nearest-neighbour cosine sanity (incl. loving)
    5. cross-layer stability          (Fig 9)
    6. [optional] PC1/PC2 <-> valence/arousal correlation (Fig 8)
       -- only if a ratings CSV is supplied; never fabricated. See
          --ratings and the companion `make_valence_arousal_ratings.py`.

Outputs: prints a report to stdout AND writes figures + a machine-readable
summary to --out (default ./geometry_out/).

Dependencies: torch, numpy, scikit-learn, scipy, matplotlib.
    pip install scikit-learn scipy matplotlib   # torch+numpy already in lens_env

Usage:
    python geometry_comparison.py \
        --vectors data/vectors/emotion_vectors_171.pt \
        --layer 31 \
        --out geometry_out/ \
        [--ratings valence_arousal_171.csv] \
        [--use-raw]            # use vectors_raw instead of PCA-cleaned

The paper uses a "mid-late" layer ~2/3 through the model for its headline
analyses; for Gemma 2 27B (46 captured layers) that is layer 31 (the run's
documented primary layer). Pass --layer to override.
"""

from __future__ import annotations

# --- basis gate: refuse to run unless the vectors match the expected sha256.
# Identify the vectors by their hash, not the filename `emotion_vectors_171.pt`. ---
import hashlib as _hlg, pathlib as _plg
_VECG = _plg.Path("data/vectors/emotion_vectors_171.pt")
_EXPG = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
if _VECG.exists():
    _hg = _hlg.sha256(_VECG.read_bytes()).hexdigest()
    assert _hg == _EXPG, (
        "%s basis mismatch: %s != expected canonical %s -- refusing non-canonical basis"
        % (_VECG, _hg[:12], _EXPG[:12]))
# --- end basis gate ---

import argparse
import json
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------
# Anchors used only to (a) orient PCA sign and (b) sanity-check rankings.
# PCA signs are arbitrary; we flip so positive valence / high arousal point +.
# These never enter the variance or correlation numbers.
# --------------------------------------------------------------------------
POS_VALENCE_ANCHORS = ["happy", "joyful", "content", "blissful", "delighted",
                       "cheerful", "optimistic", "loving"]
NEG_VALENCE_ANCHORS = ["afraid", "sad", "angry", "miserable", "panicked",
                       "terrified", "depressed", "heartbroken"]
HIGH_AROUSAL_ANCHORS = ["enraged", "panicked", "ecstatic", "furious",
                        "terrified", "hysterical", "thrilled", "alarmed"]
LOW_AROUSAL_ANCHORS = ["calm", "serene", "sleepy", "relaxed", "listless",
                       "peaceful", "droopy", "at ease"]

# Emotions to probe for nearest-neighbour cosine sanity (Fig 5 expectations:
# fear~anxiety, joy~excitement). `loving` included for the essay program.
NEIGHBOUR_PROBES = ["afraid", "joyful", "angry", "sad", "loving",
                    "calm", "proud", "grateful", "compassionate"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_vectors(path: Path, use_raw: bool) -> tuple[list[str], np.ndarray]:
    """
    Load emotion_vectors_171.pt -> (emotions, V) where V is (n_emotions, n_layers,
    d_model) float32. Tolerates the EmotionVectors save schema (dict with
    'emotions', 'vectors', 'vectors_raw', ...).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict payload in {path}, got {type(payload)}")
    emotions = list(payload["emotions"])
    key = "vectors_raw" if use_raw else "vectors"
    V = payload[key]
    if not torch.is_tensor(V):
        V = torch.as_tensor(V)
    V = V.float().numpy()
    if V.ndim != 3:
        raise ValueError(f"{key} must be (n_emotions, n_layers, d_model); got {V.shape}")
    if V.shape[0] != len(emotions):
        raise ValueError(f"emotion count {len(emotions)} != vectors rows {V.shape[0]}")
    if np.isnan(V).any():
        raise ValueError("vectors contain NaNs")
    return emotions, V


# --------------------------------------------------------------------------
# Geometry primitives (self-contained; mirror emotion_probes.probes)
# --------------------------------------------------------------------------
def pca_at_layer(Vl: np.ndarray, n_components: int = 5):
    """
    PCA across emotion vectors at one layer.

    Returns (coords (n_emotions, n_components), explained_ratio (n_components,)).
    Centering matches emotion_probes.probes.emotion_pca; explained ratio is
    over the full spectrum (sum of all singular values squared), so the
    numbers are directly comparable to the paper's "PC1 = 26%".
    """
    Vc = Vl - Vl.mean(axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(Vc, full_matrices=False)
    coords = U[:, :n_components] * S[:n_components]
    ratio = (S[:n_components] ** 2) / (S ** 2).sum()
    return coords, ratio


def cosine_matrix(Vl: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Vl, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    Vn = Vl / norms
    return Vn @ Vn.T


def _anchor_mean(coords1d: np.ndarray, emotions: list[str], names: list[str]) -> float:
    idx = [emotions.index(n) for n in names if n in emotions]
    return float(coords1d[idx].mean()) if idx else 0.0


def orient_axes(coords: np.ndarray, emotions: list[str]):
    """
    Flip PC1 so positive-valence anchors are on the + side, and PC2 so
    high-arousal anchors are on the + side. Returns (coords, flipped_flags).
    Sign-only; affects readability of rankings/plots, not variance or |r|.
    """
    flipped = {}
    pc1 = coords[:, 0]
    if _anchor_mean(pc1, emotions, POS_VALENCE_ANCHORS) < _anchor_mean(pc1, emotions, NEG_VALENCE_ANCHORS):
        coords[:, 0] *= -1
        flipped["PC1"] = True
    else:
        flipped["PC1"] = False
    if coords.shape[1] > 1:
        pc2 = coords[:, 1]
        if _anchor_mean(pc2, emotions, HIGH_AROUSAL_ANCHORS) < _anchor_mean(pc2, emotions, LOW_AROUSAL_ANCHORS):
            coords[:, 1] *= -1
            flipped["PC2"] = True
        else:
            flipped["PC2"] = False
    return coords, flipped


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------
def analyse_pca(emotions, V, layer, out_dir, summary):
    print("\n" + "=" * 72)
    print(f"1. PCA / AFFECTIVE CIRCUMPLEX  (layer {layer})")
    print("=" * 72)
    Vl = V[:, layer, :]
    coords, ratio = pca_at_layer(Vl, n_components=5)
    coords, flipped = orient_axes(coords, emotions)

    pc1_pct, pc2_pct = 100 * ratio[0], 100 * ratio[1]
    pc3_pct = 100 * ratio[2]
    print(f"PC1 variance: {pc1_pct:5.1f}%   (paper ~26%)   sign-flipped={flipped['PC1']}")
    print(f"PC2 variance: {pc2_pct:5.1f}%   (paper ~15%)   sign-flipped={flipped.get('PC2')}")
    print(f"PC3 variance: {pc3_pct:5.1f}%   (paper notes arousal mixes PC2/PC3)")
    print(f"Top-5 cumulative: {100*ratio[:5].sum():.1f}%")

    order1 = np.argsort(-coords[:, 0])
    print("\nPC1 (valence) top 10  [expect joy/optimism/content]:")
    print("  " + ", ".join(emotions[i] for i in order1[:10]))
    print("PC1 (valence) bottom 10  [expect fear/panic/despair]:")
    print("  " + ", ".join(emotions[i] for i in order1[-10:]))

    order2 = np.argsort(-coords[:, 1])
    print("\nPC2 (arousal) top 10  [expect high-arousal: enraged/panicked]:")
    print("  " + ", ".join(emotions[i] for i in order2[:10]))
    print("PC2 (arousal) bottom 10  [expect low-arousal: serene/sleepy]:")
    print("  " + ", ".join(emotions[i] for i in order2[-10:]))

    # loving's location is essay-relevant
    li = emotions.index("loving")
    print(f"\nloving:  PC1={coords[li,0]:+.3f}  PC2={coords[li,1]:+.3f}  "
          f"(PC1 rank {list(order1).index(li)+1}/{len(emotions)}, "
          f"PC2 rank {list(order2).index(li)+1}/{len(emotions)})")

    _plot_circumplex(emotions, coords, pc1_pct, pc2_pct, out_dir)

    summary["pca"] = {
        "layer": layer,
        "explained_variance_ratio_top5": [float(x) for x in ratio],
        "pc1_pct": float(pc1_pct), "pc2_pct": float(pc2_pct), "pc3_pct": float(pc3_pct),
        "sign_flipped": flipped,
        "pc1_top10": [emotions[i] for i in order1[:10]],
        "pc1_bottom10": [emotions[i] for i in order1[-10:]],
        "pc2_top10": [emotions[i] for i in order2[:10]],
        "pc2_bottom10": [emotions[i] for i in order2[-10:]],
        "loving": {"pc1": float(coords[li, 0]), "pc2": float(coords[li, 1])},
    }
    return coords  # reused by k-means ordering


def _plot_circumplex(emotions, coords, pc1_pct, pc2_pct, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(coords[:, 0], coords[:, 1], s=14, alpha=0.55)
    # label a readable subset: anchors + a few extras, avoid 171 overlapping labels
    label_set = set(POS_VALENCE_ANCHORS + NEG_VALENCE_ANCHORS +
                    HIGH_AROUSAL_ANCHORS + LOW_AROUSAL_ANCHORS +
                    ["loving", "grateful", "compassionate", "proud",
                     "surprised", "bored", "nostalgic", "excited"])
    for i, e in enumerate(emotions):
        if e in label_set:
            ax.annotate(e, (coords[i, 0], coords[i, 1]), fontsize=7,
                        alpha=0.85, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"PC1  ({pc1_pct:.1f}%)  ->  valence")
    ax.set_ylabel(f"PC2  ({pc2_pct:.1f}%)  ->  arousal")
    ax.set_title("Gemma 2 27B emotion-vector circumplex (171 emotions)")
    fig.tight_layout()
    p = out_dir / "circumplex.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  [figure] {p}")


def analyse_kmeans(emotions, V, layer, coords_pc, out_dir, summary, k=10):
    print("\n" + "=" * 72)
    print(f"2. K-MEANS FAMILIES  (k={k}, layer {layer})")
    print("=" * 72)
    from sklearn.cluster import KMeans
    Vl = V[:, layer, :]
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Vl)
    labels = km.labels_
    # order clusters by mean PC1 (valence), like the paper's Table 12
    mean_pc1 = [coords_pc[labels == c, 0].mean() for c in range(k)]
    cluster_order = np.argsort(mean_pc1)  # ascending valence
    print("Clusters ordered by mean valence (PC1), low -> high.")
    print("Compare families to the paper's joy/excitement, sadness/grief,")
    print("fear/anxiety, anger groupings (Table 12 / Fig 6):\n")
    clusters_out = []
    for rank, c in enumerate(cluster_order):
        members = [emotions[i] for i in range(len(emotions)) if labels[i] == c]
        print(f"  [{rank+1:2d}] valence={mean_pc1[c]:+.2f}  n={len(members):3d}  "
              f"{', '.join(members[:14])}{' ...' if len(members) > 14 else ''}")
        clusters_out.append({"rank_by_valence": rank + 1,
                             "mean_pc1": float(mean_pc1[c]),
                             "members": members})
    # where loving lands
    lc = labels[emotions.index("loving")]
    lc_members = [emotions[i] for i in range(len(emotions)) if labels[i] == lc]
    print(f"\n  loving's cluster ({len(lc_members)} members): "
          f"{', '.join(m for m in lc_members if m != 'loving')[:200]}")
    summary["kmeans"] = {"k": k, "clusters": clusters_out,
                         "loving_cluster_members": lc_members}


def analyse_dendrogram(emotions, V, layer, out_dir, summary):
    print("\n" + "=" * 72)
    print(f"3. HIERARCHICAL COSINE CLUSTERING  (layer {layer})  -> dendrogram")
    print("=" * 72)
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    Vl = V[:, layer, :]
    C = cosine_matrix(Vl)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0  # enforce symmetry for squareform
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 22))
    dendrogram(Z, labels=emotions, orientation="right",
               leaf_font_size=5, ax=ax, color_threshold=0.7 * D.max())
    ax.set_title("Hierarchical clustering of Gemma emotion vectors (1 - cosine)")
    fig.tight_layout()
    p = out_dir / "dendrogram.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  [figure] {p}")

    # cosine heatmap (leaf-ordered) -- Fig 5 analogue
    leaf_order = dendrogram(Z, no_plot=True)["leaves"]
    Cord = C[np.ix_(leaf_order, leaf_order)]
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(Cord, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Pairwise cosine similarity (hierarchically ordered)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p2 = out_dir / "cosine_heatmap.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  [figure] {p2}")
    summary["clustering"] = {"dendrogram": str(p), "heatmap": str(p2)}


def analyse_neighbours(emotions, V, layer, summary):
    print("\n" + "=" * 72)
    print(f"4. NEAREST-NEIGHBOUR COSINE SANITY  (layer {layer})")
    print("=" * 72)
    print("Expect: fear~anxiety, joy~excitement (Fig 5).  loving included for essay.\n")
    Vl = V[:, layer, :]
    C = cosine_matrix(Vl)
    nn_out = {}
    for probe in NEIGHBOUR_PROBES:
        if probe not in emotions:
            continue
        i = emotions.index(probe)
        order = np.argsort(-C[i])
        neigh = [(emotions[j], float(C[i, j])) for j in order if j != i][:5]
        print(f"  {probe:14s} -> " +
              ", ".join(f"{n}({s:.2f})" for n, s in neigh))
        nn_out[probe] = neigh
    summary["nearest_neighbours"] = nn_out


def analyse_layer_stability(emotions, V, out_dir, summary, n_layers_sample=14):
    print("\n" + "=" * 72)
    print("5. CROSS-LAYER STABILITY  (Fig 9)")
    print("=" * 72)
    n_layers = V.shape[1]
    sample = np.linspace(0, n_layers - 1, min(n_layers_sample, n_layers)).round().astype(int)
    sample = sorted(set(sample.tolist()))
    # cosine matrix per sampled layer, upper-triangle flattened
    tris = []
    iu = np.triu_indices(len(emotions), k=1)
    for L in sample:
        C = cosine_matrix(V[:, L, :])
        tris.append(C[iu])
    tris = np.stack(tris)  # (n_sample, n_pairs)
    # correlation of cosine-structure between layer pairs
    M = np.corrcoef(tris)
    # mean off-diagonal stability among mid-late layers (>= 1/2 depth)
    midlate = [k for k, L in enumerate(sample) if L >= n_layers // 2]
    if len(midlate) >= 2:
        sub = M[np.ix_(midlate, midlate)]
        ml_mean = (sub.sum() - len(midlate)) / (len(midlate) ** 2 - len(midlate))
    else:
        ml_mean = float("nan")
    overall = (M.sum() - len(sample)) / (len(sample) ** 2 - len(sample))
    print(f"Sampled layers: {sample}")
    print(f"Mean structure correlation (all sampled pairs): {overall:.3f}")
    print(f"Mean structure correlation (mid-late pairs):     {ml_mean:.3f}  "
          f"(paper: stable, esp. early-mid -> late)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(sample))); ax.set_xticklabels(sample, fontsize=7)
    ax.set_yticks(range(len(sample))); ax.set_yticklabels(sample, fontsize=7)
    ax.set_title("Cross-layer cosine-structure stability")
    ax.set_xlabel("layer"); ax.set_ylabel("layer")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = out_dir / "layer_stability.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  [figure] {p}")
    summary["layer_stability"] = {
        "sampled_layers": sample,
        "mean_overall": float(overall),
        "mean_mid_late": float(ml_mean),
        "figure": str(p),
    }


def analyse_human_correlation(emotions, coords, ratings_csv, summary):
    """
    Optional Fig-8 replication: correlate PC1/PC2 with external valence/arousal
    ratings. Ratings are NEVER fabricated -- supply a CSV (emotion,valence,
    arousal) e.g. via make_valence_arousal_ratings.py, which mirrors the paper's
    LLM-judge method. Correlation uses |r| (PCA sign is arbitrary).
    """
    import csv
    print("\n" + "=" * 72)
    print("6. PC <-> VALENCE/AROUSAL CORRELATION  (Fig 8; optional)")
    print("=" * 72)
    ratings = {}
    with open(ratings_csv) as f:
        for row in csv.DictReader(f):
            ratings[row["emotion"].strip()] = (
                float(row["valence"]), float(row["arousal"]))
    idx = [i for i, e in enumerate(emotions) if e in ratings]
    if len(idx) < 3:
        print(f"  Only {len(idx)} emotions overlap the ratings CSV; skipping.")
        return
    val = np.array([ratings[emotions[i]][0] for i in idx])
    aro = np.array([ratings[emotions[i]][1] for i in idx])
    pc1 = coords[idx, 0]
    pc2 = coords[idx, 1]
    r_pc1_val = abs(np.corrcoef(pc1, val)[0, 1])
    r_pc2_aro = abs(np.corrcoef(pc2, aro)[0, 1])
    r_pc1_aro = abs(np.corrcoef(pc1, aro)[0, 1])
    r_pc2_val = abs(np.corrcoef(pc2, val)[0, 1])
    print(f"  n overlapping emotions: {len(idx)}")
    print(f"  |r| PC1 <-> valence: {r_pc1_val:.2f}   (paper 0.81)")
    print(f"  |r| PC2 <-> arousal: {r_pc2_aro:.2f}   (paper 0.66)")
    print(f"  (cross terms: PC1<->arousal {r_pc1_aro:.2f}, PC2<->valence {r_pc2_val:.2f})")
    summary["human_correlation"] = {
        "n": len(idx),
        "r_pc1_valence": float(r_pc1_val),
        "r_pc2_arousal": float(r_pc2_aro),
        "r_pc1_arousal": float(r_pc1_aro),
        "r_pc2_valence": float(r_pc2_val),
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vectors", type=Path, default=Path("data/vectors/emotion_vectors_171.pt"))
    ap.add_argument("--layer", type=int, default=31,
                    help="mid-late layer for headline analyses (Gemma 2 27B: 31)")
    ap.add_argument("--out", type=Path, default=Path("geometry_out"))
    ap.add_argument("--ratings", type=Path, default=None,
                    help="optional CSV: emotion,valence,arousal -> Fig-8 correlation")
    ap.add_argument("--use-raw", action="store_true",
                    help="use vectors_raw instead of PCA-cleaned vectors")
    ap.add_argument("--k", type=int, default=10, help="k-means cluster count")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    emotions, V = load_vectors(args.vectors, args.use_raw)
    n_emotions, n_layers, d_model = V.shape
    print(f"Loaded {n_emotions} emotions x {n_layers} layers x {d_model} d_model "
          f"({'raw' if args.use_raw else 'PCA-cleaned'}) from {args.vectors}")
    if not (0 <= args.layer < n_layers):
        raise SystemExit(f"--layer {args.layer} out of range [0,{n_layers})")
    for required in ("loving",):
        if required not in emotions:
            print(f"  WARNING: '{required}' not in emotion list; some readouts skipped")

    summary: dict = {"vectors_file": str(args.vectors),
                     "n_emotions": n_emotions, "n_layers": n_layers,
                     "d_model": d_model, "layer": args.layer,
                     "vector_set": "raw" if args.use_raw else "cleaned"}

    coords = analyse_pca(emotions, V, args.layer, args.out, summary)
    analyse_kmeans(emotions, V, args.layer, coords, args.out, summary, k=args.k)
    analyse_dendrogram(emotions, V, args.layer, args.out, summary)
    analyse_neighbours(emotions, V, args.layer, summary)
    analyse_layer_stability(emotions, V, args.out, summary)
    if args.ratings is not None:
        analyse_human_correlation(emotions, coords, args.ratings, summary)
    else:
        print("\n[6. PC<->valence/arousal correlation skipped: no --ratings CSV. "
              "Run make_valence_arousal_ratings.py to produce one.]")

    summary_path = args.out / "geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 72)
    print(f"Wrote machine-readable summary -> {summary_path}")
    print(f"Figures -> {args.out}/  (circumplex, dendrogram, cosine_heatmap, layer_stability)")
    print("=" * 72)


if __name__ == "__main__":
    main()
