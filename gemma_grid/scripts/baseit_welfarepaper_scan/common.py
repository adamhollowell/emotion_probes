#!/usr/bin/env python3
"""Shared helpers for the base<->IT welfare-paper scan (arXiv 2605.30232).

PRODUCER-RUNNER only. Exploratory, POST-FLIP, pre-posture. Outputs go to the
NEW untracked dir gemma_grid/geometry_out/baseit_welfarepaper_scan/ and are
NOT added to the flip manifest.

100T basis discipline (load-bearing):
  * anchor cell gemma-2-27b-base -> data/vectors/emotion_vectors_171.pt,
    sha256 6de2e293...971b (its internal variant label is the known item-10
    stale label; the CONTENT is 100T -- gate on hash/path, never the label).
  * five non-anchor cells -> gemma_grid/data/<cell>/vectors/canonical.pt
  * every cleaned-vector path is asserted to contain '100T'.
  * analysis layer resolved per-model as round(2/3 * n_layers) and cross-checked
    against the locked resolution (2b->17, 9b->28, 27b->31).

Reuses gemma_grid/scripts/noise_floor.py load_vectors / pca / load_ratings /
cos so the scan sits on the exact objects the fold-in producers use.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf  # noqa: E402  load_vectors / pca / load_ratings / cos

ANCHOR_SHA256 = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
RATINGS = REPO / "data/ratings/russell_mehrabian_1977.csv"
OUT_DIR = REPO / "gemma_grid" / "geometry_out" / "baseit_welfarepaper_scan"

# scale -> (2b, 9b, 27b) locked analysis-layer resolution, for the assert guard
LOCKED_LAYER = {"2b": 17, "9b": 28, "27b": 31}

# (cell, scale, tuning, cleaned-vector path, per-cell lens_ingredients path).
# Anchor (27b-base) cleaned vectors are the repo-root 100T file; its lens W_U is
# the repo-root lens_ingredients.pt (the f32 anchor bundle), handled separately.
CELLS = {
    "gemma-2-2b-base":  dict(scale="2b",  tuning="base",
        vec="gemma_grid/data/gemma-2-2b-base/vectors/canonical.pt",
        lens="gemma_grid/data/gemma-2-2b-base/vectors/gemma-2-2b-base__lens_ingredients.pt"),
    "gemma-2-2b-it":    dict(scale="2b",  tuning="it",
        vec="gemma_grid/data/gemma-2-2b-it/vectors/canonical.pt",
        lens="gemma_grid/data/gemma-2-2b-it/vectors/gemma-2-2b-it__lens_ingredients.pt"),
    "gemma-2-9b-base":  dict(scale="9b",  tuning="base",
        vec="gemma_grid/data/gemma-2-9b-base/vectors/canonical.pt",
        lens="gemma_grid/data/gemma-2-9b-base/vectors/gemma-2-9b-base__lens_ingredients.pt"),
    "gemma-2-9b-it":    dict(scale="9b",  tuning="it",
        vec="gemma_grid/data/gemma-2-9b-it/vectors/canonical.pt",
        lens="gemma_grid/data/gemma-2-9b-it/vectors/gemma-2-9b-it__lens_ingredients.pt"),
    "gemma-2-27b-base": dict(scale="27b", tuning="base",
        vec="data/vectors/emotion_vectors_171.pt",
        lens="data/vectors/lens_ingredients.pt"),  # f32 anchor bundle (token-major)
    "gemma-2-27b-it":   dict(scale="27b", tuning="it",
        vec="gemma_grid/data/gemma-2-27b-it/vectors/canonical.pt",
        lens="gemma_grid/data/gemma-2-27b-it/vectors/gemma-2-27b-it__lens_ingredients.pt"),
}


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(buf), b""):
            h.update(blk)
    return h.hexdigest()


def assert_100T(cell, vp):
    """Guard every cleaned-vector path. Anchor is exempt from the substring
    check (its filename is emotion_vectors_171.pt so it passes anyway) but
    is hash-gated on the load-bearing 100T sha256."""
    p = str(vp)
    assert "_HA" not in p and "_PA" not in p, f"[{cell}] superseded-basis vector path: {p}"
    if cell == "gemma-2-27b-base":
        got = sha256_file(vp)
        assert got == ANCHOR_SHA256, (
            f"anchor sha256 mismatch: {got} != {ANCHOR_SHA256} -- refusing to run")
    return True


def resolve_layer(cell, n_layers):
    """Analysis layer = round(2/3 * n_layers), cross-checked against the locked
    per-scale resolution. Never carry a literal across models."""
    lay = int(round((2.0 / 3.0) * n_layers))
    scale = CELLS[cell]["scale"]
    assert lay == LOCKED_LAYER[scale], (
        f"[{cell}] round(2/3*{n_layers})={lay} != locked {LOCKED_LAYER[scale]}")
    return lay


def load_cell(cell):
    """Return (em, V, layer). Enforces 100T basis + anchor hash + layer guard."""
    vp = REPO / CELLS[cell]["vec"]
    assert_100T(cell, vp)
    em, V = nf.load_vectors(vp)               # V float32 (n, L, d)
    n, L, d = V.shape
    layer = resolve_layer(cell, L)
    return em, V, layer


def signed_pca_valence_arousal(Vl, em, idx, pl, ar, k=5):
    """PCA of Vl (n,d); sign PC1 to +valence and PC2 to +arousal via RM corr,
    exactly the convention in noise_floor.compute (lines 92-97). Returns
    coords (n,k) and variance ratio."""
    coords, ratio = nf.pca(Vl, k)
    c1 = coords[idx, 0]
    c2 = coords[idx, 1]
    if np.corrcoef(c1, pl)[0, 1] < 0:
        coords[:, 0] *= -1
    if np.corrcoef(c2, ar)[0, 1] < 0:
        coords[:, 1] *= -1
    return coords, ratio
