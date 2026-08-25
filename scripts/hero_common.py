#!/usr/bin/env python3
"""Shared helpers for the three hero-figure scripts.
PC orientation is fixed by anchor emotion sets chosen a priori (never fit to
the human norms), matching the convention used in the P6.2 geometry checks."""
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

import csv
from pathlib import Path

import numpy as np
import torch

VECTORS = Path("data/vectors/emotion_vectors_171.pt")
RM_CSV = Path("data/ratings/russell_mehrabian_1977.csv")
LAYER = 31

POS = ["happy", "joyful", "content", "blissful", "delighted", "cheerful", "optimistic", "loving"]
NEG = ["afraid", "sad", "angry", "miserable", "panicked", "terrified", "depressed", "heartbroken"]
HI = ["enraged", "panicked", "ecstatic", "furious", "terrified", "hysterical", "thrilled", "alarmed"]
LO = ["calm", "serene", "sleepy", "relaxed", "listless", "peaceful", "droopy", "at ease"]


def load_vectors(layer=LAYER):
    p = torch.load(VECTORS, map_location="cpu", weights_only=False)
    emotions = [e.lower() for e in p["emotions"]]
    V = p["vectors"][:, layer, :].float().numpy()
    return emotions, V, p


def pca_2d(V, emotions):
    """Return PC1, PC2 (valence, arousal) sign-oriented by anchors. Global fit."""
    Vc = V - V.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Vc, full_matrices=False)
    pc1, pc2 = U[:, 0] * S[0], U[:, 1] * S[1]
    idx = {e: i for i, e in enumerate(emotions)}
    m = lambda comp, names: float(np.mean([comp[idx[x]] for x in names if x in idx]))
    if m(pc1, POS) < m(pc1, NEG):
        pc1 = -pc1
    if m(pc2, HI) < m(pc2, LO):
        pc2 = -pc2
    var = (S ** 2) / (S ** 2).sum()
    return pc1, pc2, var


def load_rm():
    """Return {emotion_lower: (pleasure, arousal)}."""
    rm = {}
    for row in csv.DictReader(open(RM_CSV)):
        rm[row["Term"].strip().lower()] = (float(row["Pleasure Mean"]), float(row["Arousal Mean"]))
    return rm


def valence_color(v, vmin, vmax):
    """Red (low) -> yellow -> green (high), matching the affective convention."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    return cm.RdYlGn(Normalize(vmin, vmax)(v))


def zscore(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-12)
