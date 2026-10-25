"""Shared plumbing for the Zhang-parallels scripts (Z1/Z2/Z3).

Canonical vectors only. M.9: sha256 every input into a manifest. torch (mmap) preferred,
ptload fallback. See ZHANG_PARALLELS_CLAUDE_CODE_HANDOFF.md.
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "gemma_grid" / "geometry_out/zhang_parallels"

try:
    import torch
    HAVE_TORCH = True
except Exception:  # broken installs raise OSError, not just ImportError
    HAVE_TORCH = False
    sys.path.insert(0, str(REPO))
    import ptload  # torch-free fallback (repo root)

CELLS = {
    "gemma-2-2b-base":  dict(vec="gemma_grid/data/gemma-2-2b-base/vectors/canonical.pt",  layer=17),
    "gemma-2-2b-it":    dict(vec="gemma_grid/data/gemma-2-2b-it/vectors/canonical.pt",    layer=17),
    "gemma-2-9b-base":  dict(vec="gemma_grid/data/gemma-2-9b-base/vectors/canonical.pt",  layer=28),
    "gemma-2-9b-it":    dict(vec="gemma_grid/data/gemma-2-9b-it/vectors/canonical.pt",    layer=28),
    "gemma-2-27b-base": dict(vec="data/vectors/emotion_vectors_171.pt",                      layer=31),
    "gemma-2-27b-it":   dict(vec="gemma_grid/data/gemma-2-27b-it/vectors/canonical.pt",   layer=31),
}
FORBIDDEN = ("canonical_" "HA.pt", "geometry_out_" "HA/")  # superseded basis — hard stop (release: older bases do not ship)

POS = ["happy","joyful","content","blissful","delighted","cheerful","optimistic","loving"]
NEG = ["afraid","sad","angry","miserable","panicked","terrified","depressed","heartbroken"]
HI  = ["enraged","panicked","ecstatic","furious","terrified","hysterical","thrilled","alarmed"]
LO  = ["calm","serene","sleepy","relaxed","listless","peaceful","droopy","at ease"]


class Manifest:
    def __init__(self):
        self.d = {}

    def sha(self, path):
        path = Path(path)
        rel = str(path.relative_to(REPO))
        for bad in FORBIDDEN:
            if bad in rel and "canonical_100T" not in rel and "emotion_vectors_171_100T" not in rel:
                sys.exit(f"HARD STOP: PA-basis input touched: {rel}")
        if rel not in self.d:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for c in iter(lambda: f.read(1 << 22), b""):
                    h.update(c)
            self.d[rel] = h.hexdigest()
        return self.d[rel]

    def write(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(json.dumps(
            {"rule": "M.9 — every input hashed at run time", "inputs_sha256": self.d}, indent=1))


def load_pt(path, mmap=False):
    if HAVE_TORCH:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    return ptload.load(path)


def as_np(x):
    if HAVE_TORCH and isinstance(x, torch.Tensor):
        return x.float().numpy()
    return np.asarray(x, dtype=np.float32)


def load_vectors(cell, man, key="vectors"):
    """key='vectors' (cleaned) or 'vectors_raw'. Returns (emotions, (171,L,d) f32)."""
    p = REPO / CELLS[cell]["vec"]
    man.sha(p)
    d = load_pt(p)
    return list(d["emotions"]), as_np(d[key])


def pca_coords(Vl, k=5):
    Vc = Vl - Vl.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Vc, full_matrices=False)
    return U[:, :k] * S[:k]


def sign_orient(coords, em):
    def amean(c, names):
        idx = [em.index(x) for x in names if x in em]
        return float(c[idx].mean()) if idx else 0.0
    coords = coords.copy()
    if amean(coords[:, 0], POS) < amean(coords[:, 0], NEG): coords[:, 0] *= -1
    if amean(coords[:, 1], HI)  < amean(coords[:, 1], LO):  coords[:, 1] *= -1
    return coords


def load_rm(man):
    p = REPO / "data/ratings/russell_mehrabian_1977.csv"; man.sha(p)
    return {r["Term"].strip().lower(): (float(r["Pleasure Mean"]), float(r["Arousal Mean"]))
            for r in csv.DictReader(open(p))}


def load_nrc(man):
    p = REPO / "data/ratings/nrc_vad_v2_1.csv"; man.sha(p)
    return {r["word"].strip().lower(): (float(r["valence"]), float(r["arousal"]))
            for r in csv.DictReader(open(p))}


def signed_corrs(coords, em, tbl):
    """Plain (signed) correlations of the sign-oriented PC1/PC2 against the norm
    table — the noise_floor.py convention. `coords` must already be sign_orient-ed;
    no further flipping happens here, so a negative r is reported as negative."""
    ov = [i for i, e in enumerate(em) if e.lower() in tbl]
    pl = np.array([tbl[em[i].lower()][0] for i in ov])
    ar = np.array([tbl[em[i].lower()][1] for i in ov])
    rv = float(np.corrcoef(coords[ov, 0], pl)[0, 1])
    ra = float(np.corrcoef(coords[ov, 1], ar)[0, 1])
    return rv, ra, len(ov)
