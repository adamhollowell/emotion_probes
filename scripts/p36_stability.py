#!/usr/bin/env python3
"""P3.6 — per-emotion neighborhood stability: mean over 46 layers of the top-10
cosine-neighbor overlap with each emotion's L31 neighborhood.

Producer of record for geometry_out/p36_stability.json. Single-basis (canonical);
torch-free, version-independent (pure cosine geometry, no k-means/lens).

Run from the repo root:
    python scripts/p36_stability.py
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, "outputs")  # torch-free load_pt ships here; falls back to torch
try:
    from load_pt import load_pt
except Exception:
    import torch
    def load_pt(p): return torch.load(p, map_location="cpu", weights_only=False)

VEC = sys.argv[1] if len(sys.argv) > 1 else "data/vectors/emotion_vectors_171.pt"

def cosm(X):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None); return Xn @ Xn.T

def overlaps(path):
    p = load_pt(path); em = [str(e) for e in p["emotions"]]; V = np.asarray(p["vectors"], dtype=np.float64)
    n, L, d = V.shape
    per_layer_top10 = []
    for l in range(L):
        o = np.argsort(-cosm(V[:, l, :]), axis=1)
        per_layer_top10.append([set(o[i, 1:11]) for i in range(n)])
    L31 = per_layer_top10[31]
    ov = np.zeros((n, L))
    for l in range(L):
        for i in range(n): ov[i, l] = len(per_layer_top10[l][i] & L31[i])
    return em, ov

def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

em, ov = overlaps(VEC)
n, L = ov.shape
incl = ov.mean(1)                               # mean over 46 layers (incl L31 self=10)
excl = ov[:, [l for l in range(L) if l != 31]].mean(1)
BANDS = {"early_L0_11": range(0, 12), "mid_L12_21": range(12, 22), "seam_L22_26": range(22, 27),
         "canon_L27_34": range(27, 35), "late_L35_40": range(35, 41), "final_L41_45": range(41, 46)}
band = {bn: round(float(ov[:, list(r)].mean()), 2) for bn, r in BANDS.items()}
order = np.argsort(incl)
least = [(em[i], round(float(incl[i]), 2)) for i in order[:6]]
most = [(em[i], round(float(incl[i]), 2)) for i in order[-5:][::-1]]
bored = round(float(incl[em.index("bored")]), 2)
top7i = [em[i] for i in np.argsort(incl)[:7]]; top7e = [em[i] for i in np.argsort(excl)[:7]]
out = {"item": "P3.6 per-emotion neighborhood stability (top-10 overlap with L31)", "basis": "canonical",
       "metric": "mean over 46 layers of |top10(e,L) ∩ top10(e,L31)| on cleaned vectors",
       "band_overlap": band,
       "per_emotion_overlap_incl_L31": {em[i]: round(float(incl[i]), 3) for i in range(n)},
       "per_emotion_overlap_excl_L31": {em[i]: round(float(excl[i]), 3) for i in range(n)},
       "least_stable": least, "most_stable": most,
       "convention_robust_top7": bool(set(top7i) == set(top7e)),
       "_provenance": {"basis": "canonical", "vectors": VEC, "vectors_sha256": _sha(VEC),
                       "method": "torch-free cosine geometry, version-independent",
                       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
json.dump(out, open("geometry_out/p36_stability.json", "w"), indent=1)
print("wrote geometry_out/p36_stability.json | band:", band, "| bored:", bored)
