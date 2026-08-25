#!/usr/bin/env python3
"""Adjacent-layer RSA(N, N+1) — cosine-of-cosine between consecutive layers
(Anthropic Fig-9 method), on the cleaned vectors.

Producer of record for geometry_out/s6_layers.py also emits this array; this is the
standalone artifact geometry_out/adjacent_rsa.json. Single-basis (canonical); torch-free.

Run from the repo root:
    python scripts/adjacent_rsa.py
"""
import sys, os, json, time, hashlib
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

def adj(path):
    p = load_pt(path); V = np.asarray(p["vectors"], dtype=np.float64); n, L, d = V.shape
    iu = np.triu_indices(n, 1)
    tris = [cosm(V[:, l, :])[iu] for l in range(L)]
    cos = [float(np.dot(tris[N], tris[N + 1]) / (np.linalg.norm(tris[N]) * np.linalg.norm(tris[N + 1]))) for N in range(L - 1)]
    pear = [float(np.corrcoef(tris[N], tris[N + 1])[0, 1]) for N in range(L - 1)]
    return cos, pear

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

cos, pear = adj(VEC)
below = [N for N in range(len(cos)) if cos[N] < 0.97]
out = {
    "item": "adjacent-layer RSA(N,N+1), cosine-of-cosine (Anthropic Fig-9 method), cleaned vectors",
    "basis": "canonical",
    "adjacent_rsa_cos": [round(x, 4) for x in cos],           # index N = pair (LN, LN+1), N=0..44
    "adjacent_rsa_pearson": [round(x, 4) for x in pear],
    "true_RSA_L21_L22": round(cos[21], 4),
    "true_RSA_L43_L44": round(cos[43], 4),
    "pairs_below_0p97": [{"pair": f"L{N}->L{N+1}", "rsa": round(cos[N], 4)} for N in below],
    "n_below_0p97": len(below),
    "discontinuity_recount": ("Three adjacent-layer pairs fall below 0.97: the two seam steps "
        "L21->L22 (0.9425) and L43->L44 (0.9329), plus a marginal early-layer dip "
        "L1->L2 (0.9603, 0.01 below threshold)."),
    "paper_facing": "Seam step rounds to 0.94 (true 0.9425); three pairs sit below 0.97, so an 'exactly two discontinuities' description does not hold.",
    "cos_vs_pearson_max_delta": round(max(abs(cos[i] - pear[i]) for i in range(len(cos))), 6),
    "_provenance": {"basis": "canonical", "vectors": VEC, "vectors_sha256": _sha(VEC),
        "method": "mirror of s6_layers.py cosm() cos-of-cos; adjacent pairs (not to-L31)", "layers": 46,
        "note": "version-independent (numpy geometry, no k-means/lens)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
}
json.dump(out, open("geometry_out/adjacent_rsa.json", "w"), indent=1)
print("wrote geometry_out/adjacent_rsa.json | L21->L22:", out["true_RSA_L21_L22"],
      "| L43->L44:", out["true_RSA_L43_L44"], "| n<0.97:", out["n_below_0p97"])
