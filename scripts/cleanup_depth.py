#!/usr/bin/env python3
"""P1.1 confound-cleanup magnitude: how much the neutral-PC projection changed the
emotion vectors.

Producer of record for geometry_out/cleanup_depth.json. Single-basis (canonical); the
Frobenius/variance figures reproduce from the PUBLIC shipped vectors alone
(`vectors_raw` = before cleanup, `vectors` = after), no neutral set required.

Run from the repo root:
    python scripts/cleanup_depth.py
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

def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

p = load_pt(VEC)
Vc = np.asarray(p["vectors"], dtype=np.float64)       # cleaned  (171, 46, 4608)
Vr = np.asarray(p["vectors_raw"], dtype=np.float64)   # raw (before neutral-PC projection)
n, L, d = Vc.shape

def frob(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))

all_frob = frob(Vc, Vr)                                # whole-array aggregate
l31_frob = frob(Vc[:, 31, :], Vr[:, 31, :])           # L31 slice
per_layer = [round(frob(Vc[:, l, :], Vr[:, l, :]), 4) for l in range(L)]

out = {
    "item": "P1.1 confound-cleanup magnitude (neutral-PC projection: raw vs cleaned emotion vectors)",
    "basis": "canonical",
    "frobenius_all_layers": round(all_frob, 4),            # 0.4253 — the headline P1.1 figure
    "variance_removed_all_layers": round(all_frob ** 2, 3),   # 0.181 (= frob^2); 18.1% removed
    "frobenius_L31": round(l31_frob, 4),                   # 0.4108 — L31 slice only (distinct key; NOT the P1.1 figure)
    "variance_removed_L31": round(l31_frob ** 2, 3),       # 0.169 (16.9%)
    "per_layer_frobenius": per_layer,
    "k_at_L31": 28,
    "k_at_L31_note": ("cleanup depth at L31 = number of neutral-activation PCs (top PCs explaining "
                      ">=50% of neutral variance) projected out. Derived from the neutral set "
                      "(neutral.pt, nonpublic); regenerate per repo_E1_nonpublic/MANIFEST.md."),
    "definition": "Frobenius ||raw - clean|| / ||raw||; variance_removed = that ratio squared.",
    "_provenance": {"basis": "canonical", "vectors": VEC, "vectors_sha256": _sha(VEC),
                    "method": "raw vs cleaned emotion vectors, torch-free; public-reproducible for the Frobenius/variance figures",
                    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
}
json.dump(out, open("geometry_out/cleanup_depth.json", "w"), indent=1)
print("wrote geometry_out/cleanup_depth.json")
print(f"  all-layers Frobenius {out['frobenius_all_layers']} ({out['variance_removed_all_layers']})"
      f" | L31 {out['frobenius_L31']} ({out['variance_removed_L31']}) | k@L31 {out['k_at_L31']}")
