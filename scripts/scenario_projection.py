#!/usr/bin/env python3
"""§10.1 stimulus–response scenario projection — the two scoring matrices.

Producer of record for geometry_out/scenario_projection_matrices.json. Projects the 12
scenario activations onto the 171 emotion vectors and emits BOTH matrices §10.1 describes:
the raw projection (which recovered no scenario under argmax, owing to large per-emotion
baseline offsets) and the column-centered projection (which removes the per-emotion offset).

Inputs, both shipped: the raw scenario activations `data/vectors/scenario_acts.pt`
(12 scenarios x 46 layers x 4608, saved at the last user-turn token, `unitnorm` convention)
and the emotion vectors `data/vectors/emotion_vectors_171.pt`. The saved activations hold all
46 layers, so the analysis can be re-scored at any layer without re-extraction (§10.1);
this producer scores at the analysis layer L31 by default.

Run from the repo root:
    python scripts/scenario_projection.py            # L31
    python scripts/scenario_projection.py 26         # any layer 0..45
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, "outputs")  # torch-free load_pt ships here; falls back to torch
try:
    from load_pt import load_pt
except Exception:
    import torch
    def load_pt(p): return torch.load(p, map_location="cpu", weights_only=False)

VEC = "data/vectors/emotion_vectors_171.pt"
SCN = "data/vectors/scenario_acts.pt"
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 31

def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

p = load_pt(VEC); em = [str(e) for e in p["emotions"]]
V = np.asarray(p["vectors"], dtype=np.float64)[:, LAYER, :]          # (171, 4608) cleaned
s = load_pt(SCN)
A = np.asarray(s["acts"], dtype=np.float64)[:, LAYER, :]             # (12, 4608)
prompts = [str(x) for x in s["prompts"]]

# `unitnorm` convention: project each scenario activation onto the UNIT-normed emotion vector
Vhat = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
M_raw = A @ Vhat.T                                                   # (12, 171)
# column-centering: remove each emotion's per-column baseline offset (mean over the 12 scenarios)
M_centered = M_raw - M_raw.mean(axis=0, keepdims=True)

out = {
    "item": "10.1 stimulus-response scenario projection (raw + column-centered)",
    "basis": "canonical",
    "layer": LAYER,
    "position": s.get("convention", {}).get("position"),
    "projection": "act . (v/||v||); scores 12 scenarios x 171 emotions",
    "n_scenarios": len(prompts), "n_emotions": len(em),
    "emotions": em,
    "scenario_prompts": prompts,
    "raw_matrix": [[round(float(x), 5) for x in row] for row in M_raw],
    "column_centered_matrix": [[round(float(x), 5) for x in row] for row in M_centered],
    "note": ("Raw argmax recovers no scenario (large per-emotion baseline offsets); column-centering "
             "removes the per-emotion offset. §10.1 treats the centered result as a diagnostic clue, "
             "not a finding. Activations for all 46 layers are in scenario_acts.pt; re-score any layer."),
    "_provenance": {"basis": "canonical", "vectors": VEC, "vectors_sha256": _sha(VEC),
                    "scenario_acts": SCN, "scenario_acts_sha256": _sha(SCN),
                    "method": "torch-free; unit-normed emotion vectors; column-centered = per-emotion mean removed",
                    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
}
json.dump(out, open("geometry_out/scenario_projection_matrices.json", "w"), indent=1)
print(f"wrote geometry_out/scenario_projection_matrices.json  (L{LAYER}, {M_raw.shape[0]}x{M_raw.shape[1]})")
print(f"  raw   range [{M_raw.min():.2f}, {M_raw.max():.2f}]  per-scenario argmax emotions:")
print("   ", [em[int(i)] for i in M_raw.argmax(1)])
print(f"  centered argmax emotions:")
print("   ", [em[int(i)] for i in M_centered.argmax(1)])
