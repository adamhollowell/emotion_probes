"""
render_parallel_fig9_cross_layer_rsa.py — Figure 7, cross-layer RSA heatmap
(parallel to Anthropic's Figure 9), on the canonical basis.

Layer x layer cosine-of-cosine RSA between emotion-probe geometries at 14 evenly
spaced layers. For each sampled layer: the 171x171 pairwise-cosine matrix of the
CLEANED emotion vectors, upper triangle vectorised; then the cosine similarity between
those per-layer vectors, across layers. Anthropic's Figure 9 method exactly. Finding
P3.1 (cross-layer representational stability). Partition-free — pure geometry.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/render_parallel_fig9_cross_layer_rsa.py
    python3 repo/github_repo/scripts/render_parallel_fig9_cross_layer_rsa.py --out-prefix PATH

Outputs docs/e1_figures/E1_Figure_7_crosslayer_rsa.png plus a sibling
.audit.json. NOT LOCKED: the canonical figure-lock freeze holds until the RSA data is
independently confirmed.

canonical rebuild 2026-08-18. The METHOD is unchanged (Anthropic-faithful): 14 layers
linspace(12,40).round(), cosine-of-cosine RSA, viridis vmin 0.8 / vmax 1.0, Early /
Early-Mid / Mid-Late / Late band labels. What changed is provenance, brought up to the
Fig 2/4 standard:
  1. Basis hash gate on the vectors (the producer previously had NONE).
  2. figstyle is now a HARD import. It used to sit in a try/except that silently fell
     back to different fonts and a hardcoded rust colour on any import failure — i.e.
     it could ship DIFFERENT BYTES with no error. A figure that is gate-checked and
     lockable cannot have a silent style fallback.
  3. Content gate (L31 cosine spot-check + RSA diagonal) catches a wrong-basis or
     wrong-layer load even if the hash somehow passed.
  4. --out-prefix, a working output path, and an .audit.json.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # HARD import — no silent style fallback (see header note 2)

# ---------------------------------------------------------------------------
# BASIS GATE — hash, never filename
# ---------------------------------------------------------------------------
# The vectors are identified by their sha256, never by the name `emotion_vectors_171.pt`.
# The hash is the only identifier that means anything here.
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
DEFAULT_PATH = "data/vectors/emotion_vectors_171.pt"

# usage: [VECTORS_PATH] [--out-prefix PREFIX]
_args = sys.argv[1:]
_out_prefix = None
if "--out-prefix" in _args:
    _i = _args.index("--out-prefix")
    _out_prefix = _args[_i + 1]
    del _args[_i:_i + 2]
VECTORS_PATH = _args[0] if _args else DEFAULT_PATH

OUT_PNG = (_out_prefix + ".png") if _out_prefix else \
    "docs/e1_figures/E1_Figure_7_crosslayer_rsa.png"
OUT_AUDIT = os.path.splitext(OUT_PNG)[0] + ".audit.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_got_sha = sha256_file(VECTORS_PATH)
if _got_sha != EXPECTED_VEC_SHA:
    sys.exit("FATAL: %s sha256 %s != expected canonical basis %s (wrong basis or stray "
             "working copy)" % (VECTORS_PATH, _got_sha, EXPECTED_VEC_SHA))
print("[basis] %s sha256 %s == canonical basis (verified)" % (VECTORS_PATH, _got_sha[:12]))

# ---------------------------------------------------------------------------
# LOAD + RSA  (CLEANED vectors — never vectors_raw)
# ---------------------------------------------------------------------------
blob = torch.load(VECTORS_PATH, map_location="cpu", weights_only=False)
emotions = list(blob["emotions"])
V = np.asarray(blob["vectors"].float())          # CLEANED (not vectors_raw)
n = V.shape[0]


def cosm(Vl):
    Vn = Vl / np.clip(np.linalg.norm(Vl, axis=1, keepdims=True), 1e-12, None)
    return Vn @ Vn.T


# Content gate: catch a wrong-basis/wrong-layer load even if the hash somehow passed.
# Same L31 spot values Figure 2 uses. A raw-vector load or a layer-index slip moves
# these well outside the tolerance.
_idx = {e: i for i, e in enumerate(emotions)}
_LAYER_SPOT = 31
_SPOT = [("afraid", "scared", 0.9867), ("joyful", "happy", 0.9766),
         ("compassionate", "empathetic", 0.9655)]
_SPOT_TOL = 0.002
_cos31 = cosm(V[:, _LAYER_SPOT, :])
for _a, _b, _want in _SPOT:
    _got = float(_cos31[_idx[_a], _idx[_b]])
    if abs(_got - _want) > _SPOT_TOL:
        sys.exit("FATAL: L%d cos(%s,%s) = %.4f != expected canonical %.4f (tol %.3f) — "
                 "wrong basis or wrong layer slice"
                 % (_LAYER_SPOT, _a, _b, _got, _want, _SPOT_TOL))
print("[content] L31 cosine spot-check clean: afraid/scared %.4f, joyful/happy %.4f, "
      "compassionate/empathetic %.4f"
      % (_cos31[_idx["afraid"], _idx["scared"]],
         _cos31[_idx["joyful"], _idx["happy"]],
         _cos31[_idx["compassionate"], _idx["empathetic"]]))

# RSA: 14 sampled layers, cosine-of-cosine (Anthropic Fig-9 method). Do NOT switch to a
# Pearson variant — cosine-of-cosine matches Anthropic's wording and the Pearson
# variant differs only ~1e-5 (P3.1/P3.8).
samp = np.linspace(12, 40, 14).round().astype(int).tolist()
assert 31 in samp, "sampled layers must include the canonical L31"
iu = np.triu_indices(n, 1)
tris = np.stack([cosm(V[:, l, :])[iu] for l in samp])
tris = tris / np.clip(np.linalg.norm(tris, axis=1, keepdims=True), 1e-12, None)
M = tris @ tris.T

# Diagonal must be ~1 by construction (each layer's geometry is identical to itself).
_diag = M.diagonal()
if not np.allclose(_diag, 1.0, atol=1e-4):
    sys.exit("FATAL: RSA diagonal is not ~1.0 (min %.6f, max %.6f) — the RSA is "
             "malformed" % (float(_diag.min()), float(_diag.max())))
print("[content] RSA diagonal ~1.0 (min %.6f, max %.6f)" % (float(_diag.min()), float(_diag.max())))

# Adjacent-pair minimum: the sampled-sweep echo of the P3.1 emergence seam. On canonical it
# should land in the L22-26 seam region. This is a DIFFERENT cut from the full-46-layer
# adjacent steps in adjacent_rsa.json (sampled 12-40, layer x layer) — same story,
# different numbers; do not expect equality.
_adj = [float(M[i, i + 1]) for i in range(len(samp) - 1)]
_ai = int(np.argmin(_adj))
_adj_min = _adj[_ai]
_adj_from, _adj_to = samp[_ai], samp[_ai + 1]
_in_seam = (22 <= _adj_from <= 26) or (22 <= _adj_to <= 26)
print("[seam] adjacent min %.4f at L%d->L%d %s"
      % (_adj_min, _adj_from, _adj_to,
         "(in the L22-26 seam region, as expected)" if _in_seam
         else "*** WARNING: NOT in the L22-26 seam region — investigate ***"))

# ---------------------------------------------------------------------------
# RENDER  (method + style unchanged)
# ---------------------------------------------------------------------------
figstyle.apply()
TITLE = figstyle.TITLE_RUST
DENS = figstyle.apply_parallel_density(kind="heatmap")

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(M, cmap="viridis", vmin=0.8, vmax=1.0, aspect="equal", interpolation="nearest")
ax.set_xlim(-0.5, len(samp) - 0.5); ax.set_ylim(len(samp) - 0.5, -0.5)
# Band labels (matches Anthropic). The numeric-layer mode is a blog/diagnostic variant
# (P3.8), not this figure — keep "bands".
LABEL_STYLE = "bands"
if LABEL_STYLE == "bands":
    ticks = [0, 4, 9, len(samp) - 1]; labels = ["Early", "Early-Mid", "Mid-Late", "Late"]
else:
    ticks = list(range(len(samp))); labels = samp
ax.set_xticks(ticks); ax.set_xticklabels(labels, color="#555555", fontsize=DENS["tick_labelsize"])
ax.set_yticks(ticks); ax.set_yticklabels(labels, color="#555555", fontsize=DENS["tick_labelsize"])
ax.set_title("Cross-Layer Similarity of Emotion Probe Structure (Gemma 2 27B)",
             color=TITLE, fontsize=DENS["title_fontsize"], pad=16)
ax.set_xlabel("Layer", fontsize=DENS["axis_label_fontsize"], labelpad=10)
ax.set_ylabel("Layer", fontsize=DENS["axis_label_fontsize"], labelpad=10)
for s in ["top", "bottom", "left", "right"]:
    ax.spines[s].set_visible(True); ax.spines[s].set_color("#555555"); ax.spines[s].set_linewidth(0.8)
cb = fig.colorbar(im, fraction=0.046, pad=0.04, shrink=0.78, ticks=[0.8, 0.9, 1.0], format="%.1f")
cb.set_label("Cosine Similarity", fontsize=DENS["axis_label_fontsize"], labelpad=22, rotation=270)
cb.ax.tick_params(labelsize=DENS["tick_labelsize"], color="#555555")
cb.outline.set_color("#555555"); cb.outline.set_linewidth(0.8)
fig.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.12)

if os.path.dirname(OUT_PNG):
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", pad_inches=DENS["savefig_pad_inches"])

# ---------------------------------------------------------------------------
# AUDIT
# ---------------------------------------------------------------------------
audit = {
    "figure": "Figure 7 (cross-layer RSA heatmap; parallel to Anthropic Figure 9)",
    "finding": "P3.1 (cross-layer representational stability)",
    "basis": "canonical",
    "source_vectors": VECTORS_PATH,
    "source_vectors_sha256": _got_sha,
    "slice": "blob['vectors'] (CLEANED) — never vectors_raw",
    "method": ("cosine-of-cosine RSA (Anthropic Fig-9): per layer, upper triangle of "
               "the 171x171 pairwise-cosine matrix, L2-normalised, then cosine "
               "similarity across layers. viridis vmin 0.8 vmax 1.0; Early/Early-Mid/"
               "Mid-Late/Late band labels."),
    "sampled_layers": samp,
    "content_gate": {
        "layer": _LAYER_SPOT,
        "spot_cosines": {"%s|%s" % (a, b): float(_cos31[_idx[a], _idx[b]])
                         for a, b, _ in _SPOT},
        "rsa_diagonal_min": float(_diag.min()),
        "rsa_diagonal_max": float(_diag.max()),
    },
    "rsa_matrix": [[float(x) for x in row] for row in M],
    "rsa_diagonal": [float(x) for x in _diag],
    "adjacent_pairs": {
        "values": _adj,
        "layer_pairs": [[samp[i], samp[i + 1]] for i in range(len(samp) - 1)],
        "min_value": _adj_min,
        "min_layer_pair": [_adj_from, _adj_to],
        "min_in_seam_region_L22_26": bool(_in_seam),
        "note": ("sampled-sweep echo of the P3.1 emergence seam; a DIFFERENT cut from "
                 "the full-46-layer adjacent steps in adjacent_rsa.json "
                 "(true L21->L22 0.9425, L43->L44 0.9329) — same story, different "
                 "numbers"),
    },
    "matrix_range": {"min": float(M.min()), "max": float(M.max())},
    "partition_free": True,
    "env": {"numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "torch": torch.__version__},
}
with open(OUT_AUDIT, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=1)

print("wrote %s" % OUT_PNG)
print("  png   sha256 %s" % sha256_file(OUT_PNG))
print("wrote %s" % OUT_AUDIT)
print("  audit sha256 %s" % sha256_file(OUT_AUDIT))
print("  env: numpy %s / matplotlib %s" % (np.__version__, matplotlib.__version__))
print("  NOTE: NOT LOCKED — final lock awaits an independent RSA "
      "re-derivation before the figure is locked.")
