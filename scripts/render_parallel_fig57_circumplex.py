"""
render_parallel_fig57_circumplex.py — Figure 6, the circumplex PCA scatter
(parallel to Anthropic's Figure 57), on the canonical basis.

All 171 emotion vectors projected onto PC1 (x) and PC2 (y) of a global PCA over the
CLEANED L31 vectors, z-scored, with 25 landmark emotions labeled at a uniform
up-right offset. Same PCA as Figures 4 and 8, so coordinates and variance shares
agree with those by construction.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/render_parallel_fig57_circumplex.py
    python3 repo/github_repo/scripts/render_parallel_fig57_circumplex.py --out-prefix PATH

Outputs docs/e1_figures/E1_Figure_6_circumplex.png plus a sibling .audit.json.
NOT LOCKED: the canonical figure-lock freeze holds until the bytes are independently
confirmed.

canonical rebuild 2026-08-18. Four things changed from the earlier version, each gated:
  1. Basis. Hash gate on the vectors — see below.
  2. Axis captions are computed from the data, not hardcoded. the earlier version hardcoded
     "PC1 (26% variance)"; canonical PC1 is 26.73%, which rounds to 27%, and Figure 4
     (locked) auto-computes and prints 27%. Hardcoding would put this figure in
     direct conflict with Figure 4 over the same PCA.
  3. Landmark swaps RECOMPUTED on the canonical positions (see LANDMARKS below). The earlier
     swaps were approved against the earlier dot positions and do not transfer. The canonical
     set is finalized.
  4. y-limit retuned 2.2 -> 2.5: on canonical "playful" sits at pc2 = 2.2409 and was
     outside the earlier axes entirely. Gated below.
"""
import hashlib
import json
import os
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

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

LAYER = 31

OUT_PNG = (_out_prefix + ".png") if _out_prefix else \
    "docs/e1_figures/E1_Figure_6_circumplex.png"
OUT_AUDIT = os.path.splitext(OUT_PNG)[0] + ".audit.json"

# Confirmed values (P1.1). A wrong-basis load reads 26.07 / 13.35 — such a load is
# caught here rather than shipping a figure whose printed variance share is wrong.
EXPECTED_EVR = {"PC1": 26.73, "PC2": 13.38}
EVR_TOL = 0.05


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
# 1. Load and compute global PCA  (CLEANED vectors — never vectors_raw)
# ---------------------------------------------------------------------------
blob = torch.load(VECTORS_PATH, map_location="cpu", weights_only=False)
emotions = list(blob["emotions"])
V = np.asarray(blob["vectors"].float())[:, LAYER, :]

V_centered = V - V.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(V_centered, full_matrices=False)
evr = (S ** 2) / (S ** 2).sum() * 100.0
coords = U[:, :2] * S[:2]

for _k, _want in EXPECTED_EVR.items():
    _got = float(evr[int(_k[-1]) - 1])
    if abs(_got - _want) > EVR_TOL:
        sys.exit("FATAL: %s explained variance %.4f%% != expected canonical %.2f%% "
                 "(tol %.2f) — a wrong-basis load reads 26.07/13.35 here."
                 % (_k, _got, _want, EVR_TOL))
print("[evr] PC1 %.4f%%  PC2 %.4f%%  PC3 %.4f%% (gate clean)" % (evr[0], evr[1], evr[2]))

# Standardize to match Figure 8 scale properties
pc1 = coords[:, 0] / coords[:, 0].std()
pc2 = coords[:, 1] / coords[:, 1].std()

# ---------------------------------------------------------------------------
# 2. Orient via directional sign-flip guardrails, and VERIFY the result
# ---------------------------------------------------------------------------
POS = ["happy","joyful","content","blissful","delighted","cheerful","optimistic","loving"]
NEG = ["afraid","sad","angry","miserable","panicked","terrified","depressed","heartbroken"]
HI  = ["enraged","panicked","ecstatic","furious","terrified","hysterical","thrilled","alarmed"]
LO  = ["calm","serene","sleepy","relaxed","listless","peaceful","droopy","at ease"]

am = lambda c, ns: float(c[[emotions.index(x) for x in ns if x in emotions]].mean())
_flip1 = am(pc1, POS) < am(pc1, NEG)
_flip2 = am(pc2, HI) < am(pc2, LO)
if _flip1: pc1 *= -1
if _flip2: pc2 *= -1

_m1 = am(pc1, POS) - am(pc1, NEG)
_m2 = am(pc2, HI) - am(pc2, LO)
if _m1 <= 0 or _m2 <= 0:
    sys.exit("FATAL: orientation guardrail did not resolve (margins PC1 %+.4f PC2 %+.4f)"
             % (_m1, _m2))

# Verify against full ranks, not the sparse printed labels: P1.1 fixes these poles.
_o1, _o2 = np.argsort(pc1), np.argsort(pc2)
_poles = {
    "pc1_high": [emotions[i] for i in _o1[-3:][::-1]],
    "pc1_low":  [emotions[i] for i in _o1[:3]],
    "pc2_high": [emotions[i] for i in _o2[-3:][::-1]],
    "pc2_low":  [emotions[i] for i in _o2[:3]],
}
_EXPECTED_POLES = {
    "pc1_high": ["optimistic", "kind", "cheerful"],
    "pc1_low":  ["terrified", "hysterical", "frightened"],
    "pc2_high": ["playful", "outraged", "indignant"],
    "pc2_low":  ["nostalgic", "serene", "depressed"],
}
if _poles != _EXPECTED_POLES:
    sys.exit("FATAL: PCA poles do not match the confirmed canonical (P1.1) poles.\n"
             "  got      %s\n  expected %s" % (_poles, _EXPECTED_POLES))
print("[orient] flips PC1=%s PC2=%s; margins PC1 %+.4f PC2 %+.4f; poles match P1.1"
      % (_flip1, _flip2, _m1, _m2))

# ---------------------------------------------------------------------------
# 3. Landmark labels (25)
# ---------------------------------------------------------------------------
# Anthropic's Figure 57 landmark set, placed by Anthropic's convention: EVERY label at
# the identical fixed offset relative to its own dot (up and to the right, same
# distance for all). Uniformity is the disambiguator, so label/dot and dot/dot
# OVERLAPS ARE ACCEPTED (as in the original). The one thing that must NOT happen is
# one label WORD overlapping another label word.
ANTHROPIC_25 = {
    "outraged", "annoyed", "hysterical", "nervous", "tormented", "sluggish",
    "stuck", "depressed", "sentimental", "serene", "loving", "compassionate",
    "sympathetic", "hopeful", "happy", "cheerful", "enthusiastic", "playful",
    "amused", "excited", "proud", "vengeful", "stubborn", "bored", "self-conscious",
}

# The canonical PCA moves the dots, so the collision set differs from the earlier version. Recomputed on the
# canonical coordinates at the axis limits below: four of Anthropic's words collide —
# compassionate/sympathetic, excited/vengeful, sentimental/serene, sluggish/tormented.
# Each is resolved one-for-one by the NEAREST same-region word whose label does not
# collide, which is the same rule that produced the earlier swaps.
#
# The earlier swaps (vengeful->spiteful, compassionate->content, happy->delighted, approved
# 2026-08-07 against the earlier positions) are deliberately NOT carried forward: on canonical
# they leave four collisions standing, and compassionate->content makes its own pair
# worse (73.5px overlap vs 16.4px for the unswapped word).
#
# STATUS: landmark set finalized — the four replacements below stand as the chosen
# labels; none is exchanged for a runner-up. This fixes the LANDMARK SET only; it is
# NOT a lock instruction.
APPROVED_SWAPS = {
    "sympathetic": "calm",       # moves 0.077 in PC space; keeps compassionate
    "vengeful":    "thrilled",   # moves 0.134; keeps excited
    "sentimental": "lonely",     # moves 0.316; keeps serene
    "sluggish":    "sullen",     # moves 0.163; keeps tormented
}
SWAPS_APPROVED = True

LANDMARKS = {APPROVED_SWAPS.get(w, w) for w in ANTHROPIC_25}
assert len(LANDMARKS) == 25, "landmark set is not 25 words after swaps"
_missing = sorted(w for w in LANDMARKS if w not in emotions)
if _missing:
    sys.exit("FATAL: landmark words absent from the 171: %s" % _missing)

# ---------------------------------------------------------------------------
# 4. Canvas + density convention
# ---------------------------------------------------------------------------
figstyle.apply()
DENS = figstyle.apply_parallel_density()  # parallel-panel fonts / margins
# Dense 171-point scatter: ~2/3 the convention marker diameter (area proportional to
# s) to match the original.
CIRCUMPLEX_MARKER_S = int(round(DENS["marker_s"] * (2.0 / 3.0) ** 2))  # -> 67
fig, ax = plt.subplots(figsize=(10, 8.5))

# Axes geometry. ymax raised 2.2 -> 2.5 for canonical: "playful" sits at pc2 = 2.2409, i.e.
# outside the earlier limits entirely (its dot AND its label would have been clipped), and
# 2.5 is the first clean step that also seats the topmost label inside the axes.
XLIM = (-2.5, 3.2)
YLIM = (-2.4, 2.5)
_oob = [(emotions[i], float(pc1[i]), float(pc2[i])) for i in range(len(emotions))
        if not (XLIM[0] <= pc1[i] <= XLIM[1] and YLIM[0] <= pc2[i] <= YLIM[1])]
if _oob:
    sys.exit("FATAL: %d point(s) fall outside the axis limits — retune XLIM/YLIM: %s"
             % (len(_oob), _oob))
print("[limits] all 171 points inside x %s y %s (pc1 [%.4f, %.4f], pc2 [%.4f, %.4f])"
      % (XLIM, YLIM, pc1.min(), pc1.max(), pc2.min(), pc2.max()))

ax.set_xlim(list(XLIM)); ax.set_ylim(list(YLIM))
ax.set_xticks([-2, -1, 0, 1, 2, 3]); ax.set_yticks([-2, -1, 0, 1, 2])
fig.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.12)

# Origin crosshairs behind the points, then the scatter
ax.axhline(0, color="#dcdcdc", linewidth=0.8, linestyle="-", zorder=1)
ax.axvline(0, color="#dcdcdc", linewidth=0.8, linestyle="-", zorder=1)
ax.scatter(pc1, pc2, color="#7fb4d7", edgecolors="white", linewidths=0.4,
           s=CIRCUMPLEX_MARKER_S, zorder=2, alpha=0.95)

# Title, axis labels, spines, ticks. The variance shares are COMPUTED, not hardcoded,
# so this figure and Figure 4 cannot disagree about the same PCA.
ax.set_title("Emotion Vector PCA Projections (Gemma 2 27B)", color=figstyle.TITLE_RUST,
             fontsize=DENS["title_fontsize"], fontweight="normal", pad=34)
ax.set_xlabel("PC1 (%.0f%% variance)" % evr[0], fontweight="normal",
              fontsize=DENS["axis_label_fontsize"], labelpad=10)
ax.set_ylabel("PC2 (%.0f%% variance)" % evr[1], fontweight="normal",
              fontsize=DENS["axis_label_fontsize"], labelpad=10)
for _sp in ("top", "right"):
    ax.spines[_sp].set_visible(False)
for _sp in ("left", "bottom"):
    ax.spines[_sp].set_visible(True); ax.spines[_sp].set_color("#555555"); ax.spines[_sp].set_linewidth(0.8)
ax.tick_params(axis="both", colors="#333333", labelsize=DENS["tick_labelsize"], width=0.8)

# ---------------------------------------------------------------------------
# 5. Labels — every label at the SAME fixed offset relative to its dot
# ---------------------------------------------------------------------------
# The constant relative position is what tells the reader which dot each word labels,
# so label/dot and dot/dot overlaps are accepted (as in the original) — no per-label
# direction, no overlap-avoidance, no leader lines. Gap ~0.75 marker radius; dx = dy.
LX = LY = 0.0432
_texts = {}
for i, word in enumerate(emotions):
    if word not in LANDMARKS:
        continue
    _texts[word] = ax.text(pc1[i] + LX, pc2[i] + LY, word,
                           fontsize=DENS["point_label_fontsize"],
                           color="#222222", ha="left", va="bottom", zorder=3)

# Live gate: no label WORD may overlap another label word. This is the whole reason the
# swap set exists, so it is asserted at render time rather than trusted.
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_bb = {w: t.get_window_extent(renderer=_rend) for w, t in _texts.items()}
_ws = sorted(_bb)
_collide = []
for _i in range(len(_ws)):
    for _j in range(_i + 1, len(_ws)):
        _a, _b = _bb[_ws[_i]], _bb[_ws[_j]]
        if min(_a.x1, _b.x1) > max(_a.x0, _b.x0) and min(_a.y1, _b.y1) > max(_a.y0, _b.y0):
            _collide.append((_ws[_i], _ws[_j]))
if _collide:
    sys.exit("FATAL: label words overlap — the swap set no longer resolves the canonical "
             "layout: %s" % _collide)
print("[labels] %d landmarks placed, zero word-word collisions (gate clean)" % len(_texts))

# ---------------------------------------------------------------------------
# 6. Save + audit
# ---------------------------------------------------------------------------
if os.path.dirname(OUT_PNG):
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=DENS["savefig_pad_inches"])

import sklearn  # noqa: E402  (recorded for provenance only)
audit = {
    "figure": "Figure 6 (circumplex PCA scatter; parallel to Anthropic Figure 57)",
    "basis": "canonical",
    "source_vectors": VECTORS_PATH,
    "source_vectors_sha256": _got_sha,
    "layer": LAYER,
    "slice": "blob['vectors'] (CLEANED) — never vectors_raw",
    "explained_variance_percent": {"PC1": float(evr[0]), "PC2": float(evr[1]),
                                   "PC3": float(evr[2])},
    "axis_labels": {"x": "PC1 (%.0f%% variance)" % evr[0],
                    "y": "PC2 (%.0f%% variance)" % evr[1]},
    "orientation": {
        "flip_pc1": bool(_flip1), "flip_pc2": bool(_flip2),
        "anchor_margin_pc1": _m1, "anchor_margin_pc2": _m2,
        "poles": _poles,
    },
    "axis_limits": {"xlim": list(XLIM), "ylim": list(YLIM),
                    "note": "ymax raised 2.2 -> 2.5 for canonical; 'playful' at pc2 "
                            "2.2409 was outside the earlier limits"},
    "landmarks": {
        "anthropic_original_25": sorted(ANTHROPIC_25),
        "rendered_25": sorted(LANDMARKS),
        "swaps": APPROVED_SWAPS,
        "swaps_status": ("FINALIZED (landmark set only — NOT a lock instruction)"
                     if SWAPS_APPROVED else "PROPOSED — awaiting approval"),
        "ha_swaps_not_carried_forward": {"vengeful": "spiteful",
                                         "compassionate": "content",
                                         "happy": "delighted"},
        "collisions_resolved": [["compassionate", "sympathetic"],
                                ["excited", "vengeful"],
                                ["sentimental", "serene"],
                                ["sluggish", "tormented"]],
        "label_offset": {"dx": LX, "dy": LY},
    },
    "partition_free": True,
    "coords": {e: [float(pc1[i]), float(pc2[i])] for i, e in enumerate(emotions)},
    "env": {"numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__, "torch": torch.__version__},
}
with open(OUT_AUDIT, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=1)

print("wrote %s" % OUT_PNG)
print("  png   sha256 %s" % sha256_file(OUT_PNG))
print("wrote %s" % OUT_AUDIT)
print("  audit sha256 %s" % sha256_file(OUT_AUDIT))
print("  env: numpy %s / matplotlib %s / scikit-learn %s"
      % (np.__version__, matplotlib.__version__, sklearn.__version__))
print("  landmark swaps: %s" % ("FINALIZED" if SWAPS_APPROVED
                                else "PROPOSED — not finalized"))
print("  NOT LOCKED.")
