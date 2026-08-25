#!/usr/bin/env python3
"""
layerwise_va_global.py — per-layer global-PCA valence/arousal alignment (P3.4 producer).

Reconstructed producer, validated byte-identical against the canonical per-layer
valence/arousal outputs. Emits geometry_out/layerwise_va_global_pca.csv via the
"global PCA over all 171 emotions, then restrict to the 45 Russell-Mehrabian overlap"
pathway, adding the CSV/summary schema the release consumes.

Method (per layer L):
  X = cleaned vectors[:, L, :]; center columns; SVD; scores = U*S; var = S^2 / sum(S^2).
  Orient PC1 by POS>NEG anchor means, PC2 and PC3 by HI>LO anchor means.
  r_pc1_pleasure = corr(PC1[45], RM pleasure); r_pc2_arousal = corr(PC2[45], RM arousal).
  cross terms = the off-diagonal correlations; arousal_best_pc = 1-indexed argmax_k
  |corr(PC_k[45], arousal)| over the first 10 PCs.

Self-anchors (canonical basis): L31 pc1_var 26.732, r_pc1_pleasure 0.720,
r_pc2_arousal 0.668 (fig8-consistent). Asserted before writing.

Run from repo root:  python scripts/layerwise_va_global.py
Outputs: geometry_out/layerwise_va_global_pca.csv , geometry_out/layerwise_va_global_pca_summary.json
"""
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

import csv, json, sys
from pathlib import Path
import numpy as np
import torch

VEC = "data/vectors/emotion_vectors_171.pt"
RM = "data/ratings/russell_mehrabian_1977.csv"
OUT_CSV = "geometry_out/layerwise_va_global_pca.csv"
OUT_JSON = "geometry_out/layerwise_va_global_pca_summary.json"

POS = ["happy", "joyful", "content", "blissful", "delighted", "cheerful", "optimistic", "loving"]
NEG = ["afraid", "sad", "angry", "miserable", "panicked", "terrified", "depressed", "heartbroken"]
HI = ["enraged", "panicked", "ecstatic", "furious", "terrified", "hysterical", "thrilled", "alarmed"]
LO = ["calm", "serene", "sleepy", "relaxed", "listless", "peaceful", "droopy", "at ease"]


def main():
    blob = torch.load(VEC, map_location="cpu", weights_only=False)
    em = [str(e) for e in blob["emotions"]]
    V = np.asarray(blob["vectors"], dtype=np.float64)  # (171, 46, 4608) cleaned
    n, L, d = V.shape
    am = lambda c, ns: float(c[[em.index(x) for x in ns if x in em]].mean())
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])

    rm = {}
    for row in csv.DictReader(open(RM)):
        rm[row["Term"].strip().lower()] = (float(row["Pleasure Mean"]), float(row["Arousal Mean"]))
    m45 = [e for e in em if e in rm]
    i45 = [em.index(e) for e in m45]
    plea = np.array([rm[e][0] for e in m45])
    arou = np.array([rm[e][1] for e in m45])

    rows = []
    for l in range(L):
        X = V[:, l, :]
        Xc = X - X.mean(0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        sc = U * S
        var = S ** 2 / (S ** 2).sum()
        if am(sc[:, 0], POS) < am(sc[:, 0], NEG): sc[:, 0] *= -1
        if am(sc[:, 1], HI) < am(sc[:, 1], LO): sc[:, 1] *= -1
        if am(sc[:, 2], HI) < am(sc[:, 2], LO): sc[:, 2] *= -1
        rv = r(sc[i45, 0], plea)
        ra = r(sc[i45, 1], arou)
        cross_pc1_arousal = r(sc[i45, 0], arou)
        cross_pc2_pleasure = r(sc[i45, 1], plea)
        best_pc = int(np.argmax([abs(r(sc[i45, k], arou)) for k in range(10)])) + 1  # 1-indexed
        rows.append({
            "layer": l,
            "r_pc1_pleasure": rv, "r_pc2_arousal": ra,
            "cross_pc1_arousal": cross_pc1_arousal, "cross_pc2_pleasure": cross_pc2_pleasure,
            "arousal_best_pc": best_pc,
            "pc1_var_pct": float(100 * var[0]), "pc2_var_pct": float(100 * var[1]),
            "pc3_var_pct": float(100 * var[2]),
        })

    # ---- self-anchors (canonical basis) ----
    L31 = rows[31]
    assert abs(L31["pc1_var_pct"] - 26.732) < 0.02, "anchor pc1_var L31 %.4f" % L31["pc1_var_pct"]
    assert abs(L31["r_pc1_pleasure"] - 0.720) < 0.01, "anchor valence L31 %.4f" % L31["r_pc1_pleasure"]
    assert abs(L31["r_pc2_arousal"] - 0.668) < 0.01, "anchor arousal L31 %.4f" % L31["r_pc2_arousal"]

    Path("geometry_out").mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    bandmean = lambda key, lo, hi: float(np.mean([rows[L_][key] for L_ in range(lo, hi + 1)]))
    val = [x["r_pc1_pleasure"] for x in rows]
    aro = [x["r_pc2_arousal"] for x in rows]
    summary = {
        "n_layers": L, "n_overlap": len(m45),
        "basis": "global PCA over all 171, then slice 45",
        "bands_valence": {"early_L8_13": bandmean("r_pc1_pleasure", 8, 13),
                          "seam_L22_26": bandmean("r_pc1_pleasure", 22, 26),
                          "canonical_L27_34": bandmean("r_pc1_pleasure", 27, 34),
                          "final_L44_45": bandmean("r_pc1_pleasure", 44, 45)},
        "bands_arousal": {"early_L8_13": bandmean("r_pc2_arousal", 8, 13),
                          "seam_L22_26": bandmean("r_pc2_arousal", 22, 26),
                          "canonical_L27_34": bandmean("r_pc2_arousal", 27, 34),
                          "final_L44_45": bandmean("r_pc2_arousal", 44, 45)},
        "L31": {k: (float(v) if k != "layer" else float(v)) for k, v in L31.items()},
        "valence_peak_layer": int(np.argmax(val)),
        "arousal_peak_layer": int(np.argmax(aro)),
        "layers_where_arousal_not_on_pc2": [x["layer"] for x in rows if x["arousal_best_pc"] != 2],
        "L31_matches_fig8_0.72_0.67": bool(round(L31["r_pc1_pleasure"], 2) == 0.72
                                           and round(L31["r_pc2_arousal"], 2) == 0.67),
    }
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    print("layerwise_va_global: wrote %s (%d layers) + summary" % (OUT_CSV, L))
    print("  L31 pc1_var=%.3f valence=%.4f arousal=%.4f  (anchors pass)"
          % (L31["pc1_var_pct"], L31["r_pc1_pleasure"], L31["r_pc2_arousal"]))
    print("  L12 pc1_var=%.3f  L45 pc1_var=%.3f" % (rows[12]["pc1_var_pct"], rows[45]["pc1_var_pct"]))


if __name__ == "__main__":
    main()
