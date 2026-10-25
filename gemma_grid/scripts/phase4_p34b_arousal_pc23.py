#!/usr/bin/env python3
"""GGE1 family fold-in of E1 P3.4(b): arousal split across the PC2/PC3 plane,
per layer, all six cells. CLEANED vectors -> 100T basis all the way down
(asserted). Adapts the project's depth-arousal analysis to the grid, reusing
noise_floor's load_vectors/pca/load_ratings (the canonical Stage-A machinery).
RUN under lens_env. Output: geometry_out/p34b_arousal_pc23_family.json.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf
from phase4_stageA import CELLS
RATINGS = REPO / "data" / "ratings" / "russell_mehrabian_1977.csv"
OUT = REPO / "gemma_grid" / "geometry_out" / "p34b_arousal_pc23_family.json"


def vec100T(cell):
    v = CELLS[cell]["vec"]
    return ("data/vectors/emotion_vectors_171.pt" if cell == "gemma-2-27b-base"
            else v.replace("canonical.pt", "canonical.pt"))


def z(a):
    a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-12)


def corr(a, b):
    return float(np.corrcoef(z(a), z(b))[0, 1])


def best2d(s2, s3, t):
    z2, z3, zt = z(s2), z(s3), z(t)
    z3o = z3 - (np.dot(z3, z2) / np.dot(z2, z2)) * z2
    return float(np.sqrt(corr(z2, zt) ** 2 + corr(z3o, zt) ** 2))


def main():
    out = {"record": "P3.4b arousal PC2/PC3 split — GGE1 family fold-in (E1 P3.4b)",
           "basis": "100T cleaned vectors (asserted 100T-in-path); RM-45 overlap; per-layer global PCA; "
                    "fixed global PC2 sign oriented at each cell's analysis layer", "cells": {}}
    for cell in CELLS:
        vp = vec100T(cell)
        assert "_HA" not in vp and "_PA" not in vp, f"CANONICALITY: superseded-basis vec path for {cell}: {vp}"   # release: only 100T vectors ship
        em, V = nf.load_vectors(REPO / vp)
        n, L, d = V.shape
        La = CELLS[cell]["layer"]
        idx, pl, ar = nf.load_ratings(RATINGS, em)
        ca, _ = nf.pca(V[:, La, :], 5)
        fs = 1.0 if corr(ca[idx, 1], ar) >= 0 else -1.0        # fixed global sign
        per = []
        for l in range(L):
            c, rt = nf.pca(V[:, l, :], 5)
            per.append({"layer": l,
                        "arousal_pc2_signed": round(corr(c[idx, 1] * fs, ar), 4),
                        "R_pc2pc3": round(best2d(c[idx, 1], c[idx, 2], ar), 4),
                        "pc2_var_share": round(float(rt[1]), 4)})
        a = next(p for p in per if p["layer"] == La)
        out["cells"][cell] = {"analysis_layer": La, "n_rm_overlap": len(idx),
            "arousal_pc2_at_L": a["arousal_pc2_signed"], "R_pc2pc3_at_L": a["R_pc2pc3"],
            "pc2pc3_gain_at_L": round(a["R_pc2pc3"] - abs(a["arousal_pc2_signed"]), 4),
            "pc2_var_share_range": [round(min(p["pc2_var_share"] for p in per), 4),
                                    round(max(p["pc2_var_share"] for p in per), 4)],
            "per_layer": per}
        print(f"{cell}: L{La} arousal-PC2 {a['arousal_pc2_signed']:+.3f}  R(PC2,PC3) {a['R_pc2pc3']:.3f}  "
              f"gain {out['cells'][cell]['pc2pc3_gain_at_L']:+.3f}  pc2var {a['pc2_var_share']:.3f}")
    OUT.write_text(json.dumps(out, indent=1)); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
