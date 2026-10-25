#!/usr/bin/env python3
"""GGE1 family fold-in of E1 P3.7: dimensional velocity (adjacent-layer cosine of
each emotion's vector, rotation N->N+1), all six cells. CLEANED vectors -> 100T
basis (asserted). Adapts analyze_velocity.py. Texture buckets are cell-relative
(transitions), since E1's absolute 27B bands (L43->44 terminal, L21-23 seam) do
not transfer across model depths. RUN under lens_env. Output:
geometry_out/p37_velocity_family.json.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf
from phase4_stageA import CELLS
OUT = REPO / "gemma_grid" / "geometry_out" / "p37_velocity_family.json"


def vec100T(cell):
    v = CELLS[cell]["vec"]
    return ("data/vectors/emotion_vectors_171.pt" if cell == "gemma-2-27b-base"
            else v.replace("canonical.pt", "canonical.pt"))


def vel(Ve):
    nrm = np.clip(np.linalg.norm(Ve, axis=1, keepdims=True), 1e-8, None)
    Vn = Ve / nrm
    return np.array([float(Vn[i] @ Vn[i + 1]) for i in range(Ve.shape[0] - 1)])


def main():
    out = {"record": "P3.7 dimensional velocity — GGE1 family fold-in (E1 P3.7)",
           "basis": "100T cleaned vectors (asserted 100T-in-path); adjacent-layer cosine per emotion",
           "cells": {}}
    for cell in CELLS:
        vp = vec100T(cell)
        assert "_HA" not in vp and "_PA" not in vp, f"CANONICALITY: superseded-basis vec path for {cell}: {vp}"   # release: only 100T vectors ship
        em, V = nf.load_vectors(REPO / vp)
        n, L, d = V.shape
        curves = np.stack([vel(V[e]) for e in range(n)])        # (n, L-1)
        T = L - 1
        minL = curves.argmin(1)                                  # max-rotation transition per emotion
        maxL = curves.argmax(1)                                  # most-stable transition per emotion
        out["cells"][cell] = {"n_layers": L, "n_transitions": T,
            "median_maxrot_transition": int(np.median(minL)),
            "median_moststable_transition": int(np.median(maxL)),
            "moststable_median_cos": round(float(np.median(curves[np.arange(n), maxL])), 4),
            "count_maxrot_terminal": int(np.sum(minL >= T - 1)),
            "count_maxrot_last3": int(np.sum(minL >= T - 3)),
            "count_maxrot_seam": int(np.sum(np.abs(minL - (L // 2)) <= 1))}
        print(f"{cell}: median max-rot @T{int(np.median(minL))}/{T}  last3 {out['cells'][cell]['count_maxrot_last3']}/{n}  "
              f"seam {out['cells'][cell]['count_maxrot_seam']}/{n}  most-stable @T{int(np.median(maxL))} (cos {out['cells'][cell]['moststable_median_cos']})")
    out["note"] = ("terminal/last3/seam bands are cell-relative transition counts; E1's 27B absolute "
                   "L43->44 terminal / L21-23 seam map to these. E2 firewall: do NOT interpret loving's "
                   "velocity minimum (output-space artifact per E1 P3.7).")
    OUT.write_text(json.dumps(out, indent=1)); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
