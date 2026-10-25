#!/usr/bin/env python3
"""Stage C grading over geometry_out. Wrapper around the validated
phase4_stageC_grade module: OUT -> geometry_out. Every model's bootstrap draws
are read from geometry_out/<model>/bootstrap_B1000.npz, and the script stops
if one is missing."""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import phase4_stageC_grade as G

GEO = REPO / "gemma_grid" / "geometry_out"
G.OUT = GEO

_orig_load = G.load
def load_stored(cell):
    S = json.loads((GEO / cell / "stageA_summary.json").read_text())
    npz_p = GEO / cell / f"bootstrap_B{G.B}.npz"
    if not npz_p.exists():
        sys.exit(f"100T bootstrap missing for {cell}: {npz_p} — refusing a fallback to an older tree (100T canonicality)")
    with np.load(npz_p, allow_pickle=True) as bz:
        b = {k: bz[k] for k in ("tri", "coords", "ratios", "pair_cos", "rs", "pairs", "emotions")}
    return S, b
G.load = load_stored

if __name__ == "__main__":
    G.main()
