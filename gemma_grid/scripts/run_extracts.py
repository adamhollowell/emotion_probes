#!/usr/bin/env python3
"""Extracts (fig84 + seams + boundary + consolidation + tables) over the
tree — wrapper: patches inputs to canonical_100T / emotion_vectors_171 and OUT to
geometry_out. No logic changes."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import phase4_stageA as A
for c, spec in A.CELLS.items():
    spec["vec"] = (spec["vec"].replace("canonical.pt", "canonical.pt")
                   if c != "gemma-2-27b-base" else "data/vectors/emotion_vectors_171.pt")
import phase4_extracts as X
X.OUT = REPO / "gemma_grid" / "geometry_out"
if __name__ == "__main__":
    X.main()
