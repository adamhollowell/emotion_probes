#!/usr/bin/env python3
"""Stage B (C4 contrasts) over the cleaned vectors — wrapper: patches the
validated phase4_stageB module's inputs/outputs (no logic changes).
Reads canonical_100T per cell + geometry_out stageA summaries; writes
stageB_contrasts.json + displacement CSVs into geometry_out/."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import phase4_stageA as A
import phase4_stageB as B

for c, spec in A.CELLS.items():
    spec["vec"] = (spec["vec"].replace("canonical.pt", "canonical.pt")
                   if c != "gemma-2-27b-base" else "data/vectors/emotion_vectors_171.pt")
B.OUT = REPO / "gemma_grid" / "geometry_out"

if __name__ == "__main__":
    B.main()
