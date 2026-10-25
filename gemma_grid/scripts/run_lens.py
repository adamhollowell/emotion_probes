#!/usr/bin/env python3
"""C5 lens package over the cleaned vectors — wrapper. Patches: cell vector
paths -> canonical_100T / emotion_vectors_171; OUT -> geometry_out; SCRATCH -> a
FRESH lens scratch folder (explicitly, per the resumed-stale-state failure mode
caught in the bootstrap spot-check). Anchor W_U f16 memmap + vocab reused
(read-only, basis-independent)."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import phase4_stageA as A
for c, spec in A.CELLS.items():
    spec["vec"] = (spec["vec"].replace("canonical.pt", "canonical.pt")
                   if c != "gemma-2-27b-base" else "data/vectors/emotion_vectors_171.pt")
import phase4_lens as L
L.OUT = REPO / "gemma_grid" / "geometry_out"
import os as _os; L.SCRATCH = Path(_os.environ["BOOT_SCRATCH"]) / "lens_scratch_100T"
L.SCRATCH.mkdir(parents=True, exist_ok=True)
# reuse the converted anchor W_U memmap (read-only)
import os as _os; L.ANCHOR_F16 = Path(_os.environ["BOOT_SCRATCH"]) / "anchor_WU_f16.npy"
if __name__ == "__main__":
    L.main()
