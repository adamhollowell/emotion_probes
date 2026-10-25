#!/usr/bin/env python3
"""s12 grading over geometry_out: patches cell vector paths to canonical_100T
and rewrites the module's OUT to geometry_out before executing the
validated grading source (no logic changes)."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import phase4_stageA as A
for c, spec in A.CELLS.items():
    spec["vec"] = (spec["vec"].replace("canonical.pt", "canonical.pt")
                   if c != "gemma-2-27b-base" else "data/vectors/emotion_vectors_171.pt")
src = (REPO / "gemma_grid/scripts/phase4_c3_grade.py").read_text()
src = src.replace('OUT = REPO / "gemma_grid" / "geometry_out"',
                  'OUT = REPO / "gemma_grid" / "geometry_out"')
assert 'geometry_out' in src
exec(compile(src, "phase4_c3_grade(exec)", "exec"), {"__name__": "__main__", "__file__": str(REPO / "gemma_grid/scripts/phase4_c3_grade.py")})
