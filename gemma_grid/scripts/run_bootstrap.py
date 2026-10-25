#!/usr/bin/env python3
"""Persist the full six-cell bootstrap (audit follow-through): phase-1
resampled means are basis-independent and cached in scratch; this
re-runs PHASE 2 ONLY with each cell's stored cleanup basis and writes
geometry_out/<cell>/bootstrap_B1000.npz. Same method the auditor used;
this makes the artifacts durable. Resumable; --cell required; pass --scratch
EXPLICITLY (lesson: argparse defaults resume stale state)."""
import json, sys, shutil
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import phase4_stageC_bootstrap as BS
import phase4_stageA as A

CELL = sys.argv[1]
A.CELLS[CELL] = dict(A.CELLS[CELL], vec=(A.CELLS[CELL]["vec"].replace("canonical.pt", "canonical.pt")
                     if CELL != "gemma-2-27b-base" else "data/vectors/emotion_vectors_171.pt"))
BS.OUT = REPO / "gemma_grid" / "geometry_out"
import os as _os; SCR = Path(_os.environ["BOOT_SCRATCH"]) / "bootscratch_100T"
sdir = SCR / CELL; sdir.mkdir(parents=True, exist_ok=True)
# 100T: phase-1 means recomputed locally from raw per-story slices (basis-independent,
# deterministic seed); no reuse of older scratch. Per-cell 100T reproduction gate runs (<=1e-4).
_stored = torch.load(REPO / f"gemma_grid/data/{CELL}/vectors/{CELL}__cleanup_basis.pt",
                 map_location="cpu", weights_only=False)
_basis = np.asarray(_stored["bases_per_layer"][_stored["analysis_layer"]], dtype=np.float64)
def get_stored_basis(cell, slc, li):
    return _basis, {"source": "cleanup_basis.pt", "k": int(_basis.shape[0])}
BS.get_basis = get_stored_basis
sys.argv = [sys.argv[0], "--cell", CELL, "--scratch", str(SCR), "--budget", "1000000000"]
BS.main()
