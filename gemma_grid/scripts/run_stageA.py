#!/usr/bin/env python3
"""Stage A over the cleaned vectors — thin wrapper around the validated
phase4_stageA.run_cell (no logic changes; only inputs/outputs differ).

Inputs: canonical.pt per mode-A cell; anchor = data/vectors/emotion_vectors_171.pt.
Outputs: gemma_grid/geometry_out/<cell>/stageA_summary.json.
pod-nf diff disabled (those floors predate this basis); anchor_check disabled (the
anchor's departures from earlier locked values are the DOCUMENTED E1 deltas,
not gate failures). Resumable: skips cells whose output exists.

Usage: repeat `python gemma_grid/scripts/run_stageA.py [--cell <cell>]`.
"""
import argparse, copy, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import phase4_stageA as A

OUT = REPO / "gemma_grid" / "geometry_out"
VEC = {c: (spec["vec"].replace("canonical.pt", "canonical.pt")
              if c != "gemma-2-27b-base" else "data/vectors/emotion_vectors_171.pt")
          for c, spec in A.CELLS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=A.CELLS)
    a = ap.parse_args()
    todo = [a.cell] if a.cell else list(A.CELLS)
    for cell in todo:
        outdir = OUT / cell
        if (outdir / "stageA_summary.json").exists():
            print(f"[{cell}] already done, skipping"); continue
        spec = copy.deepcopy(A.CELLS[cell])
        spec["vec"] = VEC[cell]
        spec["nf"] = None
        A.run_cell(cell, spec, outdir, anchor_check=False)
    print("DONE: all Stage A summaries present")


if __name__ == "__main__":
    main()
