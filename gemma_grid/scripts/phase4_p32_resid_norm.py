#!/usr/bin/env python3
"""GGE1 family fold-in of E1 P3.2: residual-norm growth across layers, all six
cells. RAW activations -> BASIS-EXEMPT (E1: residual-stream norms do not depend on
the cleanup basis; RAW is the canonicality exception). Torch-free (ptload).

CAVEAT (stated in the output): E1 used per-TOKEN residual norms (915,288 tok/layer;
those raw files are pod-lost). GGE1's available raw activations are the 100T
neutral-dialogue PER-DIALOGUE MEANS (500 x L x d). This is therefore the
per-dialogue-mean residual-norm trajectory over the 100T neutral corpus — a
family-comparable, basis-exempt proxy for the growth curve + final-layer
transition, NOT a byte-match to E1's per-token metric. A faithful per-token
version would need re-extraction with norm logging (Tier-2).

Output: geometry_out/p32_resid_norm_growth_family.json.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import ptload
CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base",
         "gemma-2-9b-it", "gemma-2-27b-base", "gemma-2-27b-it"]
OUT = REPO / "gemma_grid" / "geometry_out" / "p32_resid_norm_growth_family.json"


def neut(cell):
    return ("gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__neutral.pt" if cell == "gemma-2-27b-base"
            else f"gemma_grid/data/{cell}/vectors/{cell}__neutral.pt")


def main():
    out = {"record": "P3.2 residual-norm growth — GGE1 family fold-in (E1 P3.2)",
           "basis": "RAW activations — BASIS-EXEMPT (no cleanup basis involved)",
           "aggregation": "per-dialogue-mean residual norm over the 100T neutral corpus (500 dialogues); "
                          "PROXY for E1's per-token metric (pod-lost) — growth curve + final-layer transition only",
           "cells": {}}
    for cell in CELLS:
        d = ptload.load(str(REPO / neut(cell)))
        acts = np.asarray(d["acts"], dtype=np.float64)          # (n_dialogues, L, dim)
        nrm = np.linalg.norm(acts, axis=2).mean(0)              # per-layer mean dialogue-norm
        diffs = np.diff(nrm)
        drop_t = int(diffs.argmin())
        out["cells"][cell] = {"n_dialogues": int(acts.shape[0]), "n_layers": int(nrm.shape[0]),
            "L0": round(float(nrm[0]), 2), "Lfinal": round(float(nrm[-1]), 2),
            "growth_ratio_final_over_L0": round(float(nrm[-1] / nrm[0]), 2),
            "largest_drop_transition": drop_t, "largest_drop": round(float(diffs[drop_t]), 2),
            "mean_resid_norm_by_layer": [round(float(x), 2) for x in nrm]}
        print(f"{cell}: L0 {nrm[0]:.1f} -> Lfinal {nrm[-1]:.1f} ({out['cells'][cell]['growth_ratio_final_over_L0']}x); "
              f"largest drop at T{drop_t} ({diffs[drop_t]:+.1f})")
    OUT.write_text(json.dumps(out, indent=1)); print("wrote", OUT.name)


if __name__ == "__main__":
    main()
