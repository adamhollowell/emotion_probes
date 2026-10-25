#!/usr/bin/env python3
"""
Lightweight post-extraction validation for one mode-A cell (used by the Phase-3
orchestrator to decide whether to proceed to the next cell). NO GPU.

Checks:
  1. canonical.pt exists, shape (171, n_layers, d_model), all finite.
  2. per-story slice + lens ingredients present.
  3. REAL geometry, not a random/broken tensor: PC1 variance at the derived
     analysis layer must clear a floor well above the random baseline (~1% for
     171 points). Threshold 5% — the anchor and 2B both sit ~26-30%.

Exit 0 = pass; non-zero = fail (orchestrator aborts the chain).
"""
import sys, os, math
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
PC1_FLOOR = 5.0  # percent; random 171-point data gives ~1%

def main() -> int:
    cell = sys.argv[1]
    base = REPO / "gemma_grid" / "data" / cell / "vectors"
    canon = base / "canonical.pt"
    if not canon.exists():
        print(f"FAIL {cell}: missing {canon}"); return 1
    v = torch.load(canon, map_location="cpu", weights_only=False)
    V = v["vectors"]
    n, L, d = V.shape
    if n != 171:
        print(f"FAIL {cell}: n_emotions={n} != 171"); return 1
    if not torch.isfinite(V).all():
        print(f"FAIL {cell}: non-finite values in vectors"); return 1
    for extra in (f"{cell}__per_story_slice.pt", f"{cell}__lens_ingredients.pt"):
        if not (base / extra).exists():
            print(f"FAIL {cell}: missing {extra}"); return 1
    layer = round(2 / 3 * L)
    Vl = V[:, layer, :].float().numpy()
    Vc = Vl - Vl.mean(0, keepdims=True)
    s = np.linalg.svd(Vc, compute_uv=False)
    pc1 = 100.0 * s[0] ** 2 / (s ** 2).sum()
    print(f"{cell}: shape ({n},{L},{d}) analysis L{layer} PC1var={pc1:.1f}%")
    if not math.isfinite(pc1) or pc1 < PC1_FLOOR:
        print(f"FAIL {cell}: PC1 variance {pc1:.1f}% below floor {PC1_FLOOR}% — geometry looks random/broken")
        return 1
    print(f"VALIDATED {cell}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
