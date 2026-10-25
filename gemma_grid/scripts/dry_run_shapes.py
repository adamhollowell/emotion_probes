#!/usr/bin/env python3
"""
Phase-2 shape-correct dry-run (NO GPU, NO model).

For each new cell's exact (n_layers, d_model) and analysis layer, synthesize a
(171, n_layers, d_model) vector tensor and run the full per-cell scalar-signal
analysis (noise_floor.compute — the C1–C4 signals) to confirm the pipeline is
dimension-agnostic: it must execute and return finite numbers at 2B/9B/27B dims
with the analysis layer resolved per cell. This is the cheap check the plan puts
BEFORE any pilot extraction, so a shape bug is caught with zero GPU spend.

Run (analysis env): lens_env/bin/python gemma_grid/scripts/dry_run_shapes.py
"""
from __future__ import annotations
import sys, math, tempfile
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from emotion_probes.emotions import EMOTIONS_171
from gemma_grid.scripts import noise_floor as nf

# (label, n_layers, d_model) — analysis layer is derived, not trusted.
SHAPES = [("gemma-2-2b {base,it}", 26, 2304),
          ("gemma-2-9b {base,it}", 42, 3584),
          ("gemma-2-27b-it", 46, 4608)]
RATINGS = REPO / "data/ratings/russell_mehrabian_1977.csv"
LOCKED = REPO / "gemma_grid/data/geometry_quick_summary.json"   # release: the staged copy (7aff4a70), as in phase4_stageA.py


def main() -> int:
    torch.manual_seed(0)
    all_ok = True
    tmp = Path(tempfile.mkdtemp())
    for label, n_layers, d_model in SHAPES:
        layer = round(2 / 3 * n_layers)
        synth = {"emotions": list(EMOTIONS_171),
                 "vectors": torch.randn(171, n_layers, d_model)}
        p = tmp / f"synth_{n_layers}.pt"; torch.save(synth, p)
        try:
            res = nf.compute(p, RATINGS, LOCKED, layer, kmeans_seeds=list(range(5)))
        except Exception as e:
            print(f"[FAIL] {label}: {type(e).__name__}: {e}")
            all_ok = False; continue
        det = res["deterministic"]
        shape_ok = (res["n"] == 171 and res["L"] == n_layers and res["d"] == d_model
                    and res["layer"] == layer)
        finite_ok = all(math.isfinite(v) for v in det.values())
        n_sig = len(det)
        ok = shape_ok and finite_ok
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label:22s} n_layers={n_layers} d_model={d_model} "
              f"-> analysis L{layer} | {n_sig} scalar signals finite={finite_ok} shape={shape_ok}")
        if ok:
            print(f"         var pc1/2/3={det['var_pc1']:.2f}/{det['var_pc2']:.2f}/{det['var_pc3']:.2f} "
                  f"(random data, sanity only) | kmeans self-ARI mean={res['kmeans']['self_ari_mean']:.3f}")
    print("\n" + ("ALL SHAPES PASS — per-cell analysis is dimension-agnostic"
                  if all_ok else "SOME SHAPES FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
