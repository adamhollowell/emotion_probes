#!/usr/bin/env python3
"""
Parametric search-artifact null for the two-block tuning-shift structure (paper section 7.3), run 2.

This is the computation the paper's section 7.3 numbers come from (the null maximum of .051 and the
per-size ratios). It was first run on 2026-10-09 as an independent second computation whose code
lived inline in COORDINATOR_SEARCH_ARTIFACT_NULL_SECOND_COMPUTATION_100T.md. This file commits that
code unchanged in its computational core so the stored artifact has a producer of record.

Run 1 (parametric_search_artifact_null.py, seed 20261009) is superseded on method: it searched
splits over range(1, L) and advanced one generator across the three sizes. Run 2 searches
range(2, L-2), matching block_split in fixed_probe_tuning_shifts.py verbatim, and reseeds
independently per size.

Per draw: X ~ N(0, 1) of shape (L, 171), C = corrcoef(X), record the best block-split gap.
No vector or activation file is read. Layer counts and observed gaps are fixed inputs.

Usage:
  python parametric_search_artifact_null_v2.py            # recompute and verify against the stored artifact
  python parametric_search_artifact_null_v2.py --write    # recompute and overwrite the stored artifact
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 20261010
B = 2000
N_EMOTIONS = 171
LAYER_COUNTS = {2: 26, 9: 42, 27: 46}
OBSERVED_GAPS = {2: 0.6195, 9: 0.3863, 27: 0.6053}

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "gemma_grid" / "geometry_out" / "parametric_search_artifact_null_SECOND.json"


def block_split_gap(C: np.ndarray) -> float:
    """block_split from fixed_probe_tuning_shifts.py, returning the unrounded best gap."""
    L = C.shape[0]
    best = None
    for s in range(2, L - 2):
        w = np.concatenate([C[:s, :s][np.triu_indices(s, 1)],
                            C[s:, s:][np.triu_indices(L - s, 1)]])
        x = C[:s, s:].ravel()
        gap = w.mean() - x.mean()
        if best is None or gap > best[1]:
            best = (s, gap)
    return float(best[1])


def run_null_for_size(n_layers: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaps = np.empty(B)
    for b in range(B):
        X = rng.standard_normal((n_layers, N_EMOTIONS))
        C = np.corrcoef(X)
        gaps[b] = block_split_gap(C)
    return gaps


def compute() -> dict:
    sizes = {}
    for size in [2, 9, 27]:
        n_layers = LAYER_COUNTS[size]
        obs = OBSERVED_GAPS[size]
        gaps = run_null_for_size(n_layers, B, SEED)
        null_max = float(gaps.max())
        sizes[f"{size}B"] = {
            "n_layers": n_layers,
            "observed_gap": obs,
            "null_mean": float(gaps.mean()),
            "null_std": float(gaps.std(ddof=1)),
            "null_p95": float(np.percentile(gaps, 95)),
            "null_max": null_max,
            "count_at_or_above_observed": int((gaps >= obs).sum()),
            "ratio_obs_to_null_max": round(obs / null_max, 2),
        }
    across_max = max(v["null_max"] for v in sizes.values())
    return {
        "record": "Coordinator second computation — parametric search-artifact null §7.3",
        "version": "v2 (corrected: range(2,L-2) split search; per-size independent seeding)",
        "seed": SEED,
        "B": B,
        "n_emotions": N_EMOTIONS,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "v1_deviations_corrected": [
            "split range was range(1,L); corrected to range(2,L-2) matching block_split verbatim",
            "v1 advanced a single rng across sizes; corrected to reseed from SEED per size",
        ],
        "sizes": sizes,
        "summary": {
            "across_size_max_of_null_max": across_max,
            "check_max_below_047": across_max < 0.047,
            "check_all_ratios_in_12_to_21": all(12 <= v["ratio_obs_to_null_max"] <= 21 for v in sizes.values()),
            "check_zero_draws_reached_observed_gap": all(v["count_at_or_above_observed"] == 0 for v in sizes.values()),
            "total_count_at_or_above": sum(v["count_at_or_above_observed"] for v in sizes.values()),
        },
    }


def verify(new: dict, stored: dict) -> bool:
    ok = True
    for size, vals in stored["sizes"].items():
        for k, v in vals.items():
            n = new["sizes"][size][k]
            same = (n == v) if not isinstance(v, float) else math.isclose(n, v, rel_tol=0, abs_tol=0)
            if not same:
                ok = False
                print(f"MISMATCH {size}.{k}: stored {v!r} recomputed {n!r}")
    for k, v in stored["summary"].items():
        if new["summary"][k] != v:
            ok = False
            print(f"MISMATCH summary.{k}: stored {v!r} recomputed {new['summary'][k]!r}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    new = compute()
    if a.write:
        OUT.write_text(json.dumps(new, indent=1))
        print("wrote", OUT)
    else:
        stored = json.loads(OUT.read_text())
        print("numpy", np.__version__)
        print("VERIFIED: every stored value reproduced exactly" if verify(new, stored) else "VERIFICATION FAILED")
        sys.exit(0 if verify(new, stored) else 1)
