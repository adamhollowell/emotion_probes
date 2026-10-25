#!/usr/bin/env python3
"""
Layer-permutation null for the two-block tuning-shift structure (paper section 7.3).

Producer of record for geometry_out/job2_layer_permutation_null.json, re-derived on
2026-10-14 from the procedure that artifact serializes (dispatch
DISPATCH_LEDGER_COMPUTE_L18_AND_PERMUTATION_100T.md, run 2026-10-07). The original run retained no
script. This file reproduces every stored value exactly, and the verify mode checks that.

Input: the per-size layer_corr_matrix from fixed_probe_tuning_shifts_results.json, values as
stored (rounded to 3 decimals). The input hash is asserted against the artifact's manifest.

Procedure, per size, with the generator reseeded from SEED for each size:
  block_split (fixed_probe_tuning_shifts.py, verbatim): best contiguous split s with the
  search iterating range(2, L - 2) in Python, so s takes the values 2 through L minus 3 inclusive,
  maximizing within-block minus cross-block mean off-diagonal correlation. (The artifact's procedure
  field writes this as the closed interval [2,L-2], which mis-transcribes the range. The Phase 5 audit
  of 2026-10-14 refuted the literal reading and sustained the code. The artifact is left untouched as the
  only unregenerated original of the three audited files, and the correction travels beside it.)
  Null: relabel layers by a random permutation pi, form C[pi][:, pi], rerun block_split, record the
  best gap. B = 2000 permutations. count = permutations with gap >= observed.
  p = (1 + count) / (B + 1), the add-one form applied uniformly to all three sizes.

Label carried by the artifact: descriptive support. The split search guarantees some gap, and this
null quantifies the search artifact. It grades nothing.

Usage:
  python job2_layer_permutation_null.py            # recompute and verify against the stored artifact
  python job2_layer_permutation_null.py --write    # recompute and overwrite the stored artifact
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 20261007
B = 2000
SIZES = ["2B", "9B", "27B"]

REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "gemma_grid" / "geometry_out" / "fixed_probe_tuning_shifts_results.json"
OUT = REPO / "gemma_grid" / "geometry_out" / "job2_layer_permutation_null.json"


def block_split(C: np.ndarray):
    L = C.shape[0]
    best = None
    for s in range(2, L - 2):
        w = np.concatenate([C[:s, :s][np.triu_indices(s, 1)], C[s:, s:][np.triu_indices(L - s, 1)]])
        x = C[:s, s:].ravel()
        gap = w.mean() - x.mean()
        if best is None or gap > best[1]:
            best = (s, gap, float(w.mean()), float(x.mean()))
    return best


def compute(stored_meta: dict) -> dict:
    data = json.loads(INPUT.read_text())
    results = {}
    for size in SIZES:
        rng = np.random.default_rng(SEED)
        C = np.array(data[size]["layer_corr_matrix"], dtype=float)
        L = C.shape[0]
        s, g, w, x = block_split(C)
        gaps = np.empty(B)
        for b in range(B):
            pi = rng.permutation(L)
            gaps[b] = block_split(C[pi][:, pi])[1]
        count = int((gaps >= g).sum())
        results[size] = {
            "split_layer": s,
            "within_mean": w,
            "cross_mean": x,
            "gap_observed": g,
            "perm_count_ge_observed": count,
            "B": B,
            "p_add_one": (1 + count) / (B + 1),
            "null_max": float(gaps.max()),
            "null_mean": float(gaps.mean()),
            "null_median": float(np.median(gaps)),
            "null_p95": float(np.percentile(gaps, 95)),
        }
    out = dict(stored_meta)
    out["results"] = results
    out["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["sr5_input_manifest"] = {INPUT.name: hashlib.sha256(INPUT.read_bytes()).hexdigest()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    stored = json.loads(OUT.read_text())
    in_hash = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    exp = {k.replace("_100T", ""): v for k, v in stored["sr5_input_manifest"].items()}[INPUT.name]  # stored keys carry the as-run file names
    if in_hash != exp:
        sys.exit(f"input hash mismatch: {INPUT.name} is {in_hash[:12]}, artifact manifest expects {exp[:12]}")
    meta = {k: v for k, v in stored.items() if k not in ("results", "generated_utc", "sr5_input_manifest")}
    new = compute(meta)
    if a.write:
        OUT.write_text(json.dumps(new, indent=2))
        print("wrote", OUT)
        sys.exit(0)
    ok = True
    for size in SIZES:
        for k, v in stored["results"][size].items():
            if new["results"][size][k] != v:
                ok = False
                print(f"MISMATCH {size}.{k}: stored {v!r} recomputed {new['results'][size][k]!r}")
    print("numpy", np.__version__)
    print("VERIFIED: every stored value reproduced exactly" if ok else "VERIFICATION FAILED")
    sys.exit(0 if ok else 1)
