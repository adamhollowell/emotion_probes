#!/usr/bin/env python3
"""Registered layer-stability tier-2 test (PREREGISTRATION (e): "reuse the
§6.4 bootstrap-CI code, cross-cell") — closes the gap flagged by Phase 5.

§6.4's CI (scripts/analyze_depth_arousal.py, Q2) is a CASE bootstrap:
resample the emotions entering the statistic, percentile 95% CI, 2,000 draws.
Cross-cell adaptation (logged conventions):
  * cases = the 171 emotions, resampled WITH replacement; the SAME resampled
    emotion set is applied to every cell within a replicate (paired design —
    emotion-sampling covariance cancels in Δ);
  * statistic per cell = layerstab_mid_late exactly as Stage A / noise_floor:
    mean off-diagonal Pearson corr, over the 14-layer sample, among layers
    ≥ L/2, of the per-layer pairwise-cosine upper-triangle entries — here
    evaluated on the RESAMPLED pair set (pairs of distinct resampled draws;
    self-duplicate pairs excluded, as cos=1 identically in every layer);
  * contrasts graded: tuning pooled [IT−base] (mean of the three size-pair
    Δs per replicate) and the two scale chain endpoints (2b→27b);
  * verdict = REAL iff 95% CI excludes 0 AND |Δ_obs| ≥ locked floor 0.01.
RNG seed 0 (as §6.4). B=2000 (as §6.4). Resumable via --budget seconds.

Output: gemma_grid/geometry_out/layerstab_ci.json
Usage: repeat `python gemma_grid/scripts/phase4_layerstab_ci.py` until DONE.
"""
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf
from phase4_stageA import CELLS
for _c, _s in CELLS.items():
    _s["vec"] = (_s["vec"].replace("canonical.pt", "canonical.pt") if _c != "gemma-2-27b-base" else "data/vectors/emotion_vectors_171.pt")

OUT = REPO / "gemma_grid" / "geometry_out"
import os as _os; SCRATCH = Path(_os.environ["BOOT_SCRATCH"]) / "lsci"
B = 2000
BUDGET = 1e12
TUNING = [("gemma-2-2b-base", "gemma-2-2b-it"), ("gemma-2-9b-base", "gemma-2-9b-it"),
          ("gemma-2-27b-base", "gemma-2-27b-it")]
CHAINS = {"base": ("gemma-2-2b-base", "gemma-2-27b-base"), "it": ("gemma-2-2b-it", "gemma-2-27b-it")}


def cell_cosmats(cell):
    """Per-sampled-layer full cosine matrices + the mid-late layer mask,
    conventions identical to noise_floor.py / geometry_full.py."""
    em, V = nf.load_vectors(REPO / CELLS[cell]["vec"])
    n, L, d = V.shape
    samp = sorted(set(np.linspace(0, L - 1, min(14, L)).round().astype(int).tolist()))
    mats = np.stack([nf.cos(V[:, l, :]) for l in samp])   # (S, n, n)
    ml = np.array([k for k, l in enumerate(samp) if l >= L // 2])
    return em, mats, ml


def mid_late(mats, ml, ii, jj):
    tris = mats[:, ii, jj]                                # (S, n_pairs)
    M = np.corrcoef(tris)
    sub = M[np.ix_(ml, ml)]
    return float((sub.sum() - len(ml)) / (len(ml) ** 2 - len(ml)))


def main():
    t0 = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    state_p = SCRATCH / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"b": 0}
    deltas_p = SCRATCH / "deltas.npy"

    data = {c: cell_cosmats(c) for c in CELLS}
    n = 171
    contrast_names = ["tuning_pooled"] + [f"scale_2b27b_{ch}" for ch in CHAINS]
    if deltas_p.exists():
        deltas = np.load(deltas_p)
    else:
        deltas = np.full((B, len(contrast_names)), np.nan, np.float64)

    rng = np.random.default_rng(0)
    # fast-forward RNG deterministically to the resume point
    for _ in range(state["b"]):
        rng.integers(0, n, n)

    while state["b"] < B and time.time() - t0 < BUDGET:
        b = state["b"]
        idx = rng.integers(0, n, n)
        pi, pj = np.triu_indices(n, 1)
        ii, jj = idx[pi], idx[pj]
        keep = ii != jj                                    # drop self-duplicate pairs
        ii, jj = ii[keep], jj[keep]
        vals = {c: mid_late(data[c][1], data[c][2], ii, jj) for c in CELLS}
        deltas[b, 0] = np.mean([vals[b_] - vals[a_] for a_, b_ in TUNING])
        for k, (ch, (a_, b_)) in enumerate(CHAINS.items()):
            deltas[b, 1 + k] = vals[b_] - vals[a_]
        state["b"] = b + 1
        if (b + 1) % 100 == 0:
            np.save(deltas_p, deltas); state_p.write_text(json.dumps(state))
    np.save(deltas_p, deltas); state_p.write_text(json.dumps(state))
    if state["b"] < B:
        print(f"PROGRESS {state['b']}/{B} ({time.time()-t0:.0f}s)"); return

    # observed values (full sample) + verdicts
    pi, pj = np.triu_indices(n, 1)
    obs_cell = {c: mid_late(data[c][1], data[c][2], pi, pj) for c in CELLS}
    obs = {"tuning_pooled": float(np.mean([obs_cell[b_] - obs_cell[a_] for a_, b_ in TUNING]))}
    for ch, (a_, b_) in CHAINS.items():
        obs[f"scale_2b27b_{ch}"] = obs_cell[b_] - obs_cell[a_]
    out = {"B": B, "seed": 0, "method": "§6.4 case bootstrap over emotions, paired across cells, "
           "self-duplicate pairs excluded; verdict = CI excludes 0 AND |Δ|>=0.01 (locked floor)",
           "per_cell_mid_late_obs": {c: round(v, 5) for c, v in obs_cell.items()}, "contrasts": {}}
    for k, name in enumerate(contrast_names):
        lo, hi = np.percentile(deltas[:, k], [2.5, 97.5])
        d_ = obs[name]
        real = (lo > 0 or hi < 0) and abs(d_) >= 0.01
        out["contrasts"][name] = {"delta_obs": round(float(d_), 5),
                                  "ci95": [round(float(lo), 5), round(float(hi), 5)],
                                  "floor": 0.01,
                                  "verdict": ("REAL" if real else
                                              "NULL_subfloor" if abs(d_) < 0.01 else "NULL_CI_includes_0")}
        print(f"{name}: Δ={d_:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  -> {out['contrasts'][name]['verdict']}")
    (OUT / "layerstab_ci.json").write_text(json.dumps(out, indent=2))
    print("DONE -> layerstab_ci.json")


if __name__ == "__main__":
    main()
