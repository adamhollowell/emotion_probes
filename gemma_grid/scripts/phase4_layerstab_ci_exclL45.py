#!/usr/bin/env python3
"""L45-EXCLUSION diagnostic for the registered layer-stability tier-2 test
(Memo-04 ruling), 100T basis. Faithful variant of phase4_layerstab_ci.py.

Purpose: the anchor (27b-base) mid-late RSA sits low (0.94924 vs ~0.99 elsewhere)
and the Phase-5 audit called the scale-base REAL verdict "anchor-L45-localized."
This diagnostic drops absolute layer 45 from the sampled mid-late window and
re-runs the identical §6.4 case-bootstrap, so the Ledger Chat can see whether the
scale-base (2b->27b_base) verdict survives excluding L45, or is L45-driven.

Diffs from the base producer, all marked [EXCL-L45] / [100T]:
  [100T]     cell vectors remapped to canonical_100T / emotion_vectors_171_100T.
  [EXCL-L45] cell_cosmats drops absolute layer index 45 from the sampled set
             (present only in the 46-layer 27B cells; 2B/9B unaffected).
  output     geometry_out/layerstab_ci_exclL45.json (+ exclusion metadata).
Everything else — statistic, paired resample (seed 0, B=2000), self-duplicate
exclusion, verdict rule (CI excludes 0 AND |Δ|>=0.01) — is byte-identical to the base.

RUN under lens_env (same interpreter as orchestrator step 12), one call:
  lens_env/bin/python gemma_grid/scripts/phase4_layerstab_ci_exclL45.py
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf
from phase4_stageA import CELLS

# [100T] remap cell vectors to the 100T basis (sibling wrappers' convention)
for _c, _s in CELLS.items():
    _s["vec"] = (_s["vec"].replace("canonical.pt", "canonical.pt")
                 if _c != "gemma-2-27b-base" else "data/vectors/emotion_vectors_171.pt")

OUT = REPO / "gemma_grid" / "geometry_out"
SCRATCH = Path(os.environ.get("BOOT_SCRATCH", str(REPO / "gemma_grid" / ".scratch"))) / "lsci_exclL45"
B = 2000
BUDGET = 1e12          # finish in one call (matches the 100T layerstab wrapper)
EXCL_LAYER = 45        # [EXCL-L45] absolute layer index to drop from the sampled window
TUNING = [("gemma-2-2b-base", "gemma-2-2b-it"), ("gemma-2-9b-base", "gemma-2-9b-it"),
          ("gemma-2-27b-base", "gemma-2-27b-it")]
CHAINS = {"base": ("gemma-2-2b-base", "gemma-2-27b-base"), "it": ("gemma-2-2b-it", "gemma-2-27b-it")}


def cell_cosmats(cell):
    em, V = nf.load_vectors(REPO / CELLS[cell]["vec"])
    n, L, d = V.shape
    samp = sorted(set(np.linspace(0, L - 1, min(14, L)).round().astype(int).tolist()))
    excluded = EXCL_LAYER in samp
    if excluded:
        samp = [l for l in samp if l != EXCL_LAYER]        # [EXCL-L45] drop layer 45
    mats = np.stack([nf.cos(V[:, l, :]) for l in samp])    # (S, n, n)
    ml = np.array([k for k, l in enumerate(samp) if l >= L // 2])
    return em, mats, ml, excluded


def mid_late(mats, ml, ii, jj):
    tris = mats[:, ii, jj]
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
    excl_map = {c: bool(data[c][3]) for c in CELLS}
    n = 171
    contrast_names = ["tuning_pooled"] + [f"scale_2b27b_{ch}" for ch in CHAINS]
    deltas = np.load(deltas_p) if deltas_p.exists() else np.full((B, len(contrast_names)), np.nan, np.float64)

    rng = np.random.default_rng(0)
    for _ in range(state["b"]):
        rng.integers(0, n, n)

    while state["b"] < B and time.time() - t0 < BUDGET:
        b = state["b"]
        idx = rng.integers(0, n, n)
        pi, pj = np.triu_indices(n, 1)
        ii, jj = idx[pi], idx[pj]
        keep = ii != jj
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

    pi, pj = np.triu_indices(n, 1)
    obs_cell = {c: mid_late(data[c][1], data[c][2], pi, pj) for c in CELLS}
    obs = {"tuning_pooled": float(np.mean([obs_cell[b_] - obs_cell[a_] for a_, b_ in TUNING]))}
    for ch, (a_, b_) in CHAINS.items():
        obs[f"scale_2b27b_{ch}"] = obs_cell[b_] - obs_cell[a_]
    out = {"B": B, "seed": 0,
           "diagnostic": "L45-EXCLUSION (Memo-04): absolute layer 45 dropped from the mid-late "
                         "sampled window; identical §6.4 case bootstrap otherwise. Compare to "
                         "layerstab_ci.json to read whether scale-base is L45-localized.",
           "excluded_layer": EXCL_LAYER, "cells_with_layer_excluded": [c for c in CELLS if excl_map[c]],
           "method": "§6.4 case bootstrap over emotions, paired across cells, self-duplicate pairs "
                     "excluded; verdict = CI excludes 0 AND |Δ|>=0.01 (locked floor)",
           "per_cell_mid_late_obs": {c: round(v, 5) for c, v in obs_cell.items()}, "contrasts": {}}
    for k, name in enumerate(contrast_names):
        lo, hi = np.percentile(deltas[:, k], [2.5, 97.5])
        d_ = obs[name]
        real = (lo > 0 or hi < 0) and abs(d_) >= 0.01
        out["contrasts"][name] = {"delta_obs": round(float(d_), 5),
                                  "ci95": [round(float(lo), 5), round(float(hi), 5)], "floor": 0.01,
                                  "verdict": ("REAL" if real else "NULL_subfloor" if abs(d_) < 0.01
                                              else "NULL_CI_includes_0")}
        print(f"[exclL45] {name}: delta={d_:+.5f}  CI[{lo:+.5f},{hi:+.5f}]  -> {out['contrasts'][name]['verdict']}")
    (OUT / "layerstab_ci_exclL45.json").write_text(json.dumps(out, indent=2))
    print("cells with L45 excluded:", [c for c in CELLS if excl_map[c]])
    print("DONE -> geometry_out/layerstab_ci_exclL45.json")


if __name__ == "__main__":
    main()
