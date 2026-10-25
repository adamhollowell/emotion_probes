#!/usr/bin/env python3
"""Z2 — All-layer valence/arousal depth curves (Zhang Table-3 analog).

Per cell, per layer: PCA on the cleaned vectors, noise_floor.py sign conventions,
signed PC1↔valence and PC2↔arousal correlations on R&M-45 (primary) and NRC-161
natural overlap (secondary).

Gate Z2-G1: analysis-layer R&M values must equal the stage-A table (≤5e-4).
Summary definitions (pre-stated): emergence = first layer with |r| ≥ threshold
(valence .5, arousal .35) sustained ≥2 consecutive layers; peak = argmax |r|;
final_delta = r(last) − r(peak).
"""
import csv, json
import numpy as np
from zhang_common import (REPO, OUT, CELLS, Manifest, load_vectors, pca_coords,
                          sign_orient, signed_corrs, load_rm, load_nrc)


def summarize(curve, thresh):
    curve = np.asarray(curve)
    emergence = None
    for i in range(len(curve) - 1):
        if abs(curve[i]) >= thresh and abs(curve[i + 1]) >= thresh:
            emergence = i
            break
    peak = int(np.argmax(np.abs(curve)))
    return {"emergence_layer": emergence, "peak_layer": peak,
            "peak_r": round(float(curve[peak]), 4),
            "final_r": round(float(curve[-1]), 4),
            "final_delta_from_peak": round(float(curve[-1] - curve[peak]), 4)}


def main():
    man = Manifest()
    rm, nrc = load_rm(man), load_nrc(man)
    gp = REPO / "gemma_grid/geometry_out/stageA_cross_cell_table.csv"
    man.sha(gp)
    locked = {r["cell"]: (float(r["valence_r"]), float(r["arousal_r"]))
              for r in csv.DictReader(open(gp))}
    out = {"summary_defs": "emergence: first layer |r|>=thresh (val .5 / aro .35) "
                           "sustained 2 layers; peak: argmax |r|; final_delta: r(last)-r(peak)"}
    for cell, cfg in CELLS.items():
        em, V = load_vectors(cell, man, "vectors")     # cleaned, (171, L, d)
        nL = V.shape[1]
        curves = {"rm_val": [], "rm_aro": [], "nrc_val": [], "nrc_aro": []}
        for l in range(nL):
            coords = sign_orient(pca_coords(V[:, l, :]), em)
            rv, ra, _ = signed_corrs(coords, em, rm)
            nv, na, _ = signed_corrs(coords, em, nrc)
            curves["rm_val"].append(round(rv, 4)); curves["rm_aro"].append(round(ra, 4))
            curves["nrc_val"].append(round(nv, 4)); curves["nrc_aro"].append(round(na, 4))
        L = cfg["layer"]
        dv = abs(curves["rm_val"][L] - locked[cell][0])
        da = abs(curves["rm_aro"][L] - locked[cell][1])
        gate = {"analysis_layer": L, "delta_val": round(dv, 6), "delta_aro": round(da, 6),
                "pass": dv <= 5e-4 and da <= 5e-4}
        print(f"[{cell}] Z2-G1 Δval={dv:.1e} Δaro={da:.1e} -> {'PASS' if gate['pass'] else 'FAIL'}"
              f" | val peak L{int(np.argmax(np.abs(curves['rm_val'])))} "
              f"aro peak L{int(np.argmax(np.abs(curves['rm_aro'])))} of {nL-1}")
        out[cell] = {"gate": gate, "curves": curves,
                     "summary": {"rm_val": summarize(curves["rm_val"], 0.5),
                                 "rm_aro": summarize(curves["rm_aro"], 0.35)},
                     "n_layers": nL}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zhang_z2_results.json").write_text(json.dumps(out, indent=1))
    man.write("zhang_z2_manifest.json")
    print("Z2 done; results + manifest written.")


if __name__ == "__main__":
    main()
