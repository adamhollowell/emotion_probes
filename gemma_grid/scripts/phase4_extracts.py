#!/usr/bin/env python3
"""Phase 4 — Fig-84 depth-resolved tuning curves + descriptive extracts.

All EXPLORATORY/characterization (no registered predictions; no hypothesis
tests). Items:
  (1) Fig-84 test: per-layer RSA(base-L, IT-L) per size — where in depth does
      tuning perturb the geometry? Direct parallel to Anthropic's Fig 84
      claim ("post-training modifies emotion representations primarily in the
      mid-to-late layers where emotion probe structure is most developed").
  (2) Seam extraction per cell from the existing Stage A C2 curves (largest
      adjacent-layer jump in RSA-to-analysis-layer before the analysis layer)
      + fractional depth.
  (3) Cross-cell boundary-emotion overlap (per-emotion 10-NN in-cluster
      purity, Stage A convention, recomputed for all 171 per cell).
  (4) Layerwise valence/arousal consolidation curves per cell (global PCA per
      layer -> R&M 45 overlap; E1 P3.4 pathway) + arousal-best-PC per layer.
  (5) Table-12-style k=10 membership tables per cell (markdown, GGE1 asset).

Output: geometry_out/fig84_depth_tuning.json, extracts_summary.json,
        table12_<cell>.md
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf
from phase4_stageA import CELLS

OUT = REPO / "gemma_grid" / "geometry_out"
SIZES = {"2b": ("gemma-2-2b-base", "gemma-2-2b-it"),
         "9b": ("gemma-2-9b-base", "gemma-2-9b-it"),
         "27b": ("gemma-2-27b-base", "gemma-2-27b-it")}
iu = np.triu_indices(171, 1)


def tri(Vl):
    return nf.cos(Vl)[iu]


def main():
    V = {}; em = None
    for c in CELLS:
        em_c, Vc = nf.load_vectors(REPO / CELLS[c]["vec"])
        em = em_c; V[c] = Vc

    # ---- (1) Fig-84 curves
    fig84 = {}
    for size, (b, it) in SIZES.items():
        L = V[b].shape[1]
        curve = [{"layer": l, "rsa": float(np.corrcoef(tri(V[b][:, l, :]), tri(V[it][:, l, :]))[0, 1])}
                 for l in range(L)]
        r = np.array([x["rsa"] for x in curve])
        thirds = np.array_split(np.arange(L), 3)
        band = {nm: float(1 - r[ix].mean()) for nm, ix in zip(("early", "mid", "late"), thirds)}
        fig84[size] = {"n_layers": L, "curve": curve,
                       "min_rsa_layer": int(np.argmin(r)), "min_rsa": float(r.min()),
                       "min_frac_depth": float(np.argmin(r) / L),
                       "perturbation_1minusRSA_by_third": band,
                       "analysis_layer": CELLS[b]["layer"]}
    (OUT / "fig84_depth_tuning.json").write_text(json.dumps(fig84, indent=2))
    for size, d in fig84.items():
        print(f"fig84 {size}: min RSA {d['min_rsa']:.4f} @L{d['min_rsa_layer']} (frac {d['min_frac_depth']:.2f}) | "
              f"perturb early/mid/late = " + "/".join(f"{d['perturbation_1minusRSA_by_third'][k]:.4f}" for k in ("early", "mid", "late")))

    X = {"fig84_headline": {s: {k: fig84[s][k] for k in ("min_rsa_layer", "min_frac_depth", "min_rsa",
                                                          "perturbation_1minusRSA_by_third")} for s in fig84}}

    # ---- (2) seam from existing C2 curves — CORRECTED METRIC (audit-verified)
    # First sustained crossing of RSA-to-analysis-layer >= 0.97 (3 consecutive
    # layers up to the analysis layer). The earlier largest-adjacent-jump
    # detector was fooled by early-layer wobble (it placed the anchor seam at
    # L10->L11, contradicting E1's L22-26) and is retired; correction logged in
    # the ledger (G6.4) and phase5_g6_addendum.md.
    seams = {"metric": "first layer where RSA-to-analysis-layer >= 0.97 (sustained 3 layers; E1 band-edge style). Replaces the earlier largest-adjacent-jump detector, which early wobble fooled (correction logged)."}
    for c in CELLS:
        SA = json.loads((OUT / c / "stageA_summary.json").read_text())
        layer_A = CELLS[c]["layer"]
        rsa = [r_["rsa_pearson"] for r_ in SA["c2"]["per_layer"]]
        L = len(rsa)
        cross = next((l for l in range(L)
                      if all(rsa[k] >= 0.97 for k in range(l, min(l + 3, layer_A + 1)))), None)
        pre = max(rsa[:max(1, cross or 1)]) if cross else None
        seams[c] = {"consolidation_layer_rsa_ge_0.97": cross,
                    "frac_depth": (round(cross / L, 3) if cross is not None else None),
                    "pre_seam_max_rsa": (round(pre, 3) if pre else None), "n_layers": L}
    X["seam_per_cell"] = seams

    # ---- (3) boundary-emotion overlap (10-NN purity, all 171, per cell)
    purities = {}
    for c in CELLS:
        SA = json.loads((OUT / c / "stageA_summary.json").read_text())
        lab = np.full(171, -1)
        for ci, cl in enumerate(SA["kmeans_partition"]["clusters"]):
            for m_ in cl["members"]: lab[em.index(m_)] = ci
        Vl = V[c][:, CELLS[c]["layer"], :]
        C = nf.cos(Vl); np.fill_diagonal(C, -np.inf)
        order = np.argsort(-C, axis=1)
        purities[c] = np.array([(lab[order[i, :10]] == lab[i]).mean() for i in range(171)])
    cells = list(CELLS)
    corr = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            corr[i, j] = np.corrcoef(purities[cells[i]], purities[cells[j]])[0, 1]
    bottom = {c: set(np.argsort(purities[c])[:17]) for c in cells}   # bottom decile
    inter = set.intersection(*bottom.values())
    union_ct = {em[i]: sum(1 for c in cells if i in bottom[c]) for i in range(171)}
    always = sorted([e for e, k in union_ct.items() if k == 6])
    moral = ["regretful", "remorseful", "guilty", "vindictive", "vengeful", "spiteful"]
    X["boundary_overlap"] = {
        "purity_rank_corr_mean_offdiag": round(float((corr.sum() - 6) / 30), 3),
        "bottom_decile_in_all_six": always,
        "moral_emotion_purity_mean_across_cells": {e: round(float(np.mean([purities[c][em.index(e)] for c in cells])), 3) for e in moral},
        "loving_purity_mean": round(float(np.mean([purities[c][em.index("loving")] for c in cells])), 3)}

    # ---- (4) consolidation curves (P3.4 pathway per cell)
    idx45, pl, ar = nf.load_ratings(REPO / "data/ratings/russell_mehrabian_1977.csv", em)
    consol = {}
    for c in CELLS:
        L = V[c].shape[1]
        rows = []
        for l in range(L):
            coords, ratio = nf.pca(V[c][:, l, :], 5)
            sc45 = coords[idx45]
            va = abs(float(np.corrcoef(sc45[:, 0], pl)[0, 1]))
            rs = [abs(float(np.corrcoef(sc45[:, k], ar)[0, 1])) for k in range(5)]
            rows.append({"layer": l, "valence_r": round(va, 3), "arousal_pc2_r": round(rs[1], 3),
                         "arousal_best_pc": int(np.argmax(rs)) + 1})
        first_pc2 = next((r_["layer"] for r_ in rows if r_["arousal_best_pc"] == 2 and
                          all(x["arousal_best_pc"] == 2 for x in rows[r_["layer"]:min(r_["layer"] + 3, L)])), None)
        consol[c] = {"per_layer": rows, "arousal_settles_on_pc2_from": first_pc2,
                     "settle_frac_depth": (round(first_pc2 / L, 3) if first_pc2 is not None else None)}
        print(f"consolidation {c}: arousal settles on PC2 from L{first_pc2} "
              f"(frac {consol[c]['settle_frac_depth']})")
    X["consolidation"] = {c: {k: consol[c][k] for k in ("arousal_settles_on_pc2_from", "settle_frac_depth")} for c in CELLS}
    (OUT / "consolidation_curves.json").write_text(json.dumps(consol, indent=2))

    # ---- (5) Table-12-style membership tables
    for c in CELLS:
        SA = json.loads((OUT / c / "stageA_summary.json").read_text())
        lines = [f"# {c} — k=10 membership (seed 0, cleaned, analysis layer L{CELLS[c]['layer']})",
                 "", "| rank (by mean PC1) | n | members |", "|---|---|---|"]
        for cl in SA["kmeans_partition"]["clusters"]:
            lines.append(f"| {cl['rank']} | {len(cl['members'])} | {', '.join(cl['members'])} |")
        (OUT / f"table12_{c}.md").write_text("\n".join(lines))

    (OUT / "extracts_summary.json").write_text(json.dumps(X, indent=2))
    print("wrote fig84_depth_tuning.json, consolidation_curves.json, extracts_summary.json, table12_*.md")
    print("seams:", {c: f"L{s['consolidation_layer_rsa_ge_0.97']} (frac {s['frac_depth']})"
                     for c, s in seams.items() if isinstance(s, dict)})
    print("boundary: purity rank-corr", X["boundary_overlap"]["purity_rank_corr_mean_offdiag"],
          "| always-boundary:", X["boundary_overlap"]["bottom_decile_in_all_six"])


if __name__ == "__main__":
    main()
