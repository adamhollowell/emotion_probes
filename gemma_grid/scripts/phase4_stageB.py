#!/usr/bin/env python3
"""Phase 4 / Stage B — between-cell C4 contrasts (PREREGISTRATION.md (c) C4).

Signals per cell pair, ALL on dimension-free objects (registered constraint —
raw activation vectors of different d_model are never compared):
  i   cross-cell RSA          — similarity of the two 171x171 cosine matrices
                                (upper triangles; cosine = registered headline,
                                Pearson/Spearman reported alongside, §6.1 method)
  ii  Procrustes alignment    — orthogonal Procrustes on standardized
                                per-emotion PC-coordinate matrices (k=2
                                headline: the valence/arousal plane; k=3,5
                                supplementary); rotation angle + disparity
  iii cross-cell ARI          — ARI between the two cells' canonical k=10
                                partitions (Stage A, seed 0), read against the
                                cells' own 20-seed self-ARI bands (Stage A)
  iv  per-emotion displacement — row-wise distance after Procrustes (k=2),
                                ranked; loving/warm-affiliative tracked

Emotion identity is ALWAYS joined by name (loader contract, PHASE4_SCOPE §2);
all six canonical emotion lists are alphabetical and identical — asserted.

Outputs: gemma_grid/geometry_out/stageB_contrasts.json (+ displacement CSVs).
Point estimates ONLY. Floors/error bars for i, ii, iv are pre-registered to be
measured at Stage C (story bootstrap) before any of these numbers is graded;
iii is readable against the self-ARI band rule (< band-2sigma => reorg) but is
still ledgered only after Phase 5.

Usage (repo root): python gemma_grid/scripts/phase4_stageB.py
"""
import json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf
from phase4_stageA import CELLS  # cell registry (paths, layers)

OUT = REPO / "gemma_grid" / "geometry_out"

# Registered contrast families (PREREGISTRATION (d) test definitions)
TUNING = [("gemma-2-2b-base", "gemma-2-2b-it"),
          ("gemma-2-9b-base", "gemma-2-9b-it"),
          ("gemma-2-27b-base", "gemma-2-27b-it")]
SCALE = [("gemma-2-2b-base", "gemma-2-9b-base"), ("gemma-2-9b-base", "gemma-2-27b-base"),
         ("gemma-2-2b-base", "gemma-2-27b-base"),
         ("gemma-2-2b-it", "gemma-2-9b-it"), ("gemma-2-9b-it", "gemma-2-27b-it"),
         ("gemma-2-2b-it", "gemma-2-27b-it")]


def load_cell(cell):
    spec = CELLS[cell]
    em, V = nf.load_vectors(REPO / spec["vec"])
    Vl = V[:, spec["layer"], :]
    coords, ratio = nf.pca(Vl, 5)
    if nf.amean(coords[:, 0], em, nf.POS) < nf.amean(coords[:, 0], em, nf.NEG): coords[:, 0] *= -1
    if nf.amean(coords[:, 1], em, nf.HI) < nf.amean(coords[:, 1], em, nf.LO): coords[:, 1] *= -1
    C = nf.cos(Vl)
    SA = json.loads((OUT / cell / "stageA_summary.json").read_text())
    lab = np.full(len(em), -1)
    for ci, cl in enumerate(SA["kmeans_partition"]["clusters"]):
        for m in cl["members"]:
            lab[em.index(m)] = ci
    assert (lab >= 0).all()
    band = SA["kmeans_seed_sweep"]
    return {"em": em, "coords": coords, "C": C, "labels": lab,
            "self_ari_mean": band["self_ari_mean"], "self_ari_std": band["self_ari_std"],
            "loving_cluster": set(SA["kmeans_partition"]["loving_cluster"])}


def rsa(Ca, Cb, n):
    iu = np.triu_indices(n, 1)
    a, b = Ca[iu], Cb[iu]
    cos_sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    pear = float(np.corrcoef(a, b)[0, 1])
    from scipy.stats import spearmanr
    return {"cosine": cos_sim, "pearson": pear,
            "spearman": float(spearmanr(a, b).statistic)}


def procrustes(Xa, Xb, k):
    """Orthogonal Procrustes on standardized (centered, unit-Frobenius) k-PC
    coordinates: rotation R aligning A->B, disparity = ||A R - B||_F^2, and
    per-emotion residual rows. Scale-free so cells of different coordinate
    magnitude are comparable (the 27B coords are ~30x the 2B ones)."""
    from scipy.linalg import orthogonal_procrustes
    A = Xa[:, :k] - Xa[:, :k].mean(0); A /= np.linalg.norm(A)
    B = Xb[:, :k] - Xb[:, :k].mean(0); B /= np.linalg.norm(B)
    R, scale = orthogonal_procrustes(A, B)
    resid = A @ R - B
    out = {"disparity": float((resid ** 2).sum()),
           "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))) if k == 2 else None,
           "det_R": float(np.linalg.det(R))}
    return out, np.linalg.norm(resid, axis=1)


def contrast(cells, a, b):
    A, B = cells[a], cells[b]
    assert A["em"] == B["em"], "emotion lists differ — name-join violated"
    em, n = A["em"], len(A["em"])
    r = {"pair": f"{a}<->{b}", "rsa": rsa(A["C"], B["C"], n)}
    pro2, disp = procrustes(A["coords"], B["coords"], 2)
    r["procrustes_k2"] = pro2
    r["procrustes_k3"] = procrustes(A["coords"], B["coords"], 3)[0]
    r["procrustes_k5"] = procrustes(A["coords"], B["coords"], 5)[0]
    from sklearn.metrics import adjusted_rand_score as ari
    x = float(ari(A["labels"], B["labels"]))
    bands = {c: cells[c]["self_ari_mean"] - 2 * cells[c]["self_ari_std"] for c in (a, b)}
    r["cross_ari"] = {"ari": x, "band_a_mean_minus_2sd": round(bands[a], 4),
                      "band_b_mean_minus_2sd": round(bands[b], 4),
                      "below_both_bands": bool(x < min(bands.values()))}
    order = np.argsort(-disp)
    warm = A["loving_cluster"] | B["loving_cluster"]
    r["displacement"] = {
        "unit": "post-Procrustes residual, unit-Frobenius standardized k=2 plane",
        "mean": float(disp.mean()), "max": float(disp.max()),
        "top10": [(em[i], float(disp[i])) for i in order[:10]],
        "bottom5": [(em[i], float(disp[i])) for i in order[-5:]],
        "loving": {"value": float(disp[em.index("loving")]),
                   "rank_of_171": int(np.where(order == em.index("loving"))[0][0]) + 1},
        "warm_cluster_mean": float(np.mean([disp[em.index(e)] for e in warm if e in em])),
        "nonwarm_mean": float(np.mean([disp[i] for i in range(n) if em[i] not in warm])),
    }
    return r, disp


def main():
    cells = {c: load_cell(c) for c in CELLS}
    results = {"family_tuning": [], "family_scale": [], "displacement_csv": {}}
    for fam, pairs in [("family_tuning", TUNING), ("family_scale", SCALE)]:
        for a, b in pairs:
            r, disp = contrast(cells, a, b)
            results[fam].append(r)
            results["displacement_csv"][r["pair"]] = None  # written below
            em = cells[a]["em"]
            csvp = OUT / f"stageB_displacement__{a}__vs__{b}.csv"
            with open(csvp, "w") as f:
                f.write("emotion,displacement\n")
                for i in np.argsort(-disp):
                    f.write(f"{em[i]},{disp[i]:.6f}\n")
            results["displacement_csv"][r["pair"]] = csvp.name
            print(f"{r['pair']}: RSA cos {r['rsa']['cosine']:.4f} (pear {r['rsa']['pearson']:.4f}) | "
                  f"Procrustes k2 disp {r['procrustes_k2']['disparity']:.4f} rot {r['procrustes_k2']['rotation_deg']:.1f}° | "
                  f"ARI {r['cross_ari']['ari']:.3f} (below bands: {r['cross_ari']['below_both_bands']}) | "
                  f"loving disp {r['displacement']['loving']['value']:.4f} rank {r['displacement']['loving']['rank_of_171']}")
    (OUT / "stageB_contrasts.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT/'stageB_contrasts.json'}")


if __name__ == "__main__":
    main()
