#!/usr/bin/env python3
"""Phase 4 / Stage A — per-cell C1+C2 signals for the Gemma scale×tuning grid.

Computes, for one cell, every registered per-cell signal (PREREGISTRATION.md
(c): C1 #1–9b, C2 #10–11) from the cell's canonical.pt. No GPU, no forward
pass. C3 (embedding baseline) is a separate script.

REUSE, NOT FORK: the deterministic signal definitions are imported directly
from gemma_grid/scripts/noise_floor.py (which itself mirrors
scripts/verify_reproduction.py). Extras below copy their conventions verbatim
from the round-one scripts named next to each function.

Per-cell outputs: gemma_grid/geometry_out/<cell>/stageA_summary.json
If a pod-side noise-floor JSON exists for the cell, the deterministic block is
diffed against it (pod-vs-local reproducibility check) and the max |Δ| stored.

Usage (repo root):
  python gemma_grid/scripts/phase4_stageA.py --cell gemma-2-2b-base
  python gemma_grid/scripts/phase4_stageA.py --all
  python gemma_grid/scripts/phase4_stageA.py --cell gemma-2-27b-base --anchor-check
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf  # locked signal definitions (GATE 1 artifact)

ANCHOR_LOCKED = REPO / "gemma_grid" / "data" / "geometry_quick_summary.json"  # the replication study's locked anchor summary as run, sha256 7aff4a70...
RATINGS = REPO / "data" / "ratings" / "russell_mehrabian_1977.csv"

# Cell registry — layers per RUN_MANIFEST.json (derived rule, asserted below).
CELLS = {
    "gemma-2-2b-base":  dict(vec="gemma_grid/data/gemma-2-2b-base/vectors/canonical.pt",  n_layers=26, d_model=2304, layer=17, sens=[18], nf="nf_2b_base.json"),
    "gemma-2-2b-it":    dict(vec="gemma_grid/data/gemma-2-2b-it/vectors/canonical.pt",    n_layers=26, d_model=2304, layer=17, sens=[18], nf="nf_2b_it.json"),
    "gemma-2-9b-base":  dict(vec="gemma_grid/data/gemma-2-9b-base/vectors/canonical.pt",  n_layers=42, d_model=3584, layer=28, sens=[],   nf="nf_9b_base.json"),
    "gemma-2-9b-it":    dict(vec="gemma_grid/data/gemma-2-9b-it/vectors/canonical.pt",    n_layers=42, d_model=3584, layer=28, sens=[],   nf="nf_9b_it.json"),
    "gemma-2-27b-base": dict(vec="data/vectors/vectors_171.pt",                           n_layers=46, d_model=4608, layer=31, sens=[],   nf=None),  # anchor = locked round-one artifact
    "gemma-2-27b-it":   dict(vec="gemma_grid/data/gemma-2-27b-it/vectors/canonical.pt",   n_layers=46, d_model=4608, layer=31, sens=[],   nf="nf_27b_it.json"),
}
LAYER_RULE = lambda n: round(2 / 3 * n)  # CLAUDE.md: derived, never a literal


def perm_p(x, target, robs, nperm=5000, seed=0):
    """scripts/analyze_sun_choi_engagement.py #6, with a fixed RNG for
    re-derivability (round one used the global RNG; p is a count, so seeding
    changes nothing material)."""
    rng = np.random.default_rng(seed)
    r = lambda a, b: np.corrcoef(a, b)[0, 1]
    cnt = sum(abs(r(x, rng.permutation(target))) >= abs(robs) for _ in range(nperm))
    return (cnt + 1) / (nperm + 1)


def scale_refit(Vl, em, idx45, arou):
    """Signal #5 (negative control), conventions from
    analyze_sun_choi_engagement.py: refit PCA on the 45-overlap alone; report
    arousal's best PC and PC2 |r|. 'Survives' iff arousal's best PC is PC2."""
    V45 = Vl[idx45] - Vl[idx45].mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(V45, full_matrices=False)
    sc = U[:, :10] * S[:10]
    r = lambda a, b: abs(float(np.corrcoef(a, b)[0, 1]))
    rs = [r(sc[:, k], arou) for k in range(min(10, sc.shape[1]))]
    bk = int(np.argmax(rs))
    return {"arousal_best_pc": bk + 1, "arousal_best_abs_r": rs[bk],
            "arousal_pc2_abs_r": rs[1], "survives_refit": bk == 1}


def canonical_partition(Vl, em, k=10, seed=0):
    """geometry_full.py convention: KMeans(n_init=10, random_state=0) on the
    cleaned analysis-layer vectors (unnormalized); clusters ranked by mean PC1."""
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Vl).labels_
    return lab


def cohesion(Vl, em, lab, topk=10):
    """analyze_nearest_neighbor_atlas.py conventions: cosine 10-NN in-cluster
    purity (mean over emotions) + mutual TOP-1 pair count (the '47' at the
    anchor) + lowest-purity boundary emotions."""
    C = nf.cos(Vl)
    n = len(em)
    np.fill_diagonal(C, -np.inf)
    order = np.argsort(-C, axis=1)
    purity = np.array([(lab[order[i, :topk]] == lab[i]).mean() for i in range(n)])
    top1 = order[:, 0]
    mutual = sum(1 for i in range(n) if top1[top1[i]] == i and i < top1[i])
    worst = np.argsort(purity)[:10]
    return {"mean_purity_top10": float(purity.mean()),
            "mutual_top1_pairs": int(mutual),
            "boundary_lowest_purity": [(em[i], float(purity[i])) for i in worst]}


def c2_curves(V, layer, topk_small=5, topk_big=10):
    """C2 #10–11: per-layer RSA (Pearson + Spearman on cosine-matrix upper
    triangles vs analysis layer — analyze_layerwise_internal_geometry.py) and
    neighbor retention (top-5 overlap, top-10 Jaccard — l31_characterization.py
    / layer-stability convention)."""
    from scipy.stats import spearmanr
    n, L, _ = V.shape
    iu = np.triu_indices(n, 1)
    ref = nf.cos(V[:, layer, :])[iu]
    Cl_ref = nf.cos(V[:, layer, :]); np.fill_diagonal(Cl_ref, -np.inf)
    oref = np.argsort(-Cl_ref, axis=1)
    ref5 = [set(oref[i, :topk_small]) for i in range(n)]
    ref10 = [set(oref[i, :topk_big]) for i in range(n)]
    rows = []
    for l in range(L):
        Cl = nf.cos(V[:, l, :])
        tri = Cl[iu]
        pear = float(np.corrcoef(tri, ref)[0, 1])
        spear = float(spearmanr(tri, ref).statistic)
        np.fill_diagonal(Cl, -np.inf)
        o = np.argsort(-Cl, axis=1)
        ov5 = np.mean([len(set(o[i, :topk_small]) & ref5[i]) for i in range(n)])
        jac10 = np.mean([len(set(o[i, :topk_big]) & ref10[i]) /
                         len(set(o[i, :topk_big]) | ref10[i]) for i in range(n)])
        rows.append({"layer": l, "rsa_pearson": pear, "rsa_spearman": spear,
                     "top5_overlap": float(ov5), "top10_jaccard": float(jac10)})
    band = [r["layer"] for r in rows if r["rsa_pearson"] >= 0.95]
    return {"per_layer": rows, "band_rsa_ge_0.95": band}


def run_cell(cell, spec, outdir, anchor_check=False):
    vec = REPO / spec["vec"]
    em, V = nf.load_vectors(vec)
    n, L, d = V.shape
    assert L == spec["n_layers"] and d == spec["d_model"], f"shape {V.shape} != manifest"
    assert LAYER_RULE(L) == spec["layer"], "derived layer mismatch"
    layer = spec["layer"]
    S = {"cell": cell, "layer": layer, "n": n, "L": L, "d": d,
         "layer_rule": f"round(2/3*{L})={layer}"}

    # ---- deterministic C1 block + 20-seed k-means (locked defs, noise_floor.py)
    base = nf.compute(vec, RATINGS, ANCHOR_LOCKED, layer, kmeans_seeds=list(range(20)))
    S["deterministic"] = base["deterministic"]
    S["kmeans_seed_sweep"] = base["kmeans"]  # ari_to_locked here = vs ANCHOR partition (pod convention)

    # ---- extras: rankings, permutation p, refit, partition, cohesion, top-5
    Vl = V[:, layer, :]
    coords, ratio = nf.pca(Vl, 5)
    if nf.amean(coords[:, 0], em, nf.POS) < nf.amean(coords[:, 0], em, nf.NEG): coords[:, 0] *= -1
    if nf.amean(coords[:, 1], em, nf.HI) < nf.amean(coords[:, 1], em, nf.LO): coords[:, 1] *= -1
    o1, o2 = np.argsort(-coords[:, 0]), np.argsort(-coords[:, 1])
    S["rankings"] = {"pc1_top": [em[i] for i in o1[:10]], "pc1_bot": [em[i] for i in o1[-10:]],
                     "pc2_top": [em[i] for i in o2[:10]], "pc2_bot": [em[i] for i in o2[-10:]]}

    idx45, pl, ar = nf.load_ratings(RATINGS, em)
    c1 = coords[idx45, 0].copy(); c2c = coords[idx45, 1].copy()
    if np.corrcoef(c1, pl)[0, 1] < 0: c1 = -c1
    if np.corrcoef(c2c, ar)[0, 1] < 0: c2c = -c2c
    S["perm_p_5000"] = {"valence": perm_p(c1, pl, np.corrcoef(c1, pl)[0, 1]),
                        "arousal": perm_p(c2c, ar, np.corrcoef(c2c, ar)[0, 1], seed=1)}
    S["scale_refit_45"] = scale_refit(Vl, em, idx45, ar)

    lab = canonical_partition(Vl, em)
    mpc1 = [coords[lab == c, 0].mean() for c in range(10)]
    S["kmeans_partition"] = {"clusters": [{"rank": r + 1, "members": [em[i] for i in range(n) if lab[i] == c]}
                                          for r, c in enumerate(np.argsort(mpc1))],
                             "loving_cluster": [em[i] for i in range(n) if lab[i] == lab[em.index("loving")]]}
    S["cohesion"] = cohesion(Vl, em, lab)

    C = nf.cos(Vl)
    nn = {}
    for p in nf.PROBES:
        if p in em:
            i = em.index(p); od = np.argsort(-C[i])
            nn[p] = [(em[j], float(C[i, j])) for j in od if j != i][:5]
    S["neighbours_top5"] = nn
    norms = np.linalg.norm(V, axis=2)  # (n, L)
    S["norm_cv_across_depth"] = float(np.mean(np.std(norms, axis=1) / np.mean(norms, axis=1)))

    # ---- C2 curves
    S["c2"] = c2_curves(V, layer)

    # ---- 2B sensitivity layers (pre-registered)
    for sl in spec["sens"]:
        sens = nf.compute(vec, RATINGS, ANCHOR_LOCKED, sl, kmeans_seeds=list(range(20)))
        S[f"sensitivity_L{sl}"] = {"deterministic": sens["deterministic"], "kmeans": sens["kmeans"]}

    # ---- pod-vs-local reproducibility diff
    if spec["nf"]:
        pod = json.loads((REPO / "gemma_grid" / "handoff" / "noise_floor" / spec["nf"]).read_text())
        deltas = {k: abs(S["deterministic"][k] - pod["deterministic"][k])
                  for k in pod["deterministic"] if k in S["deterministic"]}
        S["pod_vs_local"] = {"max_abs_delta_deterministic": max(deltas.values()),
                            "argmax": max(deltas, key=deltas.get),
                            "kmeans_self_ari_pod": pod["kmeans"]["self_ari_mean"],
                            "kmeans_self_ari_local": S["kmeans_seed_sweep"]["self_ari_mean"]}

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "stageA_summary.json").write_text(json.dumps(S, indent=2))
    print(f"[{cell}] wrote {outdir/'stageA_summary.json'}")
    print(f"  var {S['deterministic']['var_pc1']:.2f}/{S['deterministic']['var_pc2']:.2f}/{S['deterministic']['var_pc3']:.2f}"
          f"  val_r {S['deterministic']['valence_r']:.3f}  aro_r {S['deterministic']['arousal_r']:.3f}"
          f"  refit_survives={S['scale_refit_45']['survives_refit']}"
          f"  cohesion {S['cohesion']['mean_purity_top10']:.3f}  mutual {S['cohesion']['mutual_top1_pairs']}"
          f"  selfARI {S['kmeans_seed_sweep']['self_ari_mean']:.3f}")
    if "pod_vs_local" in S:
        print(f"  pod-vs-local max|Δ| = {S['pod_vs_local']['max_abs_delta_deterministic']:.2e} ({S['pod_vs_local']['argmax']})")

    if anchor_check:
        anchor_gate(S)
    return S


def anchor_gate(S):
    """Consistency gate vs locked E1 anchor values (verify_reproduction TOLs;
    plus cluster_robustness_summary.json for cohesion/mutual)."""
    locked = json.loads(ANCHOR_LOCKED.read_text())
    rob = json.loads((REPO / "geometry_out" / "cluster_robustness_summary.json").read_text())
    fig8 = json.loads((REPO / "geometry_out" / "fig8_pca_vs_russell_mehrabian_REBUILD_summary.json").read_text())
    checks = []
    d = S["deterministic"]
    lp = locked["pca"]
    checks += [("var_pc1", d["var_pc1"], lp["pc1_pct"], 0.05), ("var_pc2", d["var_pc2"], lp["pc2_pct"], 0.05),
               ("var_pc3", d["var_pc3"], lp["pc3_pct"], 0.05),
               ("loving_pc1", d["loving_pc1"], lp["loving"]["pc1"], abs(lp["loving"]["pc1"]) * 2e-3),
               ("loving_pc2", d["loving_pc2"], lp["loving"]["pc2"], abs(lp["loving"]["pc2"]) * 2e-3),
               ("valence_r", d["valence_r"], fig8["correlations"]["pc1_pleasure_r_from_plotted_points"], 1e-3),
               ("arousal_r", d["arousal_r"], fig8["correlations"]["pc2_arousal_r_from_plotted_points"], 1e-3),
               ("cohesion", S["cohesion"]["mean_purity_top10"], rob["mean_purity"], 1e-6),
               ("mutual_pairs", S["cohesion"]["mutual_top1_pairs"], rob["mutual_nn_pairs"], 0.5),
               ("layerstab_all", d["layerstab_all"], locked["layer_stability"]["all"], 2e-3),
               ("layerstab_mid_late", d["layerstab_mid_late"], locked["layer_stability"]["mid_late"], 2e-3)]
    nfail = 0
    for name, got, want, tol in checks:
        ok = abs(got - want) <= tol
        nfail += (not ok)
        print(f"  ANCHOR {'PASS' if ok else 'FAIL'} {name}: got {got:.4f} locked {want:.4f} tol {tol:g}")
    for key in ["pc1_top", "pc1_bot", "pc2_top", "pc2_bot"]:
        ok = S["rankings"][key] == lp[key]
        nfail += (not ok)
        print(f"  ANCHOR {'PASS' if ok else 'FAIL'} ranking {key}")
    print(f"  ANCHOR GATE: {'ALL PASS' if nfail == 0 else f'{nfail} FAILED'}")
    if nfail: sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=CELLS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--anchor-check", action="store_true")
    a = ap.parse_args()
    todo = list(CELLS) if a.all else [a.cell]
    for cell in todo:
        run_cell(cell, CELLS[cell], REPO / "gemma_grid" / "geometry_out" / cell,
                 anchor_check=(a.anchor_check and cell == "gemma-2-27b-base"))


if __name__ == "__main__":
    main()
