#!/usr/bin/env python3
"""
Noise-floor probe: compute every scalar C1-C4 signal from an anchor vector
set, so two independent process runs can be diffed to measure the pipeline's
run-to-run reproducibility floor (PLAN §6, GATE 1 item (e)).

Deterministic signals (PCA variance, correlations, neighbour cosines, norms,
layer stability, loving position) are computed once per run under fixed inputs
— the run-to-run |Δ| is pure FP/BLAS/threading noise. The one stochastic
signal, k=10 k-means ARI, has a floor that is SEED SPREAD, not FP noise, so it
is characterised here by a within-run multi-seed sweep (min/mean/std of ARI to
the locked partition, plus cross-seed self-ARI) rather than by the two-run diff.

Reuses the exact anchor definitions from scripts/verify_reproduction.py so the
floor is measured on the same objects the study reports.

Usage:
    python gemma_grid/scripts/noise_floor.py --out run1.json
    python gemma_grid/scripts/noise_floor.py --out run2.json
    # then diff run1.json vs run2.json (see noise_floor_diff.py)
"""
import argparse, csv, json
from pathlib import Path
import numpy as np, torch


POS = ["happy","joyful","content","blissful","delighted","cheerful","optimistic","loving"]
NEG = ["afraid","sad","angry","miserable","panicked","terrified","depressed","heartbroken"]
HI  = ["enraged","panicked","ecstatic","furious","terrified","hysterical","thrilled","alarmed"]
LO  = ["calm","serene","sleepy","relaxed","listless","peaceful","droopy","at ease"]
PROBES = ["afraid","joyful","angry","sad","loving","calm","proud","grateful","compassionate"]


def load_vectors(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    em = list(p["emotions"]); V = p["vectors"]
    V = (V if torch.is_tensor(V) else torch.as_tensor(V)).float().numpy()
    return em, V


def pca(Vl, k=5):
    Vc = Vl - Vl.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Vc, full_matrices=False)
    return U[:, :k] * S[:k], (S[:k] ** 2) / (S ** 2).sum()


def cos(Vl):
    n = np.clip(np.linalg.norm(Vl, axis=1, keepdims=True), 1e-12, None)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Vn = Vl / n
        return Vn @ Vn.T


def amean(c, em, names):
    idx = [em.index(x) for x in names if x in em]
    return float(c[idx].mean()) if idx else 0.0


def load_ratings(path, em):
    reader = csv.DictReader(open(path)); cols = reader.fieldnames or []
    def find(*subs):
        for c in cols:
            if all(s in c.lower() for s in subs): return c
        return None
    term_c = find("term") or find("emotion") or cols[0]
    pl_c = find("pleasure","mean") or find("pleasure") or find("valence")
    ar_c = find("arousal","mean") or find("arousal")
    rr = {}
    for row in reader:
        rr[row[term_c].strip().lower()] = (float(row[pl_c]), float(row[ar_c]))
    overlap = [e for e in em if e.lower() in rr]
    idx = [em.index(e) for e in overlap]
    pl = np.array([rr[e.lower()][0] for e in overlap])
    ar = np.array([rr[e.lower()][1] for e in overlap])
    return idx, pl, ar


def compute(vectors_path, ratings_path, locked_path, layer, kmeans_seeds):
    em, V = load_vectors(vectors_path)
    n, L, d = V.shape
    Vl = V[:, layer, :]
    coords, ratio = pca(Vl, 5)
    # sign convention identical to verify_reproduction.py
    if amean(coords[:, 0], em, POS) < amean(coords[:, 0], em, NEG): coords[:, 0] *= -1
    if amean(coords[:, 1], em, HI) < amean(coords[:, 1], em, LO): coords[:, 1] *= -1

    sig = {}
    sig["var_pc1"] = float(100 * ratio[0]); sig["var_pc2"] = float(100 * ratio[1]); sig["var_pc3"] = float(100 * ratio[2])
    li = em.index("loving")
    sig["loving_pc1"] = float(coords[li, 0]); sig["loving_pc2"] = float(coords[li, 1])

    idx, pl, ar = load_ratings(ratings_path, em)
    c1 = coords[idx, 0].copy(); c2 = coords[idx, 1].copy()
    if np.corrcoef(c1, pl)[0, 1] < 0: c1 = -c1
    if np.corrcoef(c2, ar)[0, 1] < 0: c2 = -c2
    sig["valence_r"] = float(np.corrcoef(c1, pl)[0, 1])
    sig["arousal_r"] = float(np.corrcoef(c2, ar)[0, 1])
    # discriminant off-diagonals (§5.2): use the same signed axes
    sig["offdiag_pc1_arousal"] = float(np.corrcoef(c1, ar)[0, 1])
    sig["offdiag_pc2_pleasure"] = float(np.corrcoef(c2, pl)[0, 1])

    # neighbour top-1 cosine per probe (identity + value)
    C = cos(Vl)
    for p in PROBES:
        if p not in em: continue
        i = em.index(p); od = np.argsort(-C[i])
        j = next(k for k in od if k != i)
        sig[f"nbr_{p}_cos"] = float(C[i, j])

    # norm -> valence correlation (§5.4)
    norms = np.linalg.norm(Vl, axis=1)
    sig["norm_valence_r"] = float(np.corrcoef(norms, coords[:, 0])[0, 1])
    sig["norm_arousal_abs_r"] = float(np.corrcoef(norms, np.abs(coords[:, 1]))[0, 1])

    # layer stability (§6.1), sampled exactly as the verifier
    samp = sorted(set(np.linspace(0, L - 1, min(14, L)).round().astype(int).tolist()))
    iu = np.triu_indices(n, 1); tris = np.stack([cos(V[:, l, :])[iu] for l in samp])
    M = np.corrcoef(tris)
    ml = [k for k, l in enumerate(samp) if l >= L // 2]
    sub = M[np.ix_(ml, ml)]
    sig["layerstab_all"] = float((M.sum() - len(samp)) / (len(samp) ** 2 - len(samp)))
    sig["layerstab_mid_late"] = float((sub.sum() - len(ml)) / (len(ml) ** 2 - len(ml)))

    # k-means ARI: stochastic. Sweep seeds -> spread to locked partition + self-ARI.
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score as ari
    locked = json.loads(Path(locked_path).read_text())
    k = locked["kmeans"]["k"]
    locked_lab = np.full(n, -1)
    for ci, cl in enumerate(locked["kmeans"]["clusters"]):
        for m in cl["members"]:
            if m in em: locked_lab[em.index(m)] = ci
    labs = []
    for s in kmeans_seeds:
        labs.append(KMeans(n_clusters=k, n_init=10, random_state=int(s)).fit(Vl).labels_)
    to_locked = [ari(locked_lab, lb) for lb in labs]
    self_ari = [ari(labs[a], labs[b]) for a in range(len(labs)) for b in range(a + 1, len(labs))]
    kmeans = {
        "seeds": list(map(int, kmeans_seeds)),
        "ari_to_locked_min": float(np.min(to_locked)),
        "ari_to_locked_mean": float(np.mean(to_locked)),
        "ari_to_locked_std": float(np.std(to_locked)),
        "self_ari_min": float(np.min(self_ari)),
        "self_ari_mean": float(np.mean(self_ari)),
        "self_ari_std": float(np.std(self_ari)),
    }
    return {"layer": layer, "n": n, "L": L, "d": d, "deterministic": sig, "kmeans": kmeans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, default=Path("data/vectors/vectors_171.pt"))
    ap.add_argument("--ratings", type=Path, default=Path("data/ratings/russell_mehrabian_1977.csv"))
    ap.add_argument("--locked", type=Path, default=Path("geometry_out/geometry_quick_summary.json"))
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = compute(a.vectors, a.ratings, a.locked, a.layer, kmeans_seeds=list(range(20)))
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
