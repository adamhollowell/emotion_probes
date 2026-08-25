#!/usr/bin/env python3
"""§4 clustering / family structure at L31.

Producer of record for geometry_out/s4_clusters.json (seed-0 canonical KMeans
partition, 50-seed ARI stability, §4.3 cohesion). The canonical partition it emits
is the single-authority L31 partition every downstream consumer reads.

Run from the repo root:
    python scripts/s4_clusters.py                                   # canonical vectors
    python scripts/s4_clusters.py data/vectors/emotion_vectors_171.pt
"""
import sys, os, json
import numpy as np
sys.path.insert(0, "outputs")  # torch-free load_pt ships here; falls back to torch
try:
    from load_pt import load_pt
except Exception:
    import torch
    def load_pt(p): return torch.load(p, map_location="cpu", weights_only=False)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def _stamp(vec_path, basis="canonical", extra=None):
    import hashlib, time
    sha = None
    if os.path.exists(vec_path):
        h = hashlib.sha256()
        with open(vec_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
    import sklearn
    d = {"basis": basis, "vectors": vec_path, "vectors_sha256": sha,
         "seed": 0, "n_init": 10, "sklearn": sklearn.__version__,
         "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if extra: d.update(extra)
    return d


VEC = sys.argv[1] if len(sys.argv) > 1 else "data/vectors/emotion_vectors_171.pt"
tag = "canonical"
LAYER = 31; K = 10
p = load_pt(VEC); em = [str(e) for e in p["emotions"]]
V = np.asarray(p["vectors"], dtype=np.float64)[:, LAYER, :]
n = len(em)

# cosine matrix + §4.1 neighbors
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
C = Vn @ Vn.T
def nbrs(e, k=3):
    i = em.index(e); o = np.argsort(-C[i]); return [(em[j], round(float(C[i,j]),3)) for j in o if j != i][:k]
out = {"tag": tag, "neighbors": {e: nbrs(e) for e in ("afraid","joyful","compassionate")}}

# seed-0 canonical partition (geometry_full.py convention: unnormalized V)
km0 = KMeans(n_clusters=K, n_init=10, random_state=0).fit(V)
lab0 = km0.labels_
# PC1 for family ordering
Xc = V - V.mean(0, keepdims=True); U,S,_ = np.linalg.svd(Xc, full_matrices=False); pc1 = U[:,0]*S[0]
POS=["happy","joyful","content","blissful","delighted","cheerful","optimistic","loving"]
NEG=["afraid","sad","angry","miserable","panicked","terrified","depressed","heartbroken"]
am=lambda c,ns: float(c[[em.index(x) for x in ns]].mean())
if am(pc1,POS) < am(pc1,NEG): pc1 = -pc1
fams = []
for c in np.argsort([-pc1[lab0==c].mean() for c in range(K)]):
    mem = [em[i] for i in range(n) if lab0[i]==c]
    fams.append({"size": len(mem), "mean_pc1": round(float(pc1[lab0==c].mean()),1), "members": mem})
out["families"] = fams
out["family_sizes"] = [f["size"] for f in fams]

# 50-seed ARI
labs = [KMeans(n_clusters=K, n_init=10, random_state=s).fit(V).labels_ for s in range(50)]
aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(50) for j in range(i+1,50)]
vs0  = [adjusted_rand_score(labs[s], lab0) for s in range(1,50)]
out["ari"] = {"mean_pairwise": round(float(np.mean(aris)),3), "min_pairwise": round(float(np.min(aris)),3),
              "mean_vs_seed0": round(float(np.mean(vs0)),3)}

# §4.3 cohesion: fraction of top-10 cosine neighbors inside own seed-0 cluster
top10 = np.argsort(-C, axis=1)[:, 1:11]
frac = np.array([np.mean(lab0[top10[i]] == lab0[i]) for i in range(n)])
out["cohesion"] = {"mean_top10_incluster": round(float(frac.mean()),3)}
# mutually nearest pairs
nn1 = np.array([np.argsort(-C[i])[1] for i in range(n)])
out["cohesion"]["mutual_nn_pairs"] = int(sum(1 for i in range(n) if nn1[nn1[i]] == i and i < nn1[i]))
# per-family purity for the three named regions (by family containing marker emotion)
for name, marker in (("hostile_anger","furious"),("fear","terrified"),("warm","compassionate")):
    fam = lab0[em.index(marker)]
    out["cohesion"][name] = round(float(frac[lab0==fam].mean()),3)
# per-emotion stability across seeds (co-assignment consistency with seed-0 cluster mates)
stab = np.zeros(n)
for L in labs:
    for i in range(n):
        mates0 = (lab0 == lab0[i]); mates0[i] = False
        if mates0.sum(): stab[i] += np.mean(L[mates0] == L[i])
stab /= len(labs)
o = np.argsort(stab)
out["least_stable"] = [em[i] for i in o[:8]]
out["_provenance"] = _stamp(VEC, tag, {"note": "fresh seed-0 KMeans(k=10) on emotion_vectors_171.pt cleaned L31; canonical native partition"})
json.dump(out, open("geometry_out/s4_clusters.json","w"), indent=1)
print(json.dumps({k:v for k,v in out.items() if k != "families"}, indent=1))
print("family sizes:", out["family_sizes"])
for f in fams: print(f["size"], f["members"][:6])
