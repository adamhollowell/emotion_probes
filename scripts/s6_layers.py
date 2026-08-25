#!/usr/bin/env python3
"""§6 layer dynamics: RSA, variance, consolidation, stability, Table 2.

Producer of record for geometry_out/s6_layers.json (the per-layer source behind
band_summary_table.csv, adjacent_rsa.json, and the §6.2/§6.4 bands). Reads the
canonical emotion vectors and writes the per-layer + banded summary.

Run from the repo root:
    python scripts/s6_layers.py                                   # canonical vectors
    python scripts/s6_layers.py data/vectors/emotion_vectors_171.pt
"""
import sys, os, csv, json
import numpy as np
sys.path.insert(0, "outputs")  # torch-free load_pt ships here; falls back to torch
try:
    from load_pt import load_pt
except Exception:
    import torch
    def load_pt(p): return torch.load(p, map_location="cpu", weights_only=False)


def _stamp(vec_path, basis="canonical"):
    """Self-describing provenance embedded in the output, so a value can never be
    mis-attributed to the wrong basis by filename alone."""
    import hashlib, time
    sha = None
    if os.path.exists(vec_path):
        h = hashlib.sha256()
        with open(vec_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
    return {"basis": basis, "vectors": vec_path, "vectors_sha256": sha,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


VEC = sys.argv[1] if len(sys.argv) > 1 else "data/vectors/emotion_vectors_171.pt"
tag = "canonical"
p = load_pt(VEC); em = [str(e) for e in p["emotions"]]
V = np.asarray(p["vectors"], dtype=np.float64)  # (171, 46, 4608) cleaned per layer
n, L, d = V.shape
POS=["happy","joyful","content","blissful","delighted","cheerful","optimistic","loving"]
NEG=["afraid","sad","angry","miserable","panicked","terrified","depressed","heartbroken"]
HI =["enraged","panicked","ecstatic","furious","terrified","hysterical","thrilled","alarmed"]
LO =["calm","serene","sleepy","relaxed","listless","peaceful","droopy","at ease"]
am = lambda c, ns: float(c[[em.index(x) for x in ns]].mean())
BANDS = {"early_L0_11": range(0,12), "mid_L12_21": range(12,22), "seam_L22_26": range(22,27),
         "canon_L27_34": range(27,35), "late_L35_40": range(35,41), "final_L41_45": range(41,46)}

def cosm(X):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return Xn @ Xn.T

iu = np.triu_indices(n, 1)
C31 = cosm(V[:,31,:]); t31 = C31[iu]
rm = {}
for row in csv.DictReader(open("data/ratings/russell_mehrabian_1977.csv")):
    rm[row["Term"].strip().lower()] = (float(row["Pleasure Mean"]), float(row["Arousal Mean"]))
m45 = [e for e in em if e in rm]; i45 = [em.index(e) for e in m45]
plea = np.array([rm[e][0] for e in m45]); arou = np.array([rm[e][1] for e in m45])
r = lambda a,b: float(np.corrcoef(a,b)[0,1])

per = []
top5_31 = None; top10_31 = None
o31 = np.argsort(-C31, axis=1)
top5_31 = [set(o31[i,1:6]) for i in range(n)]; top10_31 = [set(o31[i,1:11]) for i in range(n)]
for l in range(L):
    X = V[:,l,:]
    Cl = cosm(X); tl = Cl[iu]
    rsa_cos = float(np.dot(tl, t31)/(np.linalg.norm(tl)*np.linalg.norm(t31)))
    rsa_pear = r(tl, t31)
    Xc = X - X.mean(0, keepdims=True); U,S,_ = np.linalg.svd(Xc, full_matrices=False)
    sc = U*S; var = S**2/(S**2).sum()
    if am(sc[:,0],POS) < am(sc[:,0],NEG): sc[:,0] *= -1
    if am(sc[:,1],HI ) < am(sc[:,1],LO ): sc[:,1] *= -1
    if am(sc[:,2],HI ) < am(sc[:,2],LO ): sc[:,2] *= -1
    rv = r(sc[i45,0], plea); ra = r(sc[i45,1], arou)
    # multiple correlation of arousal on PC2+PC3
    A = np.column_stack([np.ones(len(i45)), sc[i45,1], sc[i45,2]])
    beta,*_ = np.linalg.lstsq(A, arou, rcond=None)
    R23 = r(A@beta, arou)
    ol = np.argsort(-Cl, axis=1)
    jac = np.mean([len(set(ol[i,1:6]) & top5_31[i]) / len(set(ol[i,1:6]) | top5_31[i]) for i in range(n)])
    ov10 = np.mean([len(set(ol[i,1:11]) & top10_31[i]) for i in range(n)])
    nrm = np.linalg.norm(X, axis=1)
    per.append({"layer": l, "rsa_cos": rsa_cos, "rsa_pearson": rsa_pear,
                "pc1_var": float(100*var[0]), "valence_r": rv, "arousal_r": ra, "arousal_R23": R23,
                "jaccard5": float(jac), "overlap10": float(ov10),
                "norm_cv": float(nrm.std()/nrm.mean())})
bands = {}
for bn, rng_ in BANDS.items():
    rows = [per[l] for l in rng_]
    bands[bn] = {k: round(float(np.mean([rw[k] for rw in rows])), 3)
                 for k in ("rsa_cos","jaccard5","overlap10","pc1_var","valence_r","arousal_r","norm_cv")}
# adjacent-layer RSA(N,N+1) cos-of-cos (Anthropic Fig-9), between CONSECUTIVE layers — NOT to-L31.
# Fixes the former "L21_L22_step" which mislabeled per[22]["rsa_cos"] (RSA of L22 *to L31*).
_tris = [cosm(V[:,l,:])[iu] for l in range(L)]
adjacent_rsa = [round(float(np.dot(_tris[N],_tris[N+1])/(np.linalg.norm(_tris[N])*np.linalg.norm(_tris[N+1]))),4)
                for N in range(L-1)]
out = {"tag": tag, "per_layer": per, "bands": bands,
       "adjacent_rsa": adjacent_rsa,                        # index N = RSA(LN, LN+1)
       "adjacent_rsa_L21_L22": adjacent_rsa[21], "adjacent_rsa_L43_L44": adjacent_rsa[43],
       "adjacent_below_0p97": [f"L{N}->L{N+1}" for N in range(L-1) if adjacent_rsa[N] < 0.97],
       "L21_L22_step_DEPRECATED": round(per[22]["rsa_cos"], 3),   # was mislabeled = rsa_L22, kept for trace
       "rsa_L21": round(per[21]["rsa_cos"],3), "rsa_L22": round(per[22]["rsa_cos"],3),
       "pc1_peak_layer": int(np.argmax([q["pc1_var"] for q in per])),
       "pc1_peak_val": round(max(q["pc1_var"] for q in per),1),
       "arousal_L10": round(per[10]["arousal_r"],3), "arousal_L26": round(per[26]["arousal_r"],3),
       "arousal_L31": round(per[31]["arousal_r"],3),
       "R23_L26": round(per[26]["arousal_R23"],3), "R23_L31": round(per[31]["arousal_R23"],3), "R23_L40": round(per[40]["arousal_R23"],3),
       "pc2alone_L26": round(abs(per[26]["arousal_r"]),3), "valence_L31": round(per[31]["valence_r"],3),
       "valence_peak": (int(np.argmax([q["valence_r"] for q in per])), round(max(q["valence_r"] for q in per),3)),
       "max_pearson_vs_cos_delta": float(max(abs(q["rsa_cos"]-q["rsa_pearson"]) for q in per))}
out["_provenance"] = _stamp(VEC, tag)
json.dump(out, open("geometry_out/s6_layers.json","w"), indent=1)
b = out["bands"]
print(tag, "bands:")
for k in b: print(" ", k, b[k])
for k in ("rsa_L21","rsa_L22","pc1_peak_layer","pc1_peak_val","arousal_L10","arousal_L26","arousal_L31",
          "R23_L26","R23_L31","R23_L40","valence_L31","valence_peak","max_pearson_vs_cos_delta"):
    print(" ", k, out[k])
