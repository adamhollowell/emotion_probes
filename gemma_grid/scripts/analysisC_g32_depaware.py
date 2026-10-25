#!/usr/bin/env python3
"""Analysis C (pod rider) — dependence-aware bootstrap for G3.2 arousal-scale contrast.
Template: G3.3's phase4_c3_grade.py::gap_boot machinery (resample 45 RM emotions jointly,
PCA fixed on all 171 restricted to 45, seed 0, B=2000). Statistic per resolution doc:
recompute arousal_r per cell, recompute the 2b->27b contrast (base + IT). Reference
arousal_r computation (nf.pca PC2 + |corr| with RM arousal) unchanged; only new code is
the resample wrapper + collector. Output: geometry_out/analysisC_g32_depaware.json
"""
import os, json, hashlib, sys
import numpy as np
from scipy.stats import norm
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO,"gemma_grid","scripts"))
import noise_floor as nf
B=2000; SEED=0; N45=45
z=np.arctanh; p2=lambda zz: float(2*(1-norm.cdf(abs(zz))))
SE_REG=float(np.sqrt(2/(N45-3)))   # registered Fisher pair SE
CHAINS={"base":("gemma-2-2b-base","gemma-2-27b-base"),"it":("gemma-2-2b-it","gemma-2-27b-it")}
VEC={"gemma-2-2b-base":"gemma_grid/data/gemma-2-2b-base/vectors/canonical.pt",
     "gemma-2-2b-it":"gemma_grid/data/gemma-2-2b-it/vectors/canonical.pt",
     "gemma-2-27b-base":"data/vectors/emotion_vectors_171.pt",
     "gemma-2-27b-it":"gemma_grid/data/gemma-2-27b-it/vectors/canonical.pt"}
LAYER={"gemma-2-2b-base":17,"gemma-2-2b-it":17,"gemma-2-27b-base":31,"gemma-2-27b-it":31}

def an_pc2(cell):
    em,V=nf.load_vectors(os.path.join(REPO,VEC[cell]))
    co,_=nf.pca(V[:,LAYER[cell],:],5)   # PCA fixed on all 171 (fit-then-restrict)
    return em, co[:,1]

def main():
    ratings=os.path.join(REPO,"data/ratings/russell_mehrabian_1977.csv")
    out={"record":"Analysis C — dependence-aware bootstrap for G3.2 (gap_boot machinery; arousal_r 2b->27b contrast).",
         "params":{"B":B,"seed":SEED,"resample":"45 RM emotions jointly, PCA fixed on 171","registered_SE":SE_REG},
         "chains":{}}
    for chain,(a,b) in CHAINS.items():
        ema,pa=an_pc2(a); emb,pb=an_pc2(b)
        assert ema==emb
        idx,pl,ar=nf.load_ratings(ratings, ema); idx=np.asarray(idx)
        pa45=pa[idx]; pb45=pb[idx]
        def r_of(pc2, sel): return abs(float(np.corrcoef(pc2[sel],ar[sel])[0,1]))
        # point contrast (full 45)
        full=np.arange(N45)
        point_dz=z(min(r_of(pb45,full),0.999999))-z(min(r_of(pa45,full),0.999999))
        # dependence-aware bootstrap SE (gap_boot rng scheme)
        rng=np.random.default_rng(SEED)
        boots=[]
        for _ in range(B):
            sel=rng.integers(0,N45,N45)
            boots.append(z(min(r_of(pb45,sel),0.999999))-z(min(r_of(pa45,sel),0.999999)))
        se_boot=float(np.std(boots))
        out["chains"][chain]={
            "arousal_r_2b":r_of(pa45,full),"arousal_r_27b":r_of(pb45,full),
            "delta_z_point":float(point_dz),
            "p_registered":p2(point_dz/SE_REG),"SE_registered":SE_REG,
            "SE_dependence_aware_bootstrap":se_boot,"p_dependence_aware":p2(point_dz/se_boot)}
    outp=os.path.join(REPO,"gemma_grid/geometry_out/analysisC_g32_depaware.json")
    open(outp,"w").write(json.dumps(out,indent=1))
    print("sha256:",hashlib.sha256(open(outp,"rb").read()).hexdigest())
    print(json.dumps(out["chains"],indent=1))

if __name__=="__main__":
    main()
