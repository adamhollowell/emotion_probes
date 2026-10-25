#!/usr/bin/env python3
"""Analysis A (pod rider) — displacement rank-stability bootstrap (attaches to G5.4a/G5.4b).
Cleanup-basis resampling: per draw, resample the 500 neutral dialogues w/ replacement
(SHARED draw indices across all cells, LC-51), re-derive the 50%-variance analysis-layer
cleanup basis (reference procedure), re-clean the saved raw 171 vectors, recompute per-emotion
displacement for the 3 tuning pairs (reference stageB Procrustes-k2), record 171 ranks/pair.
Only new code = the resample wrapper + collector; basis/displacement are the reference paths.
B=1000, seed=20260827, all six cells. Output: analysisA_displacement_bootstrap.json (+CSVs).
Run: python3 analysisA_displacement_bootstrap.py [--draws N] [--validate]
"""
import os, sys, json, hashlib, argparse
import numpy as np
from scipy.linalg import orthogonal_procrustes
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO,"gemma_grid","scripts"))
import noise_floor as nf
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(_v,"1")
B=1000; SEED=20260827
POS=nf.POS; NEG=nf.NEG; HI=nf.HI; LO=nf.LO
TUNING=[("gemma-2-2b-base","gemma-2-2b-it"),("gemma-2-9b-base","gemma-2-9b-it"),("gemma-2-27b-base","gemma-2-27b-it")]
CELLS=["gemma-2-2b-base","gemma-2-2b-it","gemma-2-9b-base","gemma-2-9b-it","gemma-2-27b-base","gemma-2-27b-it"]
LAYER={"gemma-2-2b-base":17,"gemma-2-2b-it":17,"gemma-2-9b-base":28,"gemma-2-9b-it":28,"gemma-2-27b-base":31,"gemma-2-27b-it":31}
def raw_path(c): return "data/vectors/emotion_vectors_171.pt" if c=="gemma-2-27b-base" else f"gemma_grid/data/{c}/vectors/canonical.pt"
def neut_path(c): return "gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__neutral.pt" if c=="gemma-2-27b-base" else f"gemma_grid/data/{c}/vectors/{c}__neutral.pt"
TRIO=["vindictive","vengeful","spiteful"]

def amean(c,em,names):
    idx=[em.index(x) for x in names if x in em]; return float(c[idx].mean()) if idx else 0.0
def signed_coords(Vl,em):
    co,_=nf.pca(Vl,5)
    if amean(co[:,0],em,POS)<amean(co[:,0],em,NEG): co[:,0]*=-1
    if amean(co[:,1],em,HI)<amean(co[:,1],em,LO): co[:,1]*=-1
    return co
def displacement(coA,coB):
    A=coA[:,:2]-coA[:,:2].mean(0); A/=np.linalg.norm(A); Bm=coB[:,:2]-coB[:,:2].mean(0); Bm/=np.linalg.norm(Bm)
    R,_=orthogonal_procrustes(A,Bm); resid=A@R-Bm; return np.linalg.norm(resid,axis=1)

def load_cell(c):
    import torch
    p=torch.load(os.path.join(REPO,raw_path(c)),map_location="cpu",weights_only=False)
    em=list(p["emotions"]); Vraw=np.asarray(p["vectors_raw"],dtype=np.float64)[:,LAYER[c],:]
    N=np.asarray(torch.load(os.path.join(REPO,neut_path(c)),map_location="cpu",weights_only=False)["acts"],dtype=np.float64)[:,LAYER[c],:]
    return em,Vraw,N   # (171,d),(500,d)
def clean(Vraw,Nsub):
    Xc=Nsub-Nsub.mean(0,keepdims=True); _,S,Vh=np.linalg.svd(Xc,full_matrices=False)
    r=(S**2)/(S**2).sum(); k=int(np.argmax(np.cumsum(r)>=0.5))+1; Vk=Vh[:k]
    return Vraw-(Vraw@Vk.T)@Vk

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--draws",type=int,default=B); ap.add_argument("--validate",action="store_true")
    a=ap.parse_args()
    data={c:load_cell(c) for c in CELLS}
    ncell={c:data[c][2].shape[0] for c in CELLS}
    assert all(v==500 for v in ncell.values())
    if a.validate:
        # full-neutral (no resample) must reproduce canonical stageB displacement CSV
        import csv
        for A_,Bc in TUNING:
            emA,VrA,NA=data[A_]; emB,VrB,NB=data[Bc]
            coA=signed_coords(clean(VrA,NA),emA); coB=signed_coords(clean(VrB,NB),emB)
            disp=displacement(coA,coB)
            cp=os.path.join(REPO,f"gemma_grid/geometry_out/stageB_displacement__{A_}__vs__{Bc}.csv")
            ref={r["emotion"]:float(r["displacement"]) for r in csv.DictReader(open(cp))}
            mx=max(abs(ref[emA[i]]-disp[i]) for i in range(len(emA)))
            print(f"VALIDATE {A_[8:]} vs {Bc[8:]}: full-neutral displacement max|diff vs canonical CSV| = {mx:.2e}")
        return
    rng=np.random.default_rng(SEED)
    em0=data[CELLS[0]][0]; n=len(em0); li=em0.index("loving"); trio_idx=[em0.index(t) for t in TRIO if t in em0]
    ranks={p:np.zeros((a.draws,n),dtype=np.int32) for p in range(3)}
    for d in range(a.draws):
        sel=rng.integers(0,500,500)   # SHARED across all cells this draw
        co={}
        for c in CELLS:
            em,Vraw,N=data[c]; co[c]=signed_coords(clean(Vraw,N[sel]),em)
        for pi,(A_,Bc) in enumerate(TUNING):
            disp=displacement(co[A_],co[Bc]); order=np.argsort(-disp)
            rk=np.empty(n,dtype=np.int32); rk[order]=np.arange(1,n+1); ranks[pi][d]=rk
    # summaries
    out={"record":"Analysis A — displacement rank-stability bootstrap (G5.4a/G5.4b).",
         "params":{"B":a.draws,"seed":SEED,"shared_draw_indices":True,"pairs":[f"{x}__{y}" for x,y in TUNING]},
         "trio":TRIO,"pairs":{}}
    outdir=os.path.join(REPO,"gemma_grid/geometry_out"); os.makedirs(outdir,exist_ok=True)
    for pi,(A_,Bc) in enumerate(TUNING):
        R=ranks[pi]; key=f"{A_}__vs__{Bc}"
        rank_ci={em0[i]:[int(np.percentile(R[:,i],2.5)),int(np.percentile(R[:,i],97.5))] for i in range(n)}
        trio_top10={t:float(np.mean(R[:,em0.index(t)]<=10)) for t in TRIO if t in em0}
        trio_mean=R[:,trio_idx].mean(1); trio_mean_topdecile=float(np.mean(trio_mean<=n*0.1))
        lov_ci=[int(np.percentile(R[:,li],2.5)),int(np.percentile(R[:,li],97.5))]
        out["pairs"][key]={"trio_each_top10_frac":trio_top10,"trio_mean_rank_top_decile_frac":trio_mean_topdecile,
                           "loving_rank_ci95":lov_ci,"rank_ci95_per_emotion":rank_ci}
        # draw-level CSV for audit
        import csv
        with open(os.path.join(outdir,f"analysisA_ranks__{key}.csv"),"w",newline="") as f:
            w=csv.writer(f); w.writerow(["draw"]+em0)
            for d in range(a.draws): w.writerow([d]+list(map(int,R[d])))
    outp=os.path.join(outdir,"analysisA_displacement_bootstrap.json")
    open(outp,"w").write(json.dumps(out,indent=1))
    print("sha256:",hashlib.sha256(open(outp,"rb").read()).hexdigest())
    for k,v in out["pairs"].items(): print(k,"trio_top10",v["trio_each_top10_frac"],"trio_mean_topdecile",v["trio_mean_rank_top_decile_frac"],"loving_ci",v["loving_rank_ci95"])

if __name__=="__main__":
    main()
