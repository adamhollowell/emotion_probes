#!/usr/bin/env python3
"""Full §6 geometry comparison: figures + JSON.
Run: python geometry_full.py --vectors data/vectors/emotion_vectors_171.pt --layer 31 --out geometry_out/
Optional Fig-8: --ratings valence_arousal_171.csv"""
# --- basis gate: refuse to run unless the vectors match the expected sha256.
# Identify the vectors by their hash, not the filename `emotion_vectors_171.pt`. ---
import hashlib as _hlg, pathlib as _plg
_VECG = _plg.Path("data/vectors/emotion_vectors_171.pt")
_EXPG = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
if _VECG.exists():
    _hg = _hlg.sha256(_VECG.read_bytes()).hexdigest()
    assert _hg == _EXPG, (
        "%s basis mismatch: %s != expected canonical %s -- refusing non-canonical basis"
        % (_VECG, _hg[:12], _EXPG[:12]))
# --- end basis gate ---

import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # tame Accelerate spurious flags
import argparse, json, warnings
from pathlib import Path
import numpy as np, torch
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
np.seterr(all="ignore")

POS=["happy","joyful","content","blissful","delighted","cheerful","optimistic","loving"]
NEG=["afraid","sad","angry","miserable","panicked","terrified","depressed","heartbroken"]
HI =["enraged","panicked","ecstatic","furious","terrified","hysterical","thrilled","alarmed"]
LO =["calm","serene","sleepy","relaxed","listless","peaceful","droopy","at ease"]
PROBES=["afraid","joyful","angry","sad","loving","calm","proud","grateful","compassionate"]
LABELS=set(POS+NEG+HI+LO+["loving","grateful","compassionate","proud","surprised","bored","nostalgic","excited"])

def load(path,raw):
    p=torch.load(path,map_location="cpu",weights_only=False)
    em=list(p["emotions"]); V=p["vectors_raw" if raw else "vectors"]
    V=(V if torch.is_tensor(V) else torch.as_tensor(V)).float().numpy()
    assert V.ndim==3 and V.shape[0]==len(em) and np.isfinite(V).all()
    return em,V

def pca(Vl,k=5):
    Vc=Vl-Vl.mean(0,keepdims=True); U,S,_=np.linalg.svd(Vc,full_matrices=False)
    return U[:,:k]*S[:k], (S[:k]**2)/(S**2).sum()

def cosm(Vl):
    n=np.clip(np.linalg.norm(Vl,axis=1,keepdims=True),1e-12,None); Vn=Vl/n; return Vn@Vn.T

def amean(c,em,ns):
    idx=[em.index(x) for x in ns if x in em]; return float(c[idx].mean()) if idx else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--vectors",type=Path,default=Path("data/vectors/emotion_vectors_171.pt"))
    ap.add_argument("--layer",type=int,default=31); ap.add_argument("--out",type=Path,default=Path("geometry_out"))
    ap.add_argument("--ratings",type=Path,default=None); ap.add_argument("--use-raw",action="store_true")
    ap.add_argument("--k",type=int,default=10); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    em,V=load(a.vectors,a.use_raw); n,L,d=V.shape; Vl=V[:,a.layer,:]
    print(f"Loaded {n}x{L}x{d} ({'raw' if a.use_raw else 'cleaned'}) layer={a.layer}")
    S={"layer":a.layer,"n":n,"L":L,"d":d}

    coords,ratio=pca(Vl,5)
    if amean(coords[:,0],em,POS)<amean(coords[:,0],em,NEG): coords[:,0]*=-1
    if amean(coords[:,1],em,HI )<amean(coords[:,1],em,LO ): coords[:,1]*=-1
    p1,p2=100*ratio[0],100*ratio[1]
    print(f"PC1 {p1:.1f}% PC2 {p2:.1f}% PC3 {100*ratio[2]:.1f}%")
    o1=np.argsort(-coords[:,0]); o2=np.argsort(-coords[:,1]); li=em.index("loving")
    S["pca"]={"pc1_pct":float(p1),"pc2_pct":float(p2),"pc3_pct":float(100*ratio[2]),
        "pc1_top":[em[i] for i in o1[:10]],"pc1_bot":[em[i] for i in o1[-10:]],
        "pc2_top":[em[i] for i in o2[:10]],"pc2_bot":[em[i] for i in o2[-10:]],
        "loving":{"pc1":float(coords[li,0]),"pc2":float(coords[li,1])}}
    fig,ax=plt.subplots(figsize=(11,9)); ax.scatter(coords[:,0],coords[:,1],s=14,alpha=.55)
    for i,e in enumerate(em):
        if e in LABELS: ax.annotate(e,(coords[i,0],coords[i,1]),fontsize=7,alpha=.85,xytext=(2,2),textcoords="offset points")
    ax.axhline(0,color="grey",lw=.5); ax.axvline(0,color="grey",lw=.5)
    ax.set_xlabel(f"PC1 ({p1:.1f}%) -> valence"); ax.set_ylabel(f"PC2 ({p2:.1f}%) -> arousal")
    ax.set_title("Gemma 2 27B emotion circumplex (171)"); fig.tight_layout()
    fig.savefig(a.out/"circumplex.png",dpi=150); plt.close(fig); print("  circumplex.png")

    from sklearn.cluster import KMeans
    lab=KMeans(n_clusters=a.k,n_init=10,random_state=0).fit(Vl).labels_
    mpc1=[coords[lab==c,0].mean() for c in range(a.k)]; cl=[]
    for r,c in enumerate(np.argsort(mpc1)):
        mem=[em[i] for i in range(n) if lab[i]==c]; cl.append({"rank":r+1,"mean_pc1":float(mpc1[c]),"members":mem})
    S["kmeans"]={"k":a.k,"clusters":cl,"loving_cluster":[em[i] for i in range(n) if lab[i]==lab[li]]}

    C=cosm(Vl)
    from scipy.cluster.hierarchy import linkage,dendrogram
    from scipy.spatial.distance import squareform
    D=1.0-C; np.fill_diagonal(D,0.0); D=(D+D.T)/2; Z=linkage(squareform(D,checks=False),method="average")
    fig,ax=plt.subplots(figsize=(10,22)); dendrogram(Z,labels=em,orientation="right",leaf_font_size=5,ax=ax,color_threshold=.7*D.max())
    ax.set_title("Hierarchical clustering (1 - cosine)"); fig.tight_layout(); fig.savefig(a.out/"dendrogram.png",dpi=150); plt.close(fig); print("  dendrogram.png")
    lo=dendrogram(Z,no_plot=True)["leaves"]; Co=C[np.ix_(lo,lo)]
    fig,ax=plt.subplots(figsize=(11,10)); im=ax.imshow(Co,cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_title("Pairwise cosine (hierarchically ordered)"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im,fraction=.046,pad=.04); fig.tight_layout(); fig.savefig(a.out/"cosine_heatmap.png",dpi=150); plt.close(fig); print("  cosine_heatmap.png")
    nn={}
    for p in PROBES:
        if p in em:
            i=em.index(p); od=np.argsort(-C[i]); nn[p]=[(em[j],float(C[i,j])) for j in od if j!=i][:5]
    S["neighbours"]=nn

    samp=sorted(set(np.linspace(0,L-1,min(14,L)).round().astype(int).tolist()))
    iu=np.triu_indices(n,1); tris=np.stack([cosm(V[:,l,:])[iu] for l in samp]); M=np.corrcoef(tris)
    ml=[k for k,l in enumerate(samp) if l>=L//2]; sub=M[np.ix_(ml,ml)]
    mlm=(sub.sum()-len(ml))/(len(ml)**2-len(ml)); ov=(M.sum()-len(samp))/(len(samp)**2-len(samp))
    print(f"layer stability all={ov:.3f} mid-late={mlm:.3f}")
    fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(M,cmap="viridis",vmin=0,vmax=1)
    ax.set_xticks(range(len(samp))); ax.set_xticklabels(samp,fontsize=7); ax.set_yticks(range(len(samp))); ax.set_yticklabels(samp,fontsize=7)
    ax.set_title("Cross-layer cosine-structure stability"); fig.colorbar(im,fraction=.046,pad=.04); fig.tight_layout()
    fig.savefig(a.out/"layer_stability.png",dpi=150); plt.close(fig); print("  layer_stability.png")
    S["layer_stability"]={"layers":samp,"all":float(ov),"mid_late":float(mlm)}

    if a.ratings:
        import csv; rt={}
        for row in csv.DictReader(open(a.ratings)): rt[row["emotion"].strip()]=(float(row["valence"]),float(row["arousal"]))
        idx=[i for i,e in enumerate(em) if e in rt]
        if len(idx)>=3:
            val=np.array([rt[em[i]][0] for i in idx]); aro=np.array([rt[em[i]][1] for i in idx])
            r1=abs(np.corrcoef(coords[idx,0],val)[0,1]); r2=abs(np.corrcoef(coords[idx,1],aro)[0,1])
            print(f"PC1<->valence |r|={r1:.2f} (paper .81) | PC2<->arousal |r|={r2:.2f} (paper .66) n={len(idx)}")
            S["human_correlation"]={"n":len(idx),"r_pc1_valence":float(r1),"r_pc2_arousal":float(r2)}

    (a.out/"geometry_summary.json").write_text(json.dumps(S,indent=2))
    print(f"wrote {a.out/'geometry_summary.json'} + 4 figures")

if __name__=="__main__":
    main()
