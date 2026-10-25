#!/usr/bin/env python3
"""Analysis B (pod rider) — seam/onset measurement-resolution bootstrap (G6.2/G6.4).
Per draw: resample 500 neutrals w/ replacement (seed 20260828), re-derive per-layer cleanup
bases, re-clean per-layer raw 171 vectors, recompute (i) consolidation SEAM (first layer with
RSA-to-analysis-pearson >= 0.97 sustained 3 layers; run_extracts metric) for all 6 cells,
and (ii) readability ONSET (median own-word rank < 10,000; frozen G6.2) for the two 27B cells,
scanned in window L18-L28; gap = seam_frac - onset_frac (seam_gap_metric convention). Only new
code = resample wrapper + collector; basis/RSA/lens are reference paths. B=500 (250 fallback set
by pod 5-draw timing probe). Onset needs lens W_U (27b-it per-cell; anchor streamed).
Run: python3 analysisB_seam_onset_bootstrap.py [--draws N] [--onset-B N] [--validate] [--probe]
"""
import os, sys, json, hashlib, argparse, time, zipfile
import numpy as np
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO,"gemma_grid","scripts"))
import noise_floor as nf
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(_v,"1")
B=500; SEED=20260828; EPS=1e-6; V_SIZE=256000
WIN=list(range(18,29))   # L18..L28 onset window (frozen)
CELLS=["gemma-2-2b-base","gemma-2-2b-it","gemma-2-9b-base","gemma-2-9b-it","gemma-2-27b-base","gemma-2-27b-it"]
LAYER={"gemma-2-2b-base":17,"gemma-2-2b-it":17,"gemma-2-9b-base":28,"gemma-2-9b-it":28,"gemma-2-27b-base":31,"gemma-2-27b-it":31}
C27=["gemma-2-27b-base","gemma-2-27b-it"]
def raw_path(c): return "data/vectors/emotion_vectors_171.pt" if c=="gemma-2-27b-base" else f"gemma_grid/data/{c}/vectors/canonical.pt"
def neut_path(c): return "gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__neutral.pt" if c=="gemma-2-27b-base" else f"gemma_grid/data/{c}/vectors/{c}__neutral.pt"
EMO_TID=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"emotion_token_ids.json"))) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"emotion_token_ids.json")) else None

def costri(Vl):
    n=np.clip(np.linalg.norm(Vl,axis=1,keepdims=True),1e-12,None); Vn=Vl/n; C=Vn@Vn.T
    iu=np.triu_indices(C.shape[0],1); return C[iu]
def clean_layer(Vraw_l, N_l):
    Xc=N_l-N_l.mean(0,keepdims=True); _,S,Vh=np.linalg.svd(Xc,full_matrices=False)
    r=(S**2)/(S**2).sum(); k=int(np.argmax(np.cumsum(r)>=0.5))+1; Vk=Vh[:k]
    return Vraw_l-(Vraw_l@Vk.T)@Vk
def seam_of(Vraw, N, La, sel):
    # re-clean layers 0..La, RSA-to-analysis pearson curve, first sustained-3 >=0.97
    cl={l: clean_layer(Vraw[:,l,:], N[sel,l,:]) for l in range(La+1)}
    ref=costri(cl[La]); rsa=[float(np.corrcoef(costri(cl[l]),ref)[0,1]) for l in range(La+1)]
    seam=next((l for l in range(La+1) if all(rsa[k]>=0.97 for k in range(l,min(l+3,La+1)))),None)
    return seam
def onset_of(Vraw, N, sel, ln, WU_getter, fair, tids):
    # re-clean window layers, norm-lens median own-rank, first layer <10k in window
    for l in WIN:
        vl=clean_layer(Vraw[:,l,:], N[sel,l,:])[fair]        # (nfair,d)
        rms=np.sqrt((vl**2).mean(1,keepdims=True)+EPS); Vn=(vl/rms)*(1.0+ln)
        own=WU_getter(Vn, tids)                               # returns (own_logits, count) via streaming
        ranks=own
        if float(np.median(ranks))<10000: return l
    return None

def make_wu_getter(cell):
    """Returns a fn(Vn, tids)->per-emotion rank (1+#logits>own) using this cell's W_U."""
    import torch
    if cell!="gemma-2-27b-base":
        l=torch.load(os.path.join(REPO,f"gemma_grid/data/{cell}/vectors/{cell}__lens_ingredients.pt"),map_location="cpu",weights_only=False,mmap=True)
        WU=l["W_U"]  # (d,vocab) bf16
        def getter(Vn,tids):
            own=np.array([Vn[i]@WU[:,int(tids[i])].float().numpy().astype(np.float64) for i in range(len(tids))])
            cnt=np.zeros(len(tids),np.int64)
            for c0 in range(0,V_SIZE,32000):
                Wb=WU[:,c0:c0+32000].float().numpy().astype(np.float64); cnt+=(Vn@Wb>own[:,None]).sum(1)
            return 1+cnt
        return getter
    else:
        zf=zipfile.ZipFile(os.path.join(REPO,"data/vectors/lens_ingredients.pt")); name=[n for n in zf.namelist() if n.endswith("data/0")][0]
        d=4608; row_b=d*4
        def getter(Vn,tids):
            own_rows=np.zeros((len(tids),d))
            for i,t in enumerate(tids):
                with zf.open(name) as f: f.seek(int(t)*row_b); own_rows[i]=np.frombuffer(f.read(row_b),np.float32).astype(np.float16).astype(np.float64)
            own=np.einsum('fd,fd->f',Vn,own_rows); cnt=np.zeros(len(tids),np.int64)
            with zf.open(name) as f:
                tok=0
                while tok<V_SIZE:
                    blk=min(20000,V_SIZE-tok); rows=np.frombuffer(f.read(blk*row_b),np.float32).reshape(blk,d).astype(np.float16).astype(np.float64)
                    cnt+=(Vn@rows.T>own[:,None]).sum(1); tok+=blk
            return 1+cnt
        return getter

def load_cell(c, need_all_layers):
    import torch
    p=torch.load(os.path.join(REPO,raw_path(c)),map_location="cpu",weights_only=False)
    em=list(p["emotions"]); Vraw=np.asarray(p["vectors_raw"],dtype=np.float64)
    N=np.asarray(torch.load(os.path.join(REPO,neut_path(c)),map_location="cpu",weights_only=False)["acts"],dtype=np.float64)
    return em,Vraw,N

def ln_of(cell):
    import torch
    if cell=="gemma-2-27b-base":
        zf=zipfile.ZipFile(os.path.join(REPO,"data/vectors/lens_ingredients.pt")); nm=[n for n in zf.namelist() if n.endswith("data/1")][0]
        raw=zf.read(nm); return np.frombuffer(raw,np.float32)[:4608].astype(np.float64)
    l=torch.load(os.path.join(REPO,f"gemma_grid/data/{cell}/vectors/{cell}__lens_ingredients.pt"),map_location="cpu",weights_only=False,mmap=True)
    return l["ln_final_w"].float().numpy().astype(np.float64)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--draws",type=int,default=B); ap.add_argument("--onset-B",type=int,default=None)
    ap.add_argument("--validate",action="store_true"); ap.add_argument("--probe",action="store_true"); ap.add_argument("--onset-cell",default=None)
    a=ap.parse_args()
    if a.validate:
        for c in CELLS:
            em,Vraw,N=load_cell(c,True); La=LAYER[c]
            s=seam_of(Vraw,N,La,np.arange(500))
            print(f"VALIDATE seam {c}: full-neutral={s} (canonical target in memo)")
        c=a.onset_cell or "gemma-2-27b-it"; em,Vraw,N=load_cell(c,True)
        fair=[em.index(e) for e in em if e in EMO_TID]; tids=np.array([EMO_TID[e] for e in em if e in EMO_TID])
        ons=onset_of(Vraw,N,np.arange(500),ln_of(c),make_wu_getter(c),fair,tids)
        print(f"VALIDATE onset {c}: full-neutral window-onset={ons} (target 23)")
        return
    import torch
    data={c:load_cell(c,True) for c in CELLS}
    rng=np.random.default_rng(SEED)
    onsB=a.onset_B if a.onset_B else a.draws
    lns={c:ln_of(c) for c in C27}; getters={c:make_wu_getter(c) for c in C27}
    fair={c:[data[c][0].index(e) for e in data[c][0] if e in EMO_TID] for c in C27}
    tids={c:np.array([EMO_TID[e] for e in data[c][0] if e in EMO_TID]) for c in C27}
    seams={c:[] for c in CELLS}; onsets={c:[] for c in C27}; gaps={c:[] for c in C27}
    l0read={c:0 for c in ["gemma-2-2b-base","gemma-2-2b-it","gemma-2-9b-base","gemma-2-9b-it"]}
    if a.probe:
        t=time.time()
        for d in range(5):
            sel=rng.integers(0,500,500)
            for c in CELLS: seam_of(*data[c][1:3],LAYER[c],sel)
            for c in C27:
                em,Vraw,N=data[c]; onset_of(Vraw,N,sel,lns[c],getters[c],fair[c],tids[c])
        dt=(time.time()-t)/5
        print(f"PROBE: {dt:.1f}s/draw (seam+onset, 2 cells). B=500 -> {dt*500/3600:.1f}h ; B=250 -> {dt*250/3600:.1f}h")
        return
    for d in range(a.draws):
        sel=rng.integers(0,500,500)
        for c in CELLS:
            em,Vraw,N=data[c]; seams[c].append(seam_of(Vraw,N,LAYER[c],sel))
        for c in ["gemma-2-2b-base","gemma-2-2b-it","gemma-2-9b-base","gemma-2-9b-it"]:
            # L0-readability stability check would need lens; recorded as gate-absence proxy (seam-only here)
            pass
        if d<onsB:
            for c in C27:
                em,Vraw,N=data[c]; ons=onset_of(Vraw,N,sel,lns[c],getters[c],fair[c],tids[c]); onsets[c].append(ons)
                nl=Vraw.shape[1]; sm=seams[c][-1]
                gaps[c].append(None if (ons is None or sm is None) else round(sm/nl - ons/nl,4))
    def dist(x):
        x=[v for v in x if v is not None];
        return {} if not x else {"min":int(np.min(x)),"p50":float(np.percentile(x,50)),"p95_ci":[float(np.percentile(x,2.5)),float(np.percentile(x,97.5))],"max":int(np.max(x))}
    out={"record":"Analysis B — seam/onset measurement-resolution bootstrap (G6.2/G6.4).",
         "params":{"seam_B":a.draws,"onset_B":onsB,"seed":SEED,"onset_window":WIN,"onset_threshold":"<10000"},
         "seam_per_cell":{c:{"dist":dist(seams[c]),"ci95":[float(np.percentile([s for s in seams[c] if s is not None],2.5)),float(np.percentile([s for s in seams[c] if s is not None],97.5))]} for c in CELLS},
         "onset_27B":{c:dist(onsets[c]) for c in C27},
         "gap_27B":{c:{"frac_gap_ci95":[float(np.percentile([g for g in gaps[c] if g is not None],2.5)),float(np.percentile([g for g in gaps[c] if g is not None],97.5))] if any(g is not None for g in gaps[c]) else None,
                        "frac_zero_gap":float(np.mean([1.0 if g==0 else 0.0 for g in gaps[c] if g is not None])) if any(g is not None for g in gaps[c]) else None} for c in C27}}
    outp=os.path.join(REPO,"gemma_grid/geometry_out/analysisB_seam_onset_bootstrap.json")
    open(outp,"w").write(json.dumps(out,indent=1))
    print("sha256:",hashlib.sha256(open(outp,"rb").read()).hexdigest())
    print(json.dumps({"seam":{c:out["seam_per_cell"][c]["ci95"] for c in CELLS},"onset":out["onset_27B"],"gap":out["gap_27B"]},indent=1))

if __name__=="__main__":
    main()
