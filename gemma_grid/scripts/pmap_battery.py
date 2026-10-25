#!/usr/bin/env python3
"""P-map robustness BATTERY (Items 1-8) — NEW input-prep + collection wrapper.

LC-59 authorized single execution pass. Spec of record:
  PMAP_STRESSTEST_BATTERY_PRESTATEMENT_100T.md (sha 089fec1e4222...d2d3ef76).

Reuses the FROZEN estimator pmap_transport.py UNMODIFIED (import-only):
  center, fit_procrustes, fit_ridge, cosine_rows, eval_transport, and the frozen
  fold_ids = RandomState(0).permutation(arange(171)%5). No permutation floor is
  recomputed anywhere (the with_perm path of the estimator is never called). All
  items report retrieval metrics only: top-1, top-5, median rank, median held-out
  cosine. Item 3 half-sample cleaning uses the reference clean_layer (gesvdfix).

Realization-of-intent notes (for Coordinator verification):
 - Cleaned analysis-layer / per-layer vectors are the STORED canonical `vectors`
   slices (== reference clean_layer to ~1e-6 storage precision, verified), so
   every graded-object-matched row is byte-identical to the graded P-map inputs.
 - Item 8 "the locked k=10 partition" is resolved to the ANCHOR (27b-base) seed-0
   cleaned analysis-layer k=10 partition, per the phase4_stageA pod convention
   ("ari_to_locked = vs ANCHOR partition"); parsed from table12_gemma-2-27b-base.md.
Output: geometry_out/pmap_battery.json (+ manifest written by runner).
Producer does not grade.
"""
import os, sys, json, re, hashlib, time
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
REPO=os.path.dirname(os.path.dirname(HERE))
import pmap_transport as P                              # FROZEN estimator (unmodified)
import analysisB_seam_onset_bootstrap_gesvdfix as B     # reference clean_layer (Item 3 halves)

LAYER={"gemma-2-2b-base":17,"gemma-2-2b-it":17,"gemma-2-9b-base":28,"gemma-2-9b-it":28,
       "gemma-2-27b-base":31,"gemma-2-27b-it":31}
NLAYERS={"gemma-2-2b-base":26,"gemma-2-2b-it":26,"gemma-2-9b-base":42,"gemma-2-9b-it":42,
         "gemma-2-27b-base":46,"gemma-2-27b-it":46}
CELLS=list(LAYER)
def store_path(c): return os.path.join(REPO,"data/vectors/emotion_vectors_171.pt") if c=="gemma-2-27b-base" \
        else os.path.join(REPO,f"gemma_grid/data/{c}/vectors/canonical.pt")
ART=os.path.join(REPO,"gemma_grid/geometry_out/pmap_transport.json")
OUT=os.path.join(REPO,"gemma_grid/geometry_out/pmap_battery.json")

# ---- frozen fold assignment (verbatim from the estimator) ----
FOLD_IDS=np.random.RandomState(P.FOLD_SEED).permutation(np.arange(171)%P.N_FOLDS)

def metrics(ranks,coss):
    return {"median_rank":float(np.median(ranks)),"top1_acc":float(np.mean(ranks==1)),
            "top5_acc":float(np.mean(ranks<=5)),"median_cosine":float(np.median(coss))}

def transport(Xs,Yt,kind="procrustes",fold_ids=FOLD_IDS):
    """retrieval-only via frozen eval_transport (centered inputs)."""
    r,c=P.eval_transport(P.center(Xs),P.center(Yt),fold_ids,kind)
    return metrics(r,c)

# ---------------- shared load: stored canonical cleaned + raw, all layers ----------------
import torch
STORE={}; EMOS=None
def load_all():
    global EMOS
    for c in CELLS:
        b=torch.load(store_path(c),map_location="cpu",weights_only=False)
        em=[str(e) for e in b["emotions"]]
        if EMOS is None: EMOS=em
        assert em==EMOS, f"emotion order mismatch {c}"
        STORE[c]={"clean":np.asarray(b["vectors"],dtype=np.float64),      # (171,nl,d) canonical cleaned
                  "raw":np.asarray(b["vectors_raw"],dtype=np.float64)}     # (171,nl,d) raw
def cleaned_at(c,l): return STORE[c]["clean"][:,l,:]
def raw_at(c,l):     return STORE[c]["raw"][:,l,:]
def cleaned_analysis(c): return cleaned_at(c,LAYER[c])
def raw_analysis(c):     return raw_at(c,LAYER[c])

# master ordered-pair list helpers -------------------------------------------------
def artifact_pairs():
    return list(json.load(open(ART))["pairs"].keys())   # the 16 computed keys

# =================================================================================
# ITEM 1 — layer-wise transport curve (frac-mapped + pinned analysis-layer row)
# =================================================================================
def item1():
    def frac_target(ls,ns,nt): return int(round((ls/ns)*nt))
    def curve(src,tgt):
        ns,nt=NLAYERS[src],NLAYERS[tgt]; rows=[]
        for ls in range(ns):
            lt=frac_target(ls,ns,nt); lt=min(lt,nt-1)
            m=transport(cleaned_at(src,ls),cleaned_at(tgt,lt))
            rows.append({"src_layer":ls,"tgt_layer":lt,"frac":round(ls/ns,4),**m})
        return rows
    res={"note":"frac=layer/n_layers, tgt=round(frac*N_t); L0 included; retrieval-only.",
         "pairs":{}}
    for src,tgt in [("gemma-2-2b-base","gemma-2-27b-base"),
                    ("gemma-2-27b-base","gemma-2-27b-it"),
                    ("gemma-2-27b-it","gemma-2-27b-base")]:
        res["pairs"][f"{src}->{tgt}"]=curve(src,tgt)
    # PINNED registered analysis-layer graded config as explicit extra row (2b-L17 <-> 27b-L31)
    res["pinned_analysis_row"]={
        "gemma-2-2b-base->gemma-2-27b-base":{"src_layer":17,"tgt_layer":31,
            **transport(cleaned_at("gemma-2-2b-base",17),cleaned_at("gemma-2-27b-base",31))}}
    # branch read: deep vs L0 for primary
    prim=res["pairs"]["gemma-2-2b-base->gemma-2-27b-base"]
    l0=prim[0]; pin=res["pinned_analysis_row"]["gemma-2-2b-base->gemma-2-27b-base"]
    res["branch_read"]={"L0_median_rank":l0["median_rank"],"analysis_pin_median_rank":pin["median_rank"],
        "interpretation":"deep beats L0 -> transport of contextually constructed geometry; "
                         "L0 comparable -> weaker stimulus-driven-structure sentence ships."}
    return res

# =================================================================================
# ITEM 2 — raw-vector rerun over the 16 artifact pairs (cleanup confound)
# =================================================================================
def item2():
    out={"note":"frozen protocol on RAW analysis-layer vectors over the 16 artifact (cleaned) pairs.",
         "pairs":{}}
    for key in artifact_pairs():
        s,t=key.split("->")
        out["pairs"][key]=transport(raw_analysis(s),raw_analysis(t))
    art=json.load(open(ART))["pairs"]
    out["primary_compare"]={"raw":out["pairs"]["gemma-2-2b-base->gemma-2-27b-base"],
        "cleaned_graded":{k:art["gemma-2-2b-base->gemma-2-27b-base"][k] for k in
                          ("median_rank","top1_acc","top5_acc","median_cosine")}}
    return out

# =================================================================================
# ITEM 3 — split-half ceiling (per-story halves; reference clean_layer)
# required minimum: 2b-base, 27b-base. built-in coverage fallback.
# =================================================================================
def half_matrices(cell):
    """Return (Va_clean, Vb_clean) 171xd at analysis layer from a fixed story split, or None if no coverage."""
    p=os.path.join(REPO,f"gemma_grid/data/{cell}/vectors/{cell}__per_story_slice.pt")
    if not os.path.exists(p): return None
    blob=torch.load(p,map_location="cpu",weights_only=False)
    pe=blob["per_emotion"]; sl=blob.get("slice_layers") or blob.get("layers")
    # locate analysis-layer index within the stored slice layers
    if LAYER[cell] not in list(sl): return None
    li=list(sl).index(LAYER[cell])
    neutral=np.asarray(blob["neutral"],dtype=np.float64)   # (500, nsl, d)
    Nl=neutral[:,li,:]
    rows_a=[]; rows_b=[]
    for e in EMOS:
        if e not in pe: return None
        acts=np.asarray(pe[e],dtype=np.float64)            # (nstories, nsl, d)
        acts=acts[:,li,:]                                  # (nstories,d) at analysis layer
        ns=acts.shape[0]
        if ns<2: return None
        # frozen half assignment: story-index parity after a fixed shuffle, RandomState(0)
        perm=np.random.RandomState(0).permutation(ns)
        ha=perm[0::2]; hb=perm[1::2]
        rows_a.append(acts[ha].mean(0)); rows_b.append(acts[hb].mean(0))
    Va=np.stack(rows_a); Vb=np.stack(rows_b)               # (171,d) raw half-means
    # clean each half with the standard 100T cleanup (reference clean_layer, per-layer neutral)
    Va_c=B.clean_layer(Va,Nl); Vb_c=B.clean_layer(Vb,Nl)
    return Va_c,Vb_c
def item3():
    out={"note":"within-cell split-half transport (measurement ceiling); halves cleaned via reference clean_layer.",
         "cells":{}}
    for cell in CELLS:
        hm=half_matrices(cell)
        if hm is None:
            out["cells"][cell]={"coverage":"insufficient — coverage only"}; continue
        Va,Vb=hm
        out["cells"][cell]={"coverage":"ok","split_half":transport(Va,Vb)}
    req=["gemma-2-2b-base","gemma-2-27b-base"]
    out["required_minimum_ok"]=all(out["cells"][c].get("coverage")=="ok" for c in req)
    out["branch_read"]="ceiling near cross-model .97 -> transport as good as within-model re-measurement; " \
                       "ceiling well above .97 -> real cross-model gap."
    return out

# =================================================================================
# ITEM 4 — VA-plane ablation (own PC1/PC2 at analysis layer projected out)
# =================================================================================
def own_pcs(V,k=2):
    Vc=V-V.mean(0,keepdims=True)
    _,_,Vt=np.linalg.svd(Vc,full_matrices=False)
    return Vt[:k]                                            # (k,d)
def ablate(V,Pk):
    Vc=V-V.mean(0,keepdims=True)
    return Vc-(Vc@Pk.T)@Pk
def item4():
    out={"note":"project each cell's OWN top-2 PCs (analysis layer, cleaned) out of source & target; rerun.",
         "pairs":{}}
    for src,tgt in [("gemma-2-2b-base","gemma-2-27b-base"),
                    ("gemma-2-27b-base","gemma-2-27b-it"),
                    ("gemma-2-27b-it","gemma-2-27b-base")]:
        Xs=cleaned_analysis(src); Yt=cleaned_analysis(tgt)
        Xa=ablate(Xs,own_pcs(Xs)); Ya=ablate(Yt,own_pcs(Yt))
        out["pairs"][f"{src}->{tgt}"]={"ablated":transport(Xa,Ya),
                                       "baseline":transport(Xs,Yt)}
    out["branch_read"]="surviving retrieval -> transport carries structure beyond the circumplex plane; " \
                       "collapsed -> transportable object is principally the VA plane."
    return out

# =================================================================================
# ITEM 5 — PC-truncation curve (fit map on source's top-k PCs)
# =================================================================================
def item5():
    kgrid=[2,5,10,20,40,80,136]
    out={"note":"source reduced to its own top-k PC subspace (reconstructed), retrieval vs k.","k_grid":kgrid,"pairs":{}}
    for src,tgt in [("gemma-2-2b-base","gemma-2-27b-base"),
                    ("gemma-2-27b-base","gemma-2-27b-it"),
                    ("gemma-2-27b-it","gemma-2-27b-base")]:
        Xs=cleaned_analysis(src); Yt=cleaned_analysis(tgt)
        Xc=Xs-Xs.mean(0,keepdims=True)
        _,_,Vt=np.linalg.svd(Xc,full_matrices=False)
        curve=[]
        for k in kgrid:
            Pk=Vt[:k]; Xk=(Xc@Pk.T)@Pk                      # reconstruct source in top-k subspace
            curve.append({"k":k,**transport(Xk,Yt)})
        out["pairs"][f"{src}->{tgt}"]=curve
    return out

# =================================================================================
# ITEM 6 — centering audit (train-only per-fold centering on primary pair)
# =================================================================================
def eval_trainonly_centering(Xs,Yt,fold_ids):
    """Procrustes with per-fold train-only centering (new collector; reuses frozen fit_procrustes)."""
    n=Xs.shape[0]; ranks=np.zeros(n,int); coss=np.zeros(n)
    for f in range(P.N_FOLDS):
        tr,te=fold_ids!=f,fold_ids==f
        mx=Xs[tr].mean(0,keepdims=True); my=Yt[tr].mean(0,keepdims=True)  # train-only means
        Xtr,Ytr=Xs[tr]-mx,Yt[tr]-my
        W,s=P.fit_procrustes(Xtr,Ytr)
        pred=s*((Xs[te]-mx)@W)
        Ycen=Yt-my                                          # center all targets by train mean
        Scos=P.cosine_rows(pred,Ycen)
        idx=np.where(te)[0]
        for row,j in enumerate(idx):
            order=np.argsort(-Scos[row]); ranks[j]=int(np.where(order==j)[0][0])+1; coss[j]=float(Scos[row,j])
    return metrics(ranks,coss)
def item6():
    src,tgt="gemma-2-2b-base","gemma-2-27b-base"
    Xs,Yt=cleaned_analysis(src),cleaned_analysis(tgt)
    glob=transport(Xs,Yt)                                   # frozen global-171 centering
    tro=eval_trainonly_centering(P.center(Xs),P.center(Yt),FOLD_IDS)
    return {"note":"primary pair; global-171 centering (frozen) vs train-only per-fold centering.",
            "global_centering":glob,"train_only_centering":tro,
            "delta_median_rank":tro["median_rank"]-glob["median_rank"],
            "delta_median_cosine":tro["median_cosine"]-glob["median_cosine"],
            "expectation":"negligible; retires the leakage quibble."}

# =================================================================================
# ITEM 7 — up/down asymmetry (14 NEW reverse runs) + Procrustes/ridge parity (a)
# =================================================================================
def item7():
    art=json.load(open(ART))["pairs"]
    # (a) Procrustes vs ridge gap from existing artifact (no new compute) — primary carries ridge
    a={}
    for k,v in art.items():
        if "ridge_secondary" in v:
            a[k]={"procrustes":{m:v[m] for m in ("median_rank","top1_acc","median_cosine")},
                  "ridge":v["ridge_secondary"]}
    # (b) 14 NEW reverse-direction runs (cleaned analysis-layer), completing all 30 ordered pairs
    existing=set(art.keys())
    new_runs={}
    names=CELLS
    for s in names:
        for t in names:
            if s==t: continue
            key=f"{s}->{t}"
            rev=f"{t}->{s}"
            if key in existing: continue
            if rev in existing:   # this is a [NEW] reverse of an existing forward
                new_runs[key]=transport(cleaned_analysis(s),cleaned_analysis(t))
    # directional gaps: pair each forward(existing) with its reverse(new) where both now exist
    all_pairs={**{k:{m:art[k][m] for m in ("median_rank","top1_acc","top5_acc","median_cosine")} for k in art},
               **new_runs}
    gaps={}
    seen=set()
    for key in all_pairs:
        s,t=key.split("->"); rev=f"{t}->{s}"
        if rev in all_pairs and (rev,key) not in seen and (key,rev) not in seen:
            seen.add((key,rev))
            fwd=all_pairs[key]; bwd=all_pairs[rev]
            gaps[f"{key} vs {rev}"]={m:round(fwd[m]-bwd[m],4) for m in ("median_rank","top1_acc","top5_acc","median_cosine")}
    return {"note":"(a) Procrustes/ridge parity from artifact; (b) 14 NEW reverse runs -> 30 ordered; per-pair directional gaps.",
            "a_procrustes_ridge":a,
            "b_new_reverse_runs":new_runs,
            "b_directional_gaps":gaps,
            "c_disclosure":"artifact-only 16-key aggregate is NOT direction-balanced; the matched per-pair gaps in (b) are the asymmetry result."}

# =================================================================================
# ITEM 8 — leave-one-family-out (anchor locked k=10 partition); exploratory, ungraded
# =================================================================================
def parse_anchor_families():
    txt=open(os.path.join(REPO,"gemma_grid/geometry_out/table12_gemma-2-27b-base.md")).read()
    fam={}
    for line in txt.splitlines():
        m=re.match(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*$",line)
        if not m: continue
        rank=int(m.group(1)); members=[x.strip() for x in m.group(3).split(",")]
        for mem in members: fam[mem]=rank
    return np.array([fam[e] for e in EMOS])
def eval_family_out(Xs,Yt,labels):
    """leave-one-family-out: fit on all-but-family, retrieve held-out family among all 171."""
    n=Xs.shape[0]; ranks=np.zeros(n,int); coss=np.zeros(n)
    Xc=P.center(Xs); Yc=P.center(Yt)
    for fam in np.unique(labels):
        tr=labels!=fam; te=labels==fam
        W,s=P.fit_procrustes(Xc[tr],Yc[tr]); pred=s*(Xc[te]@W)
        Scos=P.cosine_rows(pred,Yc)
        idx=np.where(te)[0]
        for row,j in enumerate(idx):
            order=np.argsort(-Scos[row]); ranks[j]=int(np.where(order==j)[0][0])+1; coss[j]=float(Scos[row,j])
    return ranks,coss,{int(f):int((labels==f).sum()) for f in np.unique(labels)}
def item8():
    labels=parse_anchor_families()
    src,tgt="gemma-2-2b-base","gemma-2-27b-base"
    ranks,coss,sizes=eval_family_out(cleaned_analysis(src),cleaned_analysis(tgt),labels)
    per_family={}
    for f in np.unique(labels):
        te=labels==f
        per_family[int(f)]={"n":int(te.sum()),"median_rank":float(np.median(ranks[te])),
                            "top1_acc":float(np.mean(ranks[te]==1)),"median_cosine":float(np.median(coss[te]))}
    return {"note":"leave-one-family-out on ANCHOR locked k=10 partition (extrapolation). EXPLORATORY, UNGRADED "
                   "(runs under the frozen P-map spec's own registration, not this freeze).",
            "partition_source":"table12_gemma-2-27b-base.md (anchor seed-0 cleaned L31 k=10)",
            "pair":f"{src}->{tgt}","family_sizes":sizes,
            "pooled":metrics(ranks,coss),"per_family":per_family}

def main():
    t0=time.time(); load_all()
    results={"record":"P-map robustness battery Items 1-8 (retrieval-only; frozen estimator reused unmodified; "
                      "NO permutation floor recomputed).",
             "spec_sha256_battery":"089fec1e4222bc91582fdc64c40430c5fef7a9e90894d44faeebd35bd2d3ef76",
             "estimator_reuse":{"module":"pmap_transport.py","fold_seed":P.FOLD_SEED,"n_folds":P.N_FOLDS,
                                "floor_recomputed":False},
             "emotion_order_n":len(EMOS)}
    for name,fn in [("item1_layerwise_transport",item1),("item2_raw_rerun",item2),
                    ("item3_splithalf_ceiling",item3),("item4_va_plane_ablation",item4),
                    ("item5_pc_truncation",item5),("item6_centering_audit",item6),
                    ("item7_asymmetry_parity",item7),("item8_family_out",item8)]:
        ts=time.time(); results[name]=fn(); print(f"[{name}] done in {time.time()-ts:.1f}s",flush=True)
    results["wall_seconds"]=round(time.time()-t0,1)
    open(OUT,"w").write(json.dumps(results,indent=1))
    print("OUT:",OUT); print("sha256:",hashlib.sha256(open(OUT,"rb").read()).hexdigest())

if __name__=="__main__":
    main()
