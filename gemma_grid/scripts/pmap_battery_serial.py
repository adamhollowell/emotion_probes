#!/usr/bin/env python3
"""P-map battery (Items 1-8) — SERIAL, single-process, CHECKPOINTED + RESUMABLE.

Deliberately slow and bulletproof (Adam's directive after two failed runs): one transport at a
time, ~5.6 GB peak, no multiprocessing, no swap risk. Every completed task is written to a JSONL
checkpoint immediately (flush+fsync); on restart, finished tasks are skipped, so an interruption
(crash, laptop sleep, Ctrl-C) never redoes work and the battery only ever has to finish once.

FROZEN estimator pmap_transport.py is byte-stable and reused unmodified. The ONLY authorized
deviation is the Coordinator-approved gesdd->gesvd SVD fallback (= Analysis B remedy-1b): a
convergent decomposition runs gesdd UNTOUCHED (bit-identical); only a gesdd LinAlgError falls back
to scipy gesvd. SVD DRIVER ONLY, no estimator edit. No permutation floor recomputed anywhere.

Output: geometry_out/pmap_battery.json  (assembled from the checkpoint when all done).
Producer does not grade.
"""
import os, sys, json, time, hashlib
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
import numpy as np

# ---- Coordinator-authorized gesdd->gesvd fallback (SVD driver only; convergent cases bit-identical) ----
_ORIG_SVD=np.linalg.svd
def _svd_fallback(a, full_matrices=True, compute_uv=True, hermitian=False):
    try:
        return _ORIG_SVD(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as _sp
        if not compute_uv:
            return _sp(a, full_matrices=full_matrices, compute_uv=False, lapack_driver="gesvd")
        return _sp(a, full_matrices=full_matrices, lapack_driver="gesvd")
np.linalg.svd=_svd_fallback

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
REPO=os.path.dirname(os.path.dirname(HERE))
import pmap_transport as P
import pmap_battery as M   # validated prep helpers (load_all, cleaned_at, ablate, own_pcs, half_matrices, ...)

OUT=os.path.join(REPO,"gemma_grid/geometry_out/pmap_battery.json")
CKPT=os.path.join(HERE,"pmap_battery_ckpt.jsonl")
I3SKIP=set()   # release: models whose per-story slice ships without its neutral rows (item 3 skipped)
FOLD_IDS=M.FOLD_IDS
ITEM1_PAIRS=[("gemma-2-2b-base","gemma-2-27b-base"),
             ("gemma-2-27b-base","gemma-2-27b-it"),
             ("gemma-2-27b-it","gemma-2-27b-base")]

def _metrics(r,c):
    return {"median_rank":float(np.median(r)),"top1_acc":float(np.mean(r==1)),
            "top5_acc":float(np.mean(r<=5)),"median_cosine":float(np.median(c))}
def transport(Xs,Yt):
    r,c=P.eval_transport(P.center(Xs),P.center(Yt),FOLD_IDS,"procrustes"); return _metrics(r,c)
def frac_target(ls,ns,nt): return min(int(round((ls/ns)*nt)),nt-1)

# ---------------- checkpoint ----------------
def load_ckpt():
    done={}
    if os.path.exists(CKPT):
        for line in open(CKPT):
            line=line.strip()
            if not line: continue
            o=json.loads(line); done[tuple(o["k"])]=o["m"]
    return done
def append_ckpt(tid,val):
    with open(CKPT,"a") as f:
        f.write(json.dumps({"k":list(tid),"m":val})+"\n"); f.flush(); os.fsync(f.fileno())

# ---------------- task generator (lazy; arrays sliced on demand, one live at a time) ----------------
def plain_tasks(I3HALVES):
    # Item 1
    for src,tgt in ITEM1_PAIRS:
        ns,nt=M.NLAYERS[src],M.NLAYERS[tgt]; key=f"{src}->{tgt}"
        for ls in range(ns):
            lt=frac_target(ls,ns,nt)
            yield ("i1",key,ls),(lambda s=src,t=tgt,a=ls,b=lt:(M.cleaned_at(s,a),M.cleaned_at(t,b)))
    yield ("i1pin",),(lambda:(M.cleaned_at("gemma-2-2b-base",17),M.cleaned_at("gemma-2-27b-base",31)))
    # Item 2
    for k in M.artifact_pairs():
        s,t=k.split("->"); yield ("i2",k),(lambda s=s,t=t:(M.raw_analysis(s),M.raw_analysis(t)))
    # Item 3 (halves precomputed)
    for cell in M.CELLS:
        if cell in I3HALVES:
            yield ("i3",cell),(lambda c=cell:(I3HALVES[c][0],I3HALVES[c][1]))
    # Item 4
    for src,tgt in ITEM1_PAIRS:
        key=f"{src}->{tgt}"
        yield ("i4abl",key),(lambda s=src,t=tgt:(M.ablate(M.cleaned_analysis(s),M.own_pcs(M.cleaned_analysis(s))),
                                                  M.ablate(M.cleaned_analysis(t),M.own_pcs(M.cleaned_analysis(t)))))
        yield ("i4base",key),(lambda s=src,t=tgt:(M.cleaned_analysis(s),M.cleaned_analysis(t)))
    # Item 5
    for src,tgt in ITEM1_PAIRS:
        key=f"{src}->{tgt}"
        for kk in [2,5,10,20,40,80,136]:
            def mk(s=src,t=tgt,k=kk):
                Xs=M.cleaned_analysis(s); Yt=M.cleaned_analysis(t)
                Xc=Xs-Xs.mean(0,keepdims=True); _,_,Vt=np.linalg.svd(Xc,full_matrices=False)
                Pk=Vt[:k]; return (Xc@Pk.T)@Pk, Yt
            yield ("i5",key,kk),mk
    # Item 7b: 14 NEW reverse runs
    art=set(M.artifact_pairs())
    for s in M.CELLS:
        for t in M.CELLS:
            if s==t: continue
            key=f"{s}->{t}"; rev=f"{t}->{s}"
            if key in art: continue
            if rev in art:
                yield ("i7",key),(lambda s=s,t=t:(M.cleaned_analysis(s),M.cleaned_analysis(t)))

def item6_val():
    src,tgt="gemma-2-2b-base","gemma-2-27b-base"
    Xs=M.cleaned_analysis(src); Yt=M.cleaned_analysis(tgt)
    glob=transport(Xs,Yt)
    tro=M.eval_trainonly_centering(Xs,Yt,FOLD_IDS)
    return {"note":"primary pair; global-171 centering (frozen) vs train-only per-fold centering (from cleaned vectors).",
            "global_centering":glob,"train_only_centering":tro,
            "delta_median_rank":tro["median_rank"]-glob["median_rank"],
            "delta_median_cosine":tro["median_cosine"]-glob["median_cosine"],
            "expectation":"negligible (cleaned vectors already per-cell centered); retires the leakage quibble."}
def item8_val():
    labels=M.parse_anchor_families(); src,tgt="gemma-2-2b-base","gemma-2-27b-base"
    ranks,coss,sizes=M.eval_family_out(M.cleaned_analysis(src),M.cleaned_analysis(tgt),labels)
    per_family={}
    for f in np.unique(labels):
        te=labels==f; per_family[int(f)]={"n":int(te.sum()),"median_rank":float(np.median(ranks[te])),
            "top1_acc":float(np.mean(ranks[te]==1)),"median_cosine":float(np.median(coss[te]))}
    return {"note":"leave-one-family-out on ANCHOR locked k=10 partition (extrapolation). EXPLORATORY, UNGRADED "
                   "(runs under the frozen P-map spec's own registration, not this freeze).",
            "partition_source":"table12_gemma-2-27b-base.md (anchor seed-0 cleaned L31 k=10)","pair":f"{src}->{tgt}",
            "family_sizes":sizes,"pooled":_metrics(ranks,coss),"per_family":per_family}

def assemble(cache,I3set):
    art=json.load(open(M.ART))["pairs"]
    R={"record":"P-map robustness battery Items 1-8 (retrieval-only; frozen estimator reused unmodified; "
               "NO permutation floor recomputed; gesdd->gesvd fallback = Coordinator remedy-1b; serial checkpointed runner).",
       "spec_sha256_battery":"089fec1e4222bc91582fdc64c40430c5fef7a9e90894d44faeebd35bd2d3ef76",
       "estimator_reuse":{"module":"pmap_transport.py","fold_seed":P.FOLD_SEED,"n_folds":P.N_FOLDS,
                          "floor_recomputed":False,"svd_fallback":"gesdd->gesvd on non-convergence (Coordinator remedy-1b); convergent SVDs bit-identical"},
       "emotion_order_n":len(M.EMOS)}
    i1={"note":"frac=layer/n_layers, tgt=round(frac*N_t); L0 included; retrieval-only.","pairs":{}}
    for src,tgt in ITEM1_PAIRS:
        key=f"{src}->{tgt}"; rows=[]
        for ls in range(M.NLAYERS[src]):
            lt=frac_target(ls,M.NLAYERS[src],M.NLAYERS[tgt])
            rows.append({"src_layer":ls,"tgt_layer":lt,"frac":round(ls/M.NLAYERS[src],4),**cache[("i1",key,ls)]})
        i1["pairs"][key]=rows
    i1["pinned_analysis_row"]={"gemma-2-2b-base->gemma-2-27b-base":{"src_layer":17,"tgt_layer":31,**cache[("i1pin",)]}}
    p=i1["pairs"]["gemma-2-2b-base->gemma-2-27b-base"]
    i1["branch_read"]={"L0_median_rank":p[0]["median_rank"],
        "analysis_pin_median_rank":i1["pinned_analysis_row"]["gemma-2-2b-base->gemma-2-27b-base"]["median_rank"],
        "interpretation":"deep beats L0 -> transport of contextually constructed geometry; L0 comparable -> weaker stimulus-driven-structure sentence ships."}
    R["item1_layerwise_transport"]=i1
    i2={"note":"frozen protocol on RAW analysis-layer vectors over the 16 artifact (cleaned) pairs.","pairs":{}}
    for k in M.artifact_pairs(): i2["pairs"][k]=cache[("i2",k)]
    i2["primary_compare"]={"raw":i2["pairs"]["gemma-2-2b-base->gemma-2-27b-base"],
        "cleaned_graded":{m:art["gemma-2-2b-base->gemma-2-27b-base"][m] for m in ("median_rank","top1_acc","top5_acc","median_cosine")}}
    R["item2_raw_rerun"]=i2
    i3={"note":"within-cell split-half transport (measurement ceiling); halves cleaned via reference clean_layer.","cells":{}}
    for cell in M.CELLS:
        i3["cells"][cell]={"coverage":"ok","split_half":cache[("i3",cell)]} if cell in I3set else ({"coverage":"skipped: slice has no neutral key (item 3 is superseded by item3_splithalf_corrected.json)"} if cell in I3SKIP else {"coverage":"insufficient — coverage only"})
    i3["required_minimum_ok"]=(all(c in I3set for c in ["gemma-2-2b-base","gemma-2-27b-base"])
                               if not I3SKIP else "skipped: slice has no neutral key (item 3 is superseded by item3_splithalf_corrected.json)")
    i3["branch_read"]="ceiling near cross-model .97 -> transport as good as within-model re-measurement; ceiling well above .97 -> real cross-model gap."
    R["item3_splithalf_ceiling"]=i3
    i4={"note":"project each cell's OWN top-2 PCs (analysis layer, cleaned) out of source & target; rerun.","pairs":{}}
    for src,tgt in ITEM1_PAIRS:
        key=f"{src}->{tgt}"; i4["pairs"][key]={"ablated":cache[("i4abl",key)],"baseline":cache[("i4base",key)]}
    i4["branch_read"]="surviving retrieval -> transport carries structure beyond the circumplex plane; collapsed -> transportable object is principally the VA plane."
    R["item4_va_plane_ablation"]=i4
    i5={"note":"source reduced to its own top-k PC subspace (reconstructed), retrieval vs k.","k_grid":[2,5,10,20,40,80,136],"pairs":{}}
    for src,tgt in ITEM1_PAIRS:
        key=f"{src}->{tgt}"; i5["pairs"][key]=[{"k":kk,**cache[("i5",key,kk)]} for kk in i5["k_grid"]]
    R["item5_pc_truncation"]=i5
    R["item6_centering_audit"]=cache[("i6",)]
    a={}
    for k,v in art.items():
        if "ridge_secondary" in v:
            a[k]={"procrustes":{m:v[m] for m in ("median_rank","top1_acc","median_cosine")},"ridge":v["ridge_secondary"]}
    new_runs={key[1]:cache[key] for key in cache if isinstance(key,tuple) and key[0]=="i7"}
    all_pairs={**{k:{m:art[k][m] for m in ("median_rank","top1_acc","top5_acc","median_cosine")} for k in art},**new_runs}
    gaps={}; seen=set()
    for key in all_pairs:
        s,t=key.split("->"); rev=f"{t}->{s}"
        if rev in all_pairs and (key,rev) not in seen and (rev,key) not in seen:
            seen.add((key,rev)); fwd=all_pairs[key]; bwd=all_pairs[rev]
            gaps[f"{key} vs {rev}"]={m:round(fwd[m]-bwd[m],4) for m in ("median_rank","top1_acc","top5_acc","median_cosine")}
    R["item7_asymmetry_parity"]={"note":"(a) Procrustes/ridge parity from artifact; (b) 14 NEW reverse runs -> 30 ordered; per-pair directional gaps.",
        "a_procrustes_ridge":a,"b_new_reverse_runs":new_runs,"b_directional_gaps":gaps,
        "c_disclosure":"artifact-only 16-key aggregate is NOT direction-balanced; the matched per-pair gaps in (b) are the asymmetry result."}
    R["item8_family_out"]=cache[("i8",)]
    return R

def main():
    t0=time.time(); M.load_all(); print(f"[load] {time.time()-t0:.1f}s",flush=True)
    cache=load_ckpt(); print(f"[resume] {len(cache)} tasks already checkpointed",flush=True)
    # Item 3 halves — eager, only for cells not already done (avoids re-loading 3.8GB per-story files on resume)
    I3set=set(); I3HALVES={}
    for cell in M.CELLS:
        if ("i3",cell) in cache: I3set.add(cell); continue
        if ("i3none",cell) in cache: continue
        try: hm=M.half_matrices(cell)
        except KeyError as _e:   # release: the per-story slices ship without their neutral rows
            if _e.args!=("neutral",): raise
            I3SKIP.add(cell); print(f"[item3 prep] {cell}: skipped: slice has no neutral key",flush=True); continue
        if hm is None: append_ckpt(("i3none",cell),True)
        else: I3HALVES[cell]=hm; I3set.add(cell)
        print(f"[item3 prep] {cell}: {'ok' if hm else 'insufficient'}",flush=True)
    # main serial task loop (checkpoint each)
    tp=time.time(); n=0
    for tid,thunk in plain_tasks(I3HALVES):
        if tid in cache: continue
        Xs,Yt=thunk(); cache[tid]=transport(Xs,Yt); append_ckpt(tid,cache[tid]); n+=1
        print(f"  done {tid} | median_rank={cache[tid]['median_rank']} | {n} new, {time.time()-tp:.0f}s elapsed",flush=True)
    # Item 6 / Item 8 (parent-computed, checkpointed)
    if ("i6",) not in cache: cache[("i6",)]=item6_val(); append_ckpt(("i6",),cache[("i6",)]); print("  done item6",flush=True)
    if ("i8",) not in cache: cache[("i8",)]=item8_val(); append_ckpt(("i8",),cache[("i8",)]); print("  done item8",flush=True)
    R=assemble(cache,I3set); R["wall_seconds_final_assembly"]=round(time.time()-t0,1)
    open(OUT,"w").write(json.dumps(R,indent=1))
    print("OUT:",OUT); print("sha256:",hashlib.sha256(open(OUT,"rb").read()).hexdigest(),flush=True)
    print("ALL ITEMS COMPLETE.",flush=True)

if __name__=="__main__":
    main()
