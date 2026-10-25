#!/usr/bin/env python3
"""Item 5 SVD-fallback PROVENANCE re-run (closes the LC-59 gesdd->gesvd count gap).

The battery shim swallowed the gesdd->gesvd fallback without counting. This re-runs Item 5
ONLY, with a COUNTING shim that tallies and logs every fallback (the analogue of Analysis B's
'5 of 500'), then asserts the recomputed Item 5 metrics are BIT-IDENTICAL to the shipped
pmap_battery.json. Identical metrics => this run's fallback count is authoritative for the
shipped artifact. Frozen estimator byte-stable; convergent SVDs run gesdd untouched.

Governance read:
  fallbacks == 0  -> frozen estimator behavior never altered; NO Ledger freeze-amendment needed.
  fallbacks  > 0  -> null-space caveat applies to exactly those (pair,k,fold) cases; amendment recorded.
"""
import os, sys, json, hashlib
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"   # identical single-thread BLAS regime to the shipped run
import numpy as np

# ---- counting gesdd->gesvd shim (SVD driver only) ----
_ORIG=np.linalg.svd
STATE={"task":None}
EVENTS=[]; N_TOTAL={"svd":0}
def _svd_count(a, full_matrices=True, compute_uv=True, hermitian=False):
    N_TOTAL["svd"]+=1
    try:
        return _ORIG(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as _sp
        EVENTS.append({"task":STATE["task"],"shape":list(np.shape(a))})
        if not compute_uv:
            return _sp(a, full_matrices=full_matrices, compute_uv=False, lapack_driver="gesvd")
        return _sp(a, full_matrices=full_matrices, lapack_driver="gesvd")
np.linalg.svd=_svd_count

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
REPO=os.path.dirname(os.path.dirname(HERE))
import pmap_transport as P
import pmap_battery as M
M.load_all()
FOLD=M.FOLD_IDS
KGRID=[2,5,10,20,40,80,136]
PAIRS=[("gemma-2-2b-base","gemma-2-27b-base"),
       ("gemma-2-27b-base","gemma-2-27b-it"),
       ("gemma-2-27b-it","gemma-2-27b-base")]
shipped=json.load(open(os.path.join(REPO,"gemma_grid/geometry_out/pmap_battery.json")))["item5_pc_truncation"]["pairs"]

def metrics(r,c):
    return {"median_rank":float(np.median(r)),"top1_acc":float(np.mean(r==1)),
            "top5_acc":float(np.mean(r<=5)),"median_cosine":float(np.median(c))}

allmatch=True; per_task_fb={}
for src,tgt in PAIRS:
    key=f"{src}->{tgt}"
    Xs=M.cleaned_analysis(src); Yt=M.cleaned_analysis(tgt)
    Xc=Xs-Xs.mean(0,keepdims=True)
    STATE["task"]=f"{key}|prep"
    _,_,Vt=np.linalg.svd(Xc,full_matrices=False)   # source PCA (full-rank; should not fall back)
    for k in KGRID:
        STATE["task"]=f"{key}|k={k}"
        before=len(EVENTS)
        Pk=Vt[:k]; Xk=(Xc@Pk.T)@Pk
        r,c=P.eval_transport(P.center(Xk),P.center(Yt),FOLD,"procrustes")
        got=metrics(r,c)
        exp=next(row for row in shipped[key] if row["k"]==k)
        exp={m:exp[m] for m in ("median_rank","top1_acc","top5_acc","median_cosine")}
        ok=all(got[m]==exp[m] for m in exp)
        allmatch=allmatch and ok
        per_task_fb[f"{key}|k={k}"]=len(EVENTS)-before
        print(f"  {key} k={k:3d}  fb_this_task={len(EVENTS)-before}  bit_identical={ok}  median_rank={got['median_rank']}",flush=True)

out={"record":"Item 5 gesdd->gesvd fallback provenance (LC-59 count gap closure).",
     "regime":"single-thread BLAS (OMP/OPENBLAS/MKL/VECLIB/NUMEXPR=1), numpy %s scipy fallback"%np.__version__,
     "total_svd_calls_item5":N_TOTAL["svd"],
     "total_fallbacks":len(EVENTS),
     "fallbacks_by_task":{k:v for k,v in per_task_fb.items() if v>0},
     "fallback_events":EVENTS,
     "all_metrics_bit_identical_to_shipped":allmatch,
     "governance":("fallbacks==0 -> frozen estimator behavior never altered; NO amendment needed"
                   if len(EVENTS)==0 else
                   "fallbacks>0 -> null-space caveat applies to the listed (task,shape) cases; Ledger amendment recorded"),
     "prep_svd_note":"the source-PCA prep SVD is full-rank (171xd); any fallback there would be separately listed by task '|prep'"}
OUTP=os.path.join(REPO,"gemma_grid/geometry_out/item5_svd_provenance.json")
open(OUTP,"w").write(json.dumps(out,indent=1))
print("\nTOTAL SVD calls (Item5):",N_TOTAL["svd"],"| FALLBACKS:",len(EVENTS),
      "| all bit-identical to shipped:",allmatch)
print("OUT:",OUTP,"sha256:",hashlib.sha256(open(OUTP,"rb").read()).hexdigest())
