#!/usr/bin/env python3
"""P-map frozen floor — PATH A many-core node harness (LC-50 / PATH_A dispatch).

Scheduler ONLY: imports the UNMODIFIED frozen functions from pmap_transport.py
and calls eval_transport per permutation. No estimator change, no thin-SVD, float64.
Parallelism is scheduling-only -> bit-identical to serial on THIS host.

RUN ON A LINUX MANY-CORE HOST (OpenBLAS/MKL that respects *_NUM_THREADS):
    python3 pmap_floor_pathA_node.py --workers $(nproc)
BLAS is pinned single-thread PER PROCESS at the very top (before numpy import) so
N workers give true Nx. macOS/Accelerate is NOT supported (ignores the caps).

Layout expected on the node (scp these next to this script / at repo root):
    pmap_transport.py                                  (frozen, ee736f5d)
    gemma_grid/data/gemma-2-2b-base/vectors/canonical.pt
    data/vectors/emotion_vectors_171.pt                (anchor, 6de2e293, READ-ONLY)
Output: gemma_grid/geometry_out/pmap_transport_floor.json  (companion; 10c2836f untouched)
"""
import os
# ---- pin single-thread BLAS BEFORE numpy import (critical for fork pools) ----
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v]="1"
import argparse, json, time, hashlib, sys, platform
import numpy as np
from multiprocessing import Pool

HERE=os.path.dirname(os.path.abspath(__file__))
# frozen script may sit next to this harness or under gemma_grid/scripts/
for cand in (HERE, os.path.join(HERE,"gemma_grid","scripts"), "gemma_grid/scripts", "."):
    if os.path.exists(os.path.join(cand,"pmap_transport.py")):
        sys.path.insert(0,cand); break
import pmap_transport as P   # FROZEN, unmodified

ANALYSIS={"gemma-2-2b-base":17,"gemma-2-27b-base":31}
SRC_PATH="gemma_grid/data/gemma-2-2b-base/vectors/canonical.pt"
TGT_PATH="data/vectors/emotion_vectors_171.pt"   # anchor, read-only
ANCHOR_SHA="6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
OUT="gemma_grid/geometry_out/pmap_transport_floor.json"
CKPT=os.path.join(HERE,"floor_top1_node.npy")

def load_slice(path, layer):
    import torch
    blob=torch.load(path, map_location="cpu", weights_only=False)
    V=np.asarray(blob["vectors"], dtype=np.float64)   # (171, L, d)
    assert V.ndim==3 and V.shape[0]==171, f"unexpected shape {V.shape} at {path}"
    return V[:, layer, :]                              # (171, d) analysis-layer

G={}
def init_worker():
    Xs=load_slice(SRC_PATH, ANALYSIS["gemma-2-2b-base"])
    Yt=load_slice(TGT_PATH, ANALYSIS["gemma-2-27b-base"])
    G["Xs"]=P.center(Xs); G["Yt"]=P.center(Yt)
    G["fold_ids"]=np.random.RandomState(P.FOLD_SEED).permutation(np.arange(171)%P.N_FOLDS)
def eval_perm(arg):
    i,perm=arg
    pr,_=P.eval_transport(G["Xs"], G["Yt"][perm], G["fold_ids"], "procrustes")
    return i, float(np.mean(pr==1))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=os.cpu_count())
    a=ap.parse_args(); t0=time.time()
    # anchor integrity
    h=hashlib.sha256(open(TGT_PATH,"rb").read()).hexdigest()
    assert h==ANCHOR_SHA, f"ANCHOR SHA MISMATCH {h} != {ANCHOR_SHA}"
    # obs gate (serial, main proc)
    Xs=P.center(load_slice(SRC_PATH,ANALYSIS["gemma-2-2b-base"]))
    Yt=P.center(load_slice(TGT_PATH,ANALYSIS["gemma-2-27b-base"]))
    fold_ids=np.random.RandomState(P.FOLD_SEED).permutation(np.arange(171)%P.N_FOLDS)
    ro,co=P.eval_transport(Xs,Yt,fold_ids,"procrustes")
    obs=float(np.mean(ro==1)); medr=float(np.median(ro))
    print(f"OBS GATE: top1={obs} median_rank={medr}",flush=True)
    assert obs==1.0 and medr==1.0, "OBS GATE FAIL — estimator not validated on this host; STOP."
    # frozen permutation stream in draw order
    prs=np.random.RandomState(P.PERM_SEED)
    perms=[prs.permutation(171) for _ in range(P.N_PERM)]
    top1=np.load(CKPT) if os.path.exists(CKPT) else np.full(P.N_PERM,np.nan)
    todo=[(i,perms[i]) for i in range(P.N_PERM) if np.isnan(top1[i])]
    done=int(np.sum(~np.isnan(top1)))
    print(f"workers={a.workers} | {len(todo)}/{P.N_PERM} to do",flush=True)
    with Pool(a.workers, initializer=init_worker) as pool:
        for i,t1 in pool.imap_unordered(eval_perm, todo, chunksize=2):
            top1[i]=t1; done+=1
            if done%200==0:
                np.save(CKPT,top1); el=time.time()-t0
                print(f"  {done}/{P.N_PERM} ({el/3600:.2f}h, {done/max(el,1e-9):.2f} perm/s)",flush=True)
    np.save(CKPT,top1); assert not np.isnan(top1).any()
    # determinism (this host): re-run first-30 serially and confirm identical to the parallel result
    rerun=np.array([float(np.mean(P.eval_transport(Xs,Yt[perms[i]],fold_ids,"procrustes")[0]==1)) for i in range(30)])
    det_ok=bool(np.array_equal(rerun, top1[:30]))
    count=int(np.sum(top1>=obs)); p=(1+count)/(P.N_PERM+1)
    blas=""
    try: blas=np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception: blas="unknown"
    out={
      "record":"P-map frozen 10,000-perm floor (LC-50 PATH A): literal reference estimator, process-parallel scheduling-only, bit-identical to serial on this host.",
      "primary_pair":f"{P.PRIMARY_SOURCE}->{P.PRIMARY_TARGET}","observed_top1":obs,"observed_median_rank":medr,
      "frozen":{"n_perm":P.N_PERM,"perm_seed":P.PERM_SEED,"fold_seed":P.FOLD_SEED,"statistic":"top-1 accuracy","estimator":"add-one"},
      "floor_count":count,"p_add_one":p,
      "shuffled_top1_distribution":{"min":float(top1.min()),"mean":float(top1.mean()),
        "p50":float(np.percentile(top1,50)),"p90":float(np.percentile(top1,90)),
        "p99":float(np.percentile(top1,99)),"max":float(top1.max())},
      "first30_thishost":[float(x) for x in top1[:30]],
      "determinism_gate_thishost_first30_reproduced":det_ok,
      "note_perm_values_blas_host_specific":"rank-deficient Procrustes leaves a host-specific LAPACK null-space completion; per-perm top-1 tail values differ by BLAS host and will NOT bit-match the Accelerate LC-47 probe (kept as cross-check). floor_count and p are invariant: no completion lets a shuffle reach top-1=1.000 (probe maxed ~0.01).",
      "host_blas":{"platform":platform.platform(),"cpu_count":os.cpu_count(),"workers":a.workers,"numpy":np.__version__,"blas":blas,"wall_time_h":round((time.time()-t0)/3600,2)},
      "pmap_script_sha256":hashlib.sha256(open(os.path.join(sys.path[0],"pmap_transport.py"),"rb").read()).hexdigest(),
    }
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    open(OUT,"w").write(json.dumps(out,indent=1))
    print("OUTPUT sha256:",hashlib.sha256(open(OUT,"rb").read()).hexdigest())
    print(json.dumps({k:out[k] for k in ("floor_count","p_add_one","shuffled_top1_distribution","determinism_gate_thishost_first30_reproduced","host_blas")},indent=1))

if __name__=="__main__":
    main()
