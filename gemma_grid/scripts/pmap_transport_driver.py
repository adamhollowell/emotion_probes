#!/usr/bin/env python3
"""Driver for the cross-cell transport (§4.3), which writes geometry_out/pmap_transport.json.

Recovered from the run's session record. The run of 2026-08-27 used this driver from a session's
scratch folder that has since been cleared, and its text survives in that session's record. The
released copy changes it in two places only: the adapter list is an argument rather than a scratch
path, and the output goes to the release's location. Everything the result file records is
unchanged, including its notes.

The driver imports the frozen estimator scripts/pmap_transport.py unmodified. It computes every
point estimate and the primary pair's ridge comparison, defers the 10,000-permutation floor to
scripts/pmap_floor_pathA_node.py (pmap_transport_floor.json), and runs a 30-permutation
direction check that is labelled as not the floor. The inputs are the 171 x d analysis-layer
adapters that scripts/nogpu_adapters.py rebuilds.

Run from the repository root, in the analysis environment (the run of record used
OMP_NUM_THREADS=10):
    python gemma_grid/scripts/nogpu_adapters.py --out-dir <scratch>
    OMP_NUM_THREADS=10 python gemma_grid/scripts/pmap_transport_driver.py --cells <scratch>/cells_adapters.json
"""
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np, sys
sys.path.insert(0, "gemma_grid/scripts")
import pmap_transport as P

ap = argparse.ArgumentParser()
ap.add_argument("--cells", required=True, help="cells_adapters.json written by nogpu_adapters.py")
ap.add_argument("--out", default="gemma_grid/geometry_out/pmap_transport.json")
args = ap.parse_args()

cells=json.load(open(args.cells))
data={}; emos_ref=None
for cell,path in cells.items():
    V,em=P.load_vectors(path)
    if emos_ref is None: emos_ref=em
    assert em==emos_ref
    data[cell]=P.center(V)
rs=np.random.RandomState(P.FOLD_SEED); fold_ids=rs.permutation(np.arange(171)%P.N_FOLDS)

def point(src,tgt,ridge=False):
    Xs,Yt=data[src],data[tgt]
    ranks,coss=P.eval_transport(Xs,Yt,fold_ids,"procrustes")
    out={"median_rank":float(np.median(ranks)),"top1_acc":float(np.mean(ranks==1)),
         "top5_acc":float(np.mean(ranks<=5)),"median_cosine":float(np.median(coss))}
    if ridge:
        r2,c2=P.eval_transport(Xs,Yt,fold_ids,"ridge")
        out["ridge_secondary"]={"median_rank":float(np.median(r2)),"top1_acc":float(np.mean(r2==1)),"median_cosine":float(np.median(c2))}
    return out,ranks

res={"spec_sha256":"09b4dacdafa6e3b0e9aa4b0c4d75b4a44d2aaa74b2c041303e7919ae5833ad38",
     "frozen":{"fold_seed":P.FOLD_SEED,"n_folds":P.N_FOLDS,"n_perm":P.N_PERM,"perm_seed":P.PERM_SEED,"ridge_grid":P.RIDGE_GRID},
     "ceiling_branch":"B",
     "ceiling_note":"Branch B selected by artifact inventory BEFORE execution: no replicate/split-half 171xd emotion-vector matrices from independent story samples exist on disk for 2b-base or 27b-base (only PA/HA/100T/50TM basis-or-pooling variants of the SAME story sample; per_story_slice.pt exists but constructing a split-half would be ceiling-construction, which the spec forbids). Grading proceeds against the permutation floor alone.",
     "input_adapter_note":"canonical_100T.pt / emotion_vectors_171_100T.pt store (171,L,d) all-layer cleaned vectors; the frozen loader expects (171,d). Realization-of-intent: sliced each cell at its registered analysis layer (2b17/9b28/27b31 per spec ANALYSIS_LAYERS) into 2D adapters; frozen script + constants byte-identical.",
     "pairs":{}}
prim=f"{P.PRIMARY_SOURCE}->{P.PRIMARY_TARGET}"
t0=time.time()
res["pairs"][prim],prim_ranks=point(P.PRIMARY_SOURCE,P.PRIMARY_TARGET,ridge=True)
res["pairs"][prim]["perm_p_top1"]=None
res["pairs"][prim]["perm_status"]=("DEFERRED_PROHIBITIVE_RUNTIME: frozen floor N_PERM=10000 x 5-fold Procrustes SVD(2304x4608) "
    "= ~12.2s/perm -> ~34h on 10-core CPU. NOT reduced (dispatch honesty note). Point estimates + ridge computed via frozen functions.")
res["pairs"][f"{P.PRIMARY_TARGET}->{P.PRIMARY_SOURCE}"],_=point(P.PRIMARY_TARGET,P.PRIMARY_SOURCE)
names=list(cells)
for a in names:
    for b in names:
        if a>=b: continue
        k=f"{a}->{b}"
        if k not in res["pairs"]:
            res["pairs"][k],_=point(a,b)
print(f"point estimates done ({time.time()-t0:.0f}s)",flush=True)
# NON-FINAL sanity probe: 30 perms, show shuffled top1 ~ 0
obs_top1=float(np.mean(prim_ranks==1))
prs=np.random.RandomState(P.PERM_SEED); Xs,Yt=data[P.PRIMARY_SOURCE],data[P.PRIMARY_TARGET]
ptop=[]
for i in range(30):
    perm=prs.permutation(171); pr,_=P.eval_transport(Xs,Yt[perm],fold_ids,"procrustes"); ptop.append(float(np.mean(pr==1)))
res["perm_sanity_nonfinal"]={"n_perm_sanity":30,"note":"NOT the frozen floor; N_PERM stays 10000 for the graded p. Direction check only.",
    "obs_top1":obs_top1,"perm_top1_max":max(ptop),"perm_top1_mean":float(np.mean(ptop)),
    "sanity_count_ge_obs":int(sum(1 for x in ptop if x>=obs_top1))}
outp=Path(args.out)
outp.write_text(json.dumps(res,indent=1))
print("sha256:",hashlib.sha256(outp.read_bytes()).hexdigest())
print("primary:",json.dumps(res["pairs"][prim]))
print("sanity:",json.dumps(res["perm_sanity_nonfinal"]))
