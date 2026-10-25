#!/usr/bin/env python3
"""Analysis B — v2 FULL-SCOPE parallel wrapper/collector (RUNNER_REQUEST v2, 2026-09-01).

Adds the THREE Coordinator-audited deliverables, all riding ONE onset pass, as NEW wrapper/collector
code over the REFERENCE primitives imported from the gesvd successor
(analysisB_seam_onset_bootstrap_gesvdfix) — clean_layer (gesvd-patched), seam_of, make_wu_getter,
ln_of, load_cell, costri, WIN, EPS, LAYER, EMO_TID, SEED. The frozen estimator (11a12790) and the
successor (a9e5e376) are NOT modified. RNG untouched (sels from default_rng(20260828)) -> core values
must not move vs the headline run (c6d8546e).

Additions:
  1. Draw-level CSV (pre-statement 09eaa4fb §141): one row/draw x 500; columns = seam(6 cells);
     onset at {lt100,lt1000,lt10000} (2x27B); gap at each threshold (2x27B); L0-readable bool (4x 2B/9B).
     -> geometry_out/analysisB_drawlevel.csv  (flushed+hashed before teardown).
  2. lt100/lt1000 gap sensitivity riders (27B): FULL onset CURVE — median own-rank at EVERY WIN layer
     (no early return). onset(theta)=first WIN layer with median<theta for theta in {100,1000,10000};
     gap(theta)=round(seam/nl - onset(theta)/nl, 4) [frozen seam_gap_metric convention = seam_frac-onset_frac,
     matches headline]. Report dist+95%CI+frac_zero at lt100/lt1000 LABELED SENSITIVITY; <10k stays headline.
     FAITHFULNESS: curve-derived <10k onset == frozen onset_of <10k per draw (asserted in --validate).
  3. Real 2B/9B L0-readability: per 2B/9B cell, clean L0 (reference clean_layer), norm-lens via
     make_wu_getter(cell), fair-set median own-rank at L0, readable if median<10000; report frac of draws.

Modes: --validate N [--workers W] ; --timeprobe K --workers W ; --draws N --onset-B M --workers W.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v]="1"
import sys, json, time, argparse, hashlib, csv
import numpy as np
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analysisB_seam_onset_bootstrap_gesvdfix as Bmod   # gesvd-patched reference primitives

SEED=Bmod.SEED; CELLS=Bmod.CELLS; C27=Bmod.C27; LAYER=Bmod.LAYER; EMO_TID=Bmod.EMO_TID
WIN=Bmod.WIN; EPS=Bmod.EPS; REPO=Bmod.REPO
CELLS_2B9B=["gemma-2-2b-base","gemma-2-2b-it","gemma-2-9b-base","gemma-2-9b-it"]
THETAS=[100,1000,10000]; HEAD=10000
OUTP=os.path.join(REPO,"gemma_grid/geometry_out/analysisB_seam_onset_bootstrap.json")
CSVP=os.path.join(REPO,"gemma_grid/geometry_out/analysisB_drawlevel.csv")

def gen_sels(draws):
    rng=np.random.default_rng(SEED); return [rng.integers(0,500,500) for _ in range(draws)]

# ---- reference-primitive helpers (new collector code; every clean uses Bmod.clean_layer=gesvd) ----
def onset_curve(Vraw, N, sel, ln, getter, fair, tids):
    """median own-rank at EVERY WIN layer (full curve; no early return). Mirrors onset_of's per-layer
    clean/lens/median exactly, in WIN order."""
    meds={}
    for l in WIN:
        vl=Bmod.clean_layer(Vraw[:,l,:], N[sel,l,:])[fair]
        rms=np.sqrt((vl**2).mean(1,keepdims=True)+EPS); Vn=(vl/rms)*(1.0+ln)
        meds[l]=float(np.median(getter(Vn,tids)))
    return meds
def onset_at(meds, theta):
    for l in WIN:
        if meds[l]<theta: return l
    return None
def l0_median(Vraw, N, sel, ln, getter, fair, tids):
    vl=Bmod.clean_layer(Vraw[:,0,:], N[sel,0,:])[fair]
    rms=np.sqrt((vl**2).mean(1,keepdims=True)+EPS); Vn=(vl/rms)*(1.0+ln)
    return float(np.median(getter(Vn,tids)))

# ---- shared (parent, COW) vs per-process (getters) ----
G={}
def load_shared():
    G["data"]={c:Bmod.load_cell(c,True) for c in CELLS}
    G["lns"]={c:Bmod.ln_of(c) for c in CELLS}
    G["fair"]={c:[G["data"][c][0].index(e) for e in G["data"][c][0] if e in EMO_TID] for c in CELLS}
    G["tids"]={c:np.array([EMO_TID[e] for e in G["data"][c][0] if e in EMO_TID]) for c in CELLS}
def init_worker():
    if "data" not in G: load_shared()   # fork: inherited from parent (COW, memory-optimal); spawn: load here
    G["getters"]={c:Bmod.make_wu_getter(c) for c in CELLS}   # per-process handles, all 6 cells

def compute_draw(d, sel):
    data=G["data"]
    seams={c:Bmod.seam_of(*data[c][1:3],LAYER[c],sel) for c in CELLS}
    onset={c:{} for c in C27}; gap={c:{} for c in C27}
    for c in C27:
        em,Vraw,N=data[c]; nl=Vraw.shape[1]; sm=seams[c]
        meds=onset_curve(Vraw,N,sel,G["lns"][c],G["getters"][c],G["fair"][c],G["tids"][c])
        for th in THETAS:
            o=onset_at(meds,th); onset[c][th]=o
            gap[c][th]=None if (o is None or sm is None) else round(sm/nl - o/nl,4)
    l0={}
    for c in CELLS_2B9B:
        em,Vraw,N=data[c]
        med=l0_median(Vraw,N,sel,G["lns"][c],G["getters"][c],G["fair"][c],G["tids"][c])
        l0[c]=bool(med<10000)
    return d, seams, onset, gap, l0

def _work(arg): return compute_draw(*arg)

def run_parallel(draws, workers):
    sels=gen_sels(draws); load_shared()
    seams={c:[None]*draws for c in CELLS}
    onset={c:{th:[None]*draws for th in THETAS} for c in C27}
    gap={c:{th:[None]*draws for th in THETAS} for c in C27}
    l0={c:[None]*draws for c in CELLS_2B9B}
    with Pool(workers, initializer=init_worker) as pool:
        for d,sd,od,gd,l0d in pool.imap_unordered(_work, [(d,sels[d]) for d in range(draws)], chunksize=1):
            for c in CELLS: seams[c][d]=sd[c]
            for c in C27:
                for th in THETAS: onset[c][th][d]=od[c][th]; gap[c][th][d]=gd[c][th]
            for c in CELLS_2B9B: l0[c][d]=l0d[c]
    return seams, onset, gap, l0

def run_serial(draws):
    """serial replication for the bit-identity gate (single process)."""
    sels=gen_sels(draws); load_shared(); init_worker()
    seams={c:[None]*draws for c in CELLS}
    onset={c:{th:[None]*draws for th in THETAS} for c in C27}
    gap={c:{th:[None]*draws for th in THETAS} for c in C27}
    l0={c:[None]*draws for c in CELLS_2B9B}
    for d in range(draws):
        _,sd,od,gd,l0d=compute_draw(d,sels[d])
        for c in CELLS: seams[c][d]=sd[c]
        for c in C27:
            for th in THETAS: onset[c][th][d]=od[c][th]; gap[c][th][d]=gd[c][th]
        for c in CELLS_2B9B: l0[c][d]=l0d[c]
    return seams, onset, gap, l0

def dist(x):
    x=[v for v in x if v is not None]
    return {} if not x else {"min":int(np.min(x)),"p50":float(np.percentile(x,50)),
            "p95_ci":[float(np.percentile(x,2.5)),float(np.percentile(x,97.5))],"max":int(np.max(x))}
def ci95(x):
    x=[v for v in x if v is not None]; return [float(np.percentile(x,2.5)),float(np.percentile(x,97.5))] if x else None
def gap_block(g):
    have=any(v is not None for v in g)
    return {"frac_gap_ci95":ci95(g) if have else None,
            "frac_zero_gap":float(np.mean([1.0 if v==0 else 0.0 for v in g if v is not None])) if have else None}

def build_output(seams, onset, gap, l0, draws, onsB, workers, wall):
    out={"record":"Analysis B v2 full-scope — seam/onset-curve/gap + lt100/lt1000 sensitivity + 2B/9B L0-readability (G6.2/G6.4).",
     "params":{"seam_B":draws,"onset_B":onsB,"seed":SEED,"onset_window":WIN,"onset_threshold_headline":"<10000",
               "sensitivity_thresholds":["<100","<1000"],"gap_convention":"seam_frac - onset_frac (frozen seam_gap_metric)"},
     "seam_per_cell":{c:{"dist":dist(seams[c]),"ci95":ci95(seams[c])} for c in CELLS},
     "onset_27B":{c:dist(onset[c][HEAD]) for c in C27},
     "gap_27B":{c:gap_block(gap[c][HEAD]) for c in C27},
     "onset_27B_sensitivity":{f"lt{th}":{c:dist(onset[c][th]) for c in C27} for th in (100,1000)},
     "gap_27B_sensitivity":{f"lt{th}":{c:gap_block(gap[c][th]) for c in C27} for th in (100,1000)},
     "sensitivity_label":"lt100/lt1000 are DISCLOSED SENSITIVITY resolutions; the <10000 onset is the registered headline.",
     "l0_readable_frac_2B9B":{c:float(np.mean([1.0 if v else 0.0 for v in l0[c]])) for c in CELLS_2B9B},
     "parallel_wrapper":{"file":"analysisB_onset_parallel_v2fullscope.py","workers":workers,"wall_time_h":round(wall/3600,3),
       "note":"NEW wrapper/collector over gesvd-successor reference primitives; frozen 11a12790 + successor a9e5e376 byte-stable; RNG default_rng(20260828) unchanged; full onset curve over WIN; core <10k values bit-identical to headline run (Gate 3)."}}
    return out

def write_csv(seams, onset, gap, l0, draws):
    hdr=["draw"]+[f"seam_{c}" for c in CELLS] \
        +[f"onset_lt{th}_{c}" for th in THETAS for c in C27] \
        +[f"gap_lt{th}_{c}" for th in THETAS for c in C27] \
        +[f"l0readable_{c}" for c in CELLS_2B9B]
    with open(CSVP,"w",newline="") as f:
        w=csv.writer(f); w.writerow(hdr)
        for d in range(draws):
            row=[d]+[seams[c][d] for c in CELLS] \
                +[onset[c][th][d] for th in THETAS for c in C27] \
                +[gap[c][th][d] for th in THETAS for c in C27] \
                +[int(l0[c][d]) for c in CELLS_2B9B]
            w.writerow(row)
        f.flush(); os.fsync(f.fileno())

def _cmp(a,b):
    if len(a)!=len(b): return False
    for x,y in zip(a,b):
        if (x is None) or (y is None):
            if not (x is None and y is None): return False
        elif x!=y: return False
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--validate",type=int,default=0)
    ap.add_argument("--timeprobe",type=int,default=0)
    ap.add_argument("--draws",type=int,default=None)
    ap.add_argument("--onset-B",type=int,default=None)
    ap.add_argument("--workers",type=int,default=os.cpu_count())
    a=ap.parse_args()

    if a.validate>0:
        N=a.validate
        print(f"V2 VALIDATE N={N} workers={a.workers}",flush=True)
        ss,so,sg,sl=run_serial(N)
        ps,po,pg,pl=run_parallel(N,a.workers)
        ok=True
        for c in CELLS:
            if not _cmp(ss[c],ps[c]): ok=False; print(f"SEAM MISMATCH {c}",flush=True)
        for c in C27:
            if not _cmp(so[c][HEAD],po[c][HEAD]): ok=False; print(f"ONSET<10k MISMATCH {c}",flush=True)
            if not _cmp(sg[c][HEAD],pg[c][HEAD]): ok=False; print(f"GAP<10k MISMATCH {c}",flush=True)
        for c in CELLS_2B9B:
            if not _cmp(sl[c],pl[c]): ok=False; print(f"L0 MISMATCH {c}",flush=True)
        # FAITHFULNESS: curve-derived <10k onset == frozen onset_of per draw
        load_shared(); init_worker(); sels=gen_sels(N); faith=True
        for d in range(N):
            for c in C27:
                em,Vraw,Nn=G["data"][c]
                frozen=Bmod.onset_of(Vraw,Nn,sels[d],G["lns"][c],G["getters"][c],G["fair"][c],G["tids"][c])
                curve=so[c][HEAD][d]
                if frozen!=curve: faith=False; print(f"FAITH MISMATCH d={d} {c}: onset_of={frozen} curve={curve}",flush=True)
        print("SERIAL onset<10k:",{c:so[c][HEAD] for c in C27},flush=True)
        print("SERIAL seams:",{c:ss[c] for c in CELLS},flush=True)
        print("L0-readable(serial):",{c:sl[c] for c in CELLS_2B9B},flush=True)
        print(f"V2 VALIDATE RESULT: {'PASS' if (ok and faith) else 'FAIL'} | bitident={ok} curve_eq_onset_of={faith}",flush=True)
        sys.exit(0 if (ok and faith) else 2)

    if a.timeprobe>0:
        K=a.timeprobe; print(f"V2 TIMEPROBE K={K} workers={a.workers}",flush=True)
        t=time.time(); run_parallel(K,a.workers); wall=time.time()-t
        per=wall/K; print(f"V2 TIMEPROBE workers={a.workers} draws={K} wall={wall:.1f}s per_draw={per:.1f}s proj_B500_h={per*500/3600:.3f}",flush=True)
        return

    assert a.draws is not None
    onsB=a.onset_B if a.onset_B is not None else a.draws
    assert onsB==a.draws, "v2 computes onset curve every draw; onset_B must equal draws (500)"
    print(f"V2 RUN draws={a.draws} onset_B={onsB} workers={a.workers}",flush=True)
    t=time.time(); seams,onset,gap,l0=run_parallel(a.draws,a.workers); wall=time.time()-t
    write_csv(seams,onset,gap,l0,a.draws)
    out=build_output(seams,onset,gap,l0,a.draws,onsB,a.workers,wall)
    open(OUTP,"w").write(json.dumps(out,indent=1))
    print("JSON sha256:",hashlib.sha256(open(OUTP,'rb').read()).hexdigest(),flush=True)
    print("CSV  sha256:",hashlib.sha256(open(CSVP,'rb').read()).hexdigest(),flush=True)
    print(json.dumps({"seam":{c:out['seam_per_cell'][c]['ci95'] for c in CELLS},
                      "onset_10k":out['onset_27B'],"gap_10k":out['gap_27B'],
                      "l0_frac":out['l0_readable_frac_2B9B']},indent=1),flush=True)

if __name__=="__main__":
    main()
