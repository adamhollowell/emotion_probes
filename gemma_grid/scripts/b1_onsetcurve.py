#!/usr/bin/env python3
"""B1 — onset-curve seam coincidence (re G6.4); POINT-ESTIMATE ONLY.

Spec of record: B1_B2_EXPLORATORY_PRESTATEMENT_100T.md (sha ae05e8bdcd56...cf06569a).
Admitted labeled unregistered exploratory. Reopens no verdict. The 500-draw resampled CI is
NOT part of this freeze (point estimate only, full neutral, no resampling). Reuses the reference
primitives of analysisB_seam_onset_bootstrap_gesvdfix (clean_layer, ln_of, make_wu_getter,
EPS, EMO_TID) UNMODIFIED. Writes NEW names; MUST NOT touch the frozen B artifacts.

Procedure (frozen): for each 27B cell, across all sampled layers, clean layer with reference
clean_layer (full neutral), norm-lens (v/rms)*(1+ln)@W_U, fair-140 median own-word rank ->
m(l); y(l)=log10 m(l). Plateau Y_P=mean y over l in [0,4]; Floor Y_F=mean y over the deep tail
[42,46] (clamped to available L0..L45 on the 46-layer 27B model -> L42..L45); Y_M=(Y_P+Y_F)/2.
Primary knee = log-midpoint crossing: (a) interpolated fractional layer bracketing the first
descent crossing of Y_M; (b) nearest sampled layer. Secondary knee = steepest-descent step.
Comparison: seam integer L23 (CI[23,24]); COINCIDENT if nearest in {22,23,24,25}; DEEPER >=26;
SHALLOWER <=21. Signed magnitude = interpolated fractional knee - 23. Branches committed below.
Producer does not grade.
"""
import os, sys, json, hashlib
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
REPO=os.path.dirname(os.path.dirname(HERE))
import analysisB_seam_onset_bootstrap_gesvdfix as B   # reference primitives (unmodified)

C27=["gemma-2-27b-base","gemma-2-27b-it"]
SEAM=23; SEAM_CI=[23,24]
OUT_CURVE=os.path.join(REPO,"gemma_grid/geometry_out/b1_onsetcurve.json")
OUT_KNEE =os.path.join(REPO,"gemma_grid/geometry_out/b1_onsetcurve_knee.json")

def curve_for_cell(cell):
    em,Vraw,N=B.load_cell(cell,True)
    nl=Vraw.shape[1]                       # 46 -> layers 0..45
    sel=np.arange(N.shape[0])              # full neutral, point estimate (no resampling)
    fair=[em.index(e) for e in em if e in B.EMO_TID]
    tids=np.array([B.EMO_TID[e] for e in em if e in B.EMO_TID])
    ln=B.ln_of(cell); getter=B.make_wu_getter(cell)
    m={}
    for l in range(nl):
        vl=B.clean_layer(Vraw[:,l,:], N[sel,l,:])[fair]
        rms=np.sqrt((vl**2).mean(1,keepdims=True)+B.EPS); Vn=(vl/rms)*(1.0+ln)
        ranks=getter(Vn,tids)
        m[l]=float(np.median(ranks))
        print(f"  {cell} L{l}: median_own_rank={m[l]:.1f}",flush=True)
    return nl, m

def window_mean(y, lo, hi, nl):
    idx=[l for l in range(lo,hi+1) if l < nl]     # inclusive, clamped to available layers
    return float(np.mean([y[l] for l in idx])), idx

def knee_analysis(m, nl):
    ys=[np.log10(max(m[l],1.0)) for l in range(nl)]
    y={l:ys[l] for l in range(nl)}
    Y_P,plat_idx=window_mean(y,0,4,nl); Y_F,floor_idx=window_mean(y,42,46,nl); Y_M=(Y_P+Y_F)/2.0
    # (a) interpolated fractional layer at first DESCENT crossing of Y_M
    interp=None; nearest=None
    for l in range(nl-1):
        if y[l] >= Y_M and y[l+1] < Y_M:          # descending through midpoint
            frac=l + (y[l]-Y_M)/((y[l]-y[l+1]) if (y[l]-y[l+1])!=0 else 1e-12)
            interp=float(frac); nearest=int(round(frac)); break
    # secondary knee = steepest-descent step l->l+1 maximizing -(y(l+1)-y(l))
    drops=[(y[l]-y[l+1], l) for l in range(nl-1)]
    best=max(drops, key=lambda t:t[0]); sec_lower=int(best[1]); sec_frac_mid=sec_lower+0.5
    # comparison branch
    if nearest is None: branch="UNRESOLVED (no midpoint crossing)"
    elif nearest in (22,23,24,25): branch="COINCIDENT"
    elif nearest >= 26: branch="DEEPER"
    elif nearest <= 21: branch="SHALLOWER"
    else: branch="?"
    signed_mag=(interp-SEAM) if interp is not None else None
    return {"y_log10_rank":{str(l):y[l] for l in range(nl)},
            "plateau_Y_P":Y_P,"plateau_layers":plat_idx,
            "floor_Y_F":Y_F,"floor_layers":floor_idx,"midpoint_Y_M":Y_M,
            "primary_knee_interp_frac":interp,"primary_knee_nearest_sampled":nearest,
            "secondary_knee_lower_layer":sec_lower,"secondary_knee_frac_mid":sec_frac_mid,
            "seam":SEAM,"seam_ci":SEAM_CI,"branch":branch,
            "signed_frac_distance_from_seam":signed_mag}

def main():
    curves={}; knees={}
    for cell in C27:
        print(f"[B1] {cell} full-depth onset curve (point estimate)...",flush=True)
        nl,m=curve_for_cell(cell)
        curves[cell]={"n_layers":nl,"median_own_rank_by_layer":{str(l):m[l] for l in range(nl)}}
        knees[cell]=knee_analysis(m,nl)
    branch_note={"COINCIDENT":"coincidence becomes threshold-free; the G6.4 caveat dissolves.",
                 "DEEPER":"the registered <10k definition was load-bearing; weaker scoped sentence ships.",
                 "SHALLOWER":"readability consolidates ahead of the geometry seam; coincidence breaks the other way (readability leads the seam)."}
    curve_out={"record":"B1 onset-curve (point estimate, full neutral, no resampling) — 27B fair-140 median own-word rank per layer.",
               "spec_sha256":"ae05e8bdcd5611558f0999217f2555c1037a52cc95ef2e4e033fa589cf06569a",
               "note":"NEW artifact; frozen B artifacts untouched. Norm-lens (v/rms)*(1+ln)@W_U; reference clean_layer.",
               "cells":curves}
    knee_out={"record":"B1 knee vs G6.4 seam (L23, CI[23,24]) — log-midpoint crossing + steepest-descent secondary.",
              "spec_sha256":"ae05e8bdcd5611558f0999217f2555c1037a52cc95ef2e4e033fa589cf06569a",
              "window_note":"46-layer 27B model -> layers L0..L45; floor window [42,46] clamps to L42..L45.",
              "branches_committed":branch_note,"cells":knees}
    open(OUT_CURVE,"w").write(json.dumps(curve_out,indent=1))
    open(OUT_KNEE,"w").write(json.dumps(knee_out,indent=1))
    print("CURVE sha256:",hashlib.sha256(open(OUT_CURVE,'rb').read()).hexdigest())
    print("KNEE  sha256:",hashlib.sha256(open(OUT_KNEE,'rb').read()).hexdigest())
    for c in C27:
        k=knees[c]; print(f"{c}: nearest_knee={k['primary_knee_nearest_sampled']} interp={k['primary_knee_interp_frac']} branch={k['branch']}")

if __name__=="__main__":
    main()
