#!/usr/bin/env python3
"""B2 — covariate-transfer test (re G5.4a / T.3). Admitted labeled unregistered exploratory.

Spec of record: B1_B2_EXPLORATORY_PRESTATEMENT_100T.md (sha ae05e8bdcd56...cf06569a).
Both interpretation branches committed before the look. Reopens no verdict. Frozen stageB
CSVs stay byte-stable (read-only). Reuses g54a reference primitives (ols_hc3, zscore) unmodified.

Procedure (frozen):
 - Outcome = per-emotion post-Procrustes displacement from the frozen stageB CSVs
     2B: stageB_displacement__gemma-2-2b-base__vs__gemma-2-2b-it.csv
     27B: stageB_displacement__gemma-2-27b-base__vs__gemma-2-27b-it.csv
 - Covariates = arousal (NRC-VAD v2.1) and L0 readability (L0-scalar own-word rank per emotion,
   log10, documented L0-scalar view = mean over the four readable 2B/9B cells of the L0 rank,
   then log10 — g54a lines 100-104, reused verbatim).
 - Fit OLS of 2B displacement on {arousal, L0-readability} on emotions with full covariate
   coverage (report the set and drops). Apply the fitted 2B coefficients to the covariates to
   predict a 27B displacement ranking. Score = Spearman(predicted, actual-27B).
 - Primary baseline (amendment v): RAW Spearman(2B displacement ranking, 27B displacement ranking)
   from the SAME stageB CSVs. T.3's .23 is CONTEXT only, not the baseline.
Producer does not grade.
"""
import os, sys, json, csv, hashlib
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
REPO=os.path.dirname(os.path.dirname(HERE))
import g54a_displacement_model as G   # reference ols_hc3, zscore (unmodified)

GEO=os.path.join(REPO,"gemma_grid/geometry_out")
DISP_2B=os.path.join(GEO,"stageB_displacement__gemma-2-2b-base__vs__gemma-2-2b-it.csv")
DISP_27B=os.path.join(GEO,"stageB_displacement__gemma-2-27b-base__vs__gemma-2-27b-it.csv")
NRC=os.path.join(REPO,"data/ratings/nrc_vad_v2_1.csv")
L0J=os.path.join(GEO,"per_emotion_ownword_ranks.json")
OUT=os.path.join(GEO,"b2_covariate_transfer.json")

def load_disp(path):
    d={}
    with open(path) as f:
        for row in csv.DictReader(f):
            emo=row.get("emotion") or list(row.values())[0]
            val=[v for k,v in row.items() if k!="emotion"][-1]
            d[emo]=float(val)
    return d

def rankdata(x):
    x=np.asarray(x,float); order=np.argsort(x,kind="mergesort"); ranks=np.empty(len(x))
    ranks[order]=np.arange(1,len(x)+1)
    # average ties
    _,inv,cnt=np.unique(x,return_inverse=True,return_counts=True)
    sums=np.zeros(len(cnt)); np.add.at(sums,inv,ranks); avg=sums/cnt
    return avg[inv]
def spearman(a,b):
    ra,rb=rankdata(a),rankdata(b)
    return float(np.corrcoef(ra,rb)[0,1])

def main():
    disp2=load_disp(DISP_2B); disp27=load_disp(DISP_27B)
    nrc={}
    with open(NRC) as f:
        for row in csv.DictReader(f):
            term=(row.get("term") or row.get("word") or row.get("emotion") or "").strip().lower()
            try: nrc[term]=(float(row.get("valence") or row.get("V")),float(row.get("arousal") or row.get("A")))
            except (TypeError,ValueError): continue
    l0=json.load(open(L0J))
    l0_cells={c:r for c,r in l0.items() if ("2b" in c or "9b" in c)}   # four readable 2B/9B cells (documented view)
    def l0_scalar(e):
        vals=[l0_cells[c][e]["L0"] for c in l0_cells if e in l0_cells[c]]  # extract L0 field
        return float(np.mean(vals)) if vals else np.nan

    emos_all=sorted(set(disp2)&set(disp27))
    rows=[]; dropped=[]
    for e in emos_all:
        if e not in nrc: dropped.append((e,"no NRC")); continue
        a=nrc[e][1]; l0r=l0_scalar(e)
        if not np.isfinite(l0r): dropped.append((e,"no L0")); continue
        rows.append({"emotion":e,"disp2":disp2[e],"disp27":disp27[e],
                     "arousal":a,"log_l0":np.log10(max(l0r,1.0))})
    emos=[r["emotion"] for r in rows]
    y2=np.array([r["disp2"] for r in rows]); y27=np.array([r["disp27"] for r in rows])
    X=np.column_stack([np.ones(len(rows)),
                       G.zscore(np.array([r["arousal"] for r in rows])),
                       G.zscore(np.array([r["log_l0"] for r in rows]))])
    b,se,p,r2=G.ols_hc3(X,y2)                       # fit on 2B outcome
    pred=X@b                                         # apply fitted 2B coefficients
    score=spearman(pred,y27)                          # covariate-model transfer to 27B
    baseline_raw=spearman(y2,y27)                     # RAW cross-scale displacement-ranking correlation
    out={"record":"B2 covariate-transfer test (re G5.4a/T.3) — labeled unregistered exploratory.",
         "spec_sha256":"ae05e8bdcd5611558f0999217f2555c1037a52cc95ef2e4e033fa589cf06569a",
         "n_full_coverage":len(rows),"dropped":dropped,"emotions_used":emos,
         "model_terms":["intercept","arousal_z","log_l0_z"],
         "ols_fit_on_2B":{"beta":[float(x) for x in b],"se":[float(x) for x in se],
                          "p_hc3":[float(x) for x in p],"r2":float(r2)},
         "covariate_model_spearman_vs_27B":score,
         "raw_cross_scale_spearman_baseline":baseline_raw,
         "T3_context_only":0.23,
         "covariate_beats_raw":bool(score>baseline_raw),
         "branch":("covariate model beats raw baseline -> emotions moved differ by scale but the "
                   "SELECTING RULE is shared (partial reconciliation of the scale-inconsistency finding)"
                   if score>baseline_raw else
                   "covariate model does NOT beat raw baseline -> no reconciliation; reported as such")}
    open(OUT,"w").write(json.dumps(out,indent=1))
    print("OUT:",OUT); print("sha256:",hashlib.sha256(open(OUT,"rb").read()).hexdigest())
    print(json.dumps({k:out[k] for k in ("n_full_coverage","covariate_model_spearman_vs_27B",
        "raw_cross_scale_spearman_baseline","covariate_beats_raw")},indent=1))

if __name__=="__main__":
    main()
