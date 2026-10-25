#!/usr/bin/env python3
"""Re-run of the Appendix R norm-robustness battery on the canonical basis.

Identical statistics to norm_robustness_full.py (2026-08-09 run); the ONLY
changes are (1) inputs repointed to the canonical vectors — per-cell canonical.pt,
anchor emotion_vectors_171.pt, embedding
vectors from geometry_out/ — and (2) an M.9-compliant input manifest with
sha256 of EVERY input, written beside the outputs.

Torch-free: tensors are read with ptload.py (stdlib zip+pickle -> numpy).
Validation gates (printed): the R&M column must reproduce the stageA table
(valence_r/arousal_r per cell) and the c3_embedding.json values.
"""
import csv, json, math, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import ptload

REPO = Path(__file__).resolve().parents[2]
RATINGS_DIR = REPO / "data" / "ratings"
GEO = REPO / "gemma_grid" / "geometry_out"

POS = ["happy","joyful","content","blissful","delighted","cheerful","optimistic","loving"]
NEG = ["afraid","sad","angry","miserable","panicked","terrified","depressed","heartbroken"]
HI  = ["enraged","panicked","ecstatic","furious","terrified","hysterical","thrilled","alarmed"]
LO  = ["calm","serene","sleepy","relaxed","listless","peaceful","droopy","at ease"]

CELLS = {
    "gemma-2-2b-base":  ("gemma_grid/data/gemma-2-2b-base/vectors/canonical.pt",  17),
    "gemma-2-2b-it":    ("gemma_grid/data/gemma-2-2b-it/vectors/canonical.pt",    17),
    "gemma-2-9b-base":  ("gemma_grid/data/gemma-2-9b-base/vectors/canonical.pt",  28),
    "gemma-2-9b-it":    ("gemma_grid/data/gemma-2-9b-it/vectors/canonical.pt",    28),
    "gemma-2-27b-base": ("data/vectors/emotion_vectors_171.pt",                      31),
    "gemma-2-27b-it":   ("gemma_grid/data/gemma-2-27b-it/vectors/canonical.pt",   31),
}
TUNING = [("gemma-2-2b-base","gemma-2-2b-it"),
          ("gemma-2-9b-base","gemma-2-9b-it"),
          ("gemma-2-27b-base","gemma-2-27b-it")]
CHAINS = {"base": ("gemma-2-2b-base","gemma-2-27b-base"),
          "IT":   ("gemma-2-2b-it","gemma-2-27b-it")}
FLOOR_R = 0.15
z = np.arctanh
p2 = lambda zz: math.erfc(abs(zz)/math.sqrt(2))

MANIFEST = {}
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    MANIFEST[str(path.relative_to(REPO))] = h.hexdigest()
    return h.hexdigest()

def load_vectors(path):
    sha(path)
    p = ptload.load(path)
    em = list(p["emotions"]); V = np.asarray(p["vectors"], dtype=np.float32)
    return em, V

def pca(Vl, k=5):
    Vc = Vl - Vl.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Vc, full_matrices=False)
    return U[:, :k] * S[:k], (S[:k] ** 2) / (S ** 2).sum()

def amean(c, em, names):
    idx = [em.index(x) for x in names if x in em]
    return float(c[idx].mean()) if idx else 0.0

def signed_coords(Vl, em):
    coords, ratio = pca(Vl, 5)
    if amean(coords[:, 0], em, POS) < amean(coords[:, 0], em, NEG): coords[:, 0] *= -1
    if amean(coords[:, 1], em, HI)  < amean(coords[:, 1], em, LO):  coords[:, 1] *= -1
    return coords, ratio

def load_norm_table(which):
    rr = {}
    if which == "rm":
        f = RATINGS_DIR/"russell_mehrabian_1977.csv"; sha(f)
        for r in csv.DictReader(open(f)):
            rr[r["Term"].strip().lower()] = (float(r["Pleasure Mean"]), float(r["Arousal Mean"]))
    elif which == "nrc":
        f = RATINGS_DIR/"nrc_vad_v2_1.csv"; sha(f)
        for r in csv.DictReader(open(f)):
            rr[r["word"].strip().lower()] = (float(r["valence"]), float(r["arousal"]))
    elif which == "warriner":
        f = RATINGS_DIR/"warriner_cache.csv"; sha(f)
        for r in csv.DictReader(open(f)):
            rr[r["Word"].strip().lower()] = (float(r["V.Mean.Sum"]), float(r["A.Mean.Sum"]))
    return rr

def overlap_arrays(em, tbl, restrict):
    ov = [e for e in em if e.lower() in tbl and (restrict is None or e.lower() in restrict)]
    idx = [em.index(e) for e in ov]
    pl = np.array([tbl[e.lower()][0] for e in ov])
    ar = np.array([tbl[e.lower()][1] for e in ov])
    return idx, pl, ar, ov

def corr_from_coords(coords, idx, pl, ar):
    c1, c2 = coords[idx,0].copy(), coords[idx,1].copy()
    if np.corrcoef(c1, pl)[0,1] < 0: c1 = -c1
    if np.corrcoef(c2, ar)[0,1] < 0: c2 = -c2
    return float(np.corrcoef(c1, pl)[0,1]), float(np.corrcoef(c2, ar)[0,1])

def refit_survives(Vl, idx, ar):
    V = Vl[idx] - Vl[idx].mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(V, full_matrices=False)
    sc = U[:, :10] * S[:10]
    rs = [abs(float(np.corrcoef(sc[:,k], ar)[0,1])) for k in range(min(10, sc.shape[1]))]
    bk = int(np.argmax(rs))
    return {"best_pc": bk+1, "best_abs_r": rs[bk], "pc2_abs_r": rs[1], "survives": bk == 1}

def main():
    G = {}
    for c,(vec,layer) in CELLS.items():
        em, V = load_vectors(REPO/vec)
        Vl = V[:, layer, :]
        acoords, _ = signed_coords(Vl, em)
        epath = GEO/c/"c3_vectors.pt"; sha(epath)
        ep = ptload.load(epath)
        eem = list(ep["emotions"]); Ve = np.asarray(ep["vectors"], dtype=np.float32)
        Ve = Ve.reshape(len(em), -1)
        assert eem == em
        ecoords, _ = signed_coords(Ve, em)
        G[c] = dict(em=em, Vl=Vl, acoords=acoords, ecoords=ecoords)
    em_ref = G["gemma-2-2b-base"]["em"]

    rm_tbl = load_norm_table("rm")
    rm45 = {e.lower() for e in em_ref if e.lower() in rm_tbl}

    specs = [("rm","Russell-Mehrabian [reg]",None),
             ("nrc","NRC VAD [natural]",None),
             ("nrc","NRC VAD [matched45]",rm45),
             ("warriner","Warriner [natural]",None),
             ("warriner","Warriner [matched45]",rm45)]

    out = {}
    for which,label,restrict in specs:
        tbl = load_norm_table(which)
        per = {}; n = None
        for c in CELLS:
            em = G[c]["em"]
            idx, pl, ar, ov = overlap_arrays(em, tbl, restrict); n = len(ov)
            av, aa = corr_from_coords(G[c]["acoords"], idx, pl, ar)
            ev, ea = corr_from_coords(G[c]["ecoords"], idx, pl, ar)
            rf = refit_survives(G[c]["Vl"], idx, ar)
            per[c] = dict(a_val=av, a_aro=aa, e_val=ev, e_aro=ea, refit=rf)
        out[label] = dict(n=n, per=per)

    # ---- validation gates vs the canonical outputs ----------------------
    print("== GATE 1: R&M column vs geometry_out/stageA_cross_cell_table.csv ==")
    gate_csv = REPO/"gemma_grid/geometry_out/stageA_cross_cell_table.csv"; sha(gate_csv)
    locked = {r["cell"]: (float(r["valence_r"]), float(r["arousal_r"]))
              for r in csv.DictReader(open(gate_csv))}
    worst = 0.0
    for c in CELLS:
        got = out["Russell-Mehrabian [reg]"]["per"][c]
        dv = abs(got["a_val"] - locked[c][0]); da = abs(got["a_aro"] - locked[c][1])
        worst = max(worst, dv, da)
        print(f"  {c:22s} val {got['a_val']:+.4f} (locked {locked[c][0]:+.4f})  aro {got['a_aro']:+.4f} (locked {locked[c][1]:+.4f})")
    print(f"  worst |delta| = {worst:.2e}  ->  {'PASS' if worst < 5e-4 else 'FAIL'}")

    print("== GATE 2: R&M embedding vs geometry_out/<cell>/c3_embedding.json ==")
    worst_e = 0.0
    for c in CELLS:
        jf = GEO/c/"c3_embedding.json"; sha(jf)
        ce = json.load(open(jf))
        got = out["Russell-Mehrabian [reg]"]["per"][c]
        lv, la = ce.get("valence_r"), ce.get("arousal_r")
        if lv is None: lv, la = ce.get("emb_valence_r"), ce.get("emb_arousal_r")
        dv = abs(got["e_val"] - lv); da = abs(got["e_aro"] - la)
        worst_e = max(worst_e, dv, da)
        print(f"  {c:22s} e_val {got['e_val']:+.4f} (locked {lv:+.4f})  e_aro {got['e_aro']:+.4f} (locked {la:+.4f})")
    print(f"  worst |delta| = {worst_e:.2e}  ->  {'PASS' if worst_e < 5e-4 else 'FAIL'}")

    (GEO/"norm_robustness_full_results.json").write_text(json.dumps(out, indent=2))
    (REPO/"norm_robustness_full_manifest.json").write_text(json.dumps(
        {"rule": "M.9 — every input hashed at run time", "inputs_sha256": MANIFEST}, indent=2))

    def pooled(per, field, n):
        drs = [per[b][field]-per[a][field] for a,b in TUNING]
        dzs = [z(per[b][field])-z(per[a][field]) for a,b in TUNING]
        zz = float(np.mean(dzs)/np.sqrt(2/(3*(n-3))))
        return float(np.mean(drs)), zz, p2(zz)

    def scale(per, field, chain, n):
        a,b = CHAINS[chain]
        dr = per[b][field]-per[a][field]
        zz = float((z(per[b][field])-z(per[a][field]))/np.sqrt(2/(n-3)))
        return dr, zz, p2(zz)

    print("\n== R.1 (G2.1) pooled tuning [IT-base] ==")
    for label,d in out.items():
        vdr,_,vp = pooled(d["per"],"a_val",d["n"]); adr,_,ap = pooled(d["per"],"a_aro",d["n"])
        print(f"  {label:26s} n={d['n']:>3}  val {vdr:+.3f} (p={vp:.3f})   aro {adr:+.3f} (p={ap:.3f})")

    print("\n== R.2 (G3.2) scale 2b->27b arousal_r ==")
    for label,d in out.items():
        bdr,_,bp = scale(d["per"],"a_aro","base",d["n"]); idr,_,ip = scale(d["per"],"a_aro","IT",d["n"])
        print(f"  {label:26s} n={d['n']:>3}  base {bdr:+.3f} (p={bp:.3f})   IT {idr:+.3f} (p={ip:.3f})")

    print("\n== R.3 (G4.1) interaction DiD arousal_r ==")
    for label,d in out.items():
        per=d["per"]; n=d["n"]
        did=(per["gemma-2-27b-it"]["a_aro"]-per["gemma-2-27b-base"]["a_aro"])-(per["gemma-2-2b-it"]["a_aro"]-per["gemma-2-2b-base"]["a_aro"])
        didz=((z(per["gemma-2-27b-it"]["a_aro"])-z(per["gemma-2-27b-base"]["a_aro"]))-(z(per["gemma-2-2b-it"]["a_aro"])-z(per["gemma-2-2b-base"]["a_aro"])))/math.sqrt(4/(n-3))
        print(f"  {label:26s} n={d['n']:>3}  DiD {did:+.3f} (p={p2(didz):.3f})")

    print("\n== R.4 (G1.3) embedding ranges ==")
    for label,d in out.items():
        evs=[d["per"][c]["e_val"] for c in CELLS]; eas=[d["per"][c]["e_aro"] for c in CELLS]
        print(f"  {label:26s} n={d['n']:>3}  e_val {min(evs):+.3f}..{max(evs):+.3f}   e_aro {min(eas):+.3f}..{max(eas):+.3f}")

    print("\n== R.5 (G3.3) depth-gap scale contrast (z-units) ==")
    for label,d in out.items():
        per=d["per"]
        gap = {c: z(per[c]["a_aro"]) - z(per[c]["e_aro"]) for c in CELLS}
        print(f"  {label:26s} n={d['n']:>3}  base {gap['gemma-2-27b-base']-gap['gemma-2-2b-base']:+.3f}   IT {gap['gemma-2-27b-it']-gap['gemma-2-2b-it']:+.3f}")

    print("\n== R.6 (G1.2) refit best-PC per cell ==")
    for label,d in out.items():
        row = [d["per"][c]["refit"]["best_pc"] for c in CELLS]
        surv = sum(d["per"][c]["refit"]["survives"] for c in CELLS)
        print(f"  {label:26s} n={d['n']:>3}  best-PC {row}  survive {surv}/6")

if __name__ == "__main__":
    main()
