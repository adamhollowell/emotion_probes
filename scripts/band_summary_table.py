#!/usr/bin/env python3
"""
band_summary_table.py — E1 §6 extended band-summary table (P3.3 extension).

Six P3.3 bands x seven columns, all on the cleaned canonical basis (emotion_vectors_171.pt).
Self-validating: asserts every locked anchor before writing; stops on any mismatch.

Columns:
  1 RSA-to-L31 (cosine-of-cosine of 171x171 cosine matrices)           [locked P3.3, confirm]
  2 Top-5 neighbour Jaccard to L31 (mean over 171)                     [locked P3.5, confirm]
  3 Top-10 neighbour overlap with L31 (mean over 171; band incl L31)   [locked P3.6 4 bands + late/final]
  4 PC1 variance % (per-layer global PCA over 171)   <- layerwise_va_global_pca.csv pc1_var_pct
  5 Valence r, PC1<->pleasure                        <- layerwise_va_global_pca.csv r_pc1_pleasure
  6 Arousal r, PC2<->arousal                         <- layerwise_va_global_pca.csv r_pc2_arousal
  7 Norm coefficient of variation                    <- vector_norms_layer_summary.csv cv_norm (cleaned)

Cols 4/5/6 read from the corrected GLOBAL-PCA sweep (layerwise_va_global.py output);
the earlier local-PCA sweep (layerwise_external_affect_alignment.csv) is NOT used.

Run from project root:  python3 scripts/band_summary_table.py
Output: geometry_out/band_summary_table.csv
"""
# --- basis gate: refuse to run unless the vectors match the expected sha256.
# Identify the vectors by their hash, not the filename `emotion_vectors_171.pt`. ---
import hashlib as _hlg, pathlib as _plg
_VECG = _plg.Path("data/vectors/emotion_vectors_171.pt")
_EXPG = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
if _VECG.exists():
    _hg = _hlg.sha256(_VECG.read_bytes()).hexdigest()
    assert _hg == _EXPG, (
        "%s basis mismatch: %s != expected canonical %s -- refusing non-canonical basis"
        % (_VECG, _hg[:12], _EXPG[:12]))
# --- end basis gate ---

import numpy as np, csv, sys, os
try:
    import torch
    def _load(p): return torch.load(p, map_location="cpu", weights_only=False)
    def _arr(x): return np.asarray(x.float() if hasattr(x, "float") else x, dtype=np.float64)
except Exception:
    sys.path.insert(0, "outputs")
    from load_pt import load_pt as _load
    def _arr(x): return np.asarray(x, dtype=np.float64)

BANDS = [("early L0-11", range(0, 12)), ("mid L12-21", range(12, 22)), ("seam L22-26", range(22, 27)),
         ("canonical L27-34", range(27, 35)), ("late L35-40", range(35, 41)), ("final L41-45", range(41, 46))]

p = _load("data/vectors/emotion_vectors_171.pt"); em = list(p["emotions"]); n = len(em)
V = _arr(p["vectors"]); iu = np.triu_indices(n, 1)
unit = lambda X: X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

cosvec, top5, top10 = {}, {}, {}
for L in range(46):
    Xn = unit(V[:, L, :]); C = Xn @ Xn.T; cosvec[L] = C[iu]
    order = np.argsort(-C, axis=1)
    top5[L] = {i: set([j for j in order[i] if j != i][:5]) for i in range(n)}
    top10[L] = {i: set([j for j in order[i] if j != i][:10]) for i in range(n)}
c31 = cosvec[31] / np.linalg.norm(cosvec[31])
rsaL = [float((cosvec[L] / np.linalg.norm(cosvec[L])) @ c31) for L in range(46)]
jacL = [float(np.mean([len(top5[L][i] & top5[31][i]) / len(top5[L][i] | top5[31][i]) for i in range(n)])) for L in range(46)]
ovL = [float(np.mean([len(top10[L][i] & top10[31][i]) for i in range(n)])) for L in range(46)]

def readcsv(path, key, filt=None):
    return {int(float(r["layer"])): float(r[key]) for r in csv.DictReader(open(path)) if not filt or filt(r)}
pc1var = readcsv("geometry_out/layerwise_va_global_pca.csv", "pc1_var_pct")
valr = readcsv("geometry_out/layerwise_va_global_pca.csv", "r_pc1_pleasure")
aror = readcsv("geometry_out/layerwise_va_global_pca.csv", "r_pc2_arousal")
# Table 2 pre-seam arousal uses the magnitude convention (mean of per-layer |r|, disclosed
# in the caption): abs BEFORE averaging. Post-seam cells are unchanged (all r > 0). The
# per-layer CSV keeps signed r as ground truth; this is a display-column aggregation only.
aror_abs = {L: abs(v) for L, v in aror.items()}
cov = readcsv("geometry_out/vector_norms_layer_summary.csv", "cv_norm", filt=lambda r: r["vector_set"] == "cleaned")

bm = lambda d, lay: float(np.mean([d[L] for L in lay]))
bl = lambda lst, lay: float(np.mean([lst[L] for L in lay]))

def A(cond, msg):
    if not cond: sys.exit("ANCHOR FAIL: " + msg)
A(abs(pc1var[31]-26.732) < .02, "pc1var L31"); A(abs(pc1var[12]-30.09) < .2, "pc1var L12"); A(abs(pc1var[45]-42.14) < .3, "pc1var L45")
A(abs(valr[31]-0.720) < .001, "val L31"); A(abs(valr[34]-0.749) < .002 and max(valr, key=valr.get) == 34, "val peak L34")
A(abs(aror[31]-0.668) < .001 and max(aror, key=aror.get) in (38, 39, 40), "aro L31/peak L38-40 (near-tied)")
A(abs(cov[0]-0.29) < .01 and abs(cov[45]-0.36) < .01, "cov L0/L45")
rsa_lock = [0.887, 0.898, 0.978, 0.999, 0.998, 0.949]; jac_lock = [0.57, 0.59, 0.79, 0.95, 0.90, 0.77]
for k, (nm, lay) in enumerate(BANDS):
    A(abs(bl(rsaL, lay)-rsa_lock[k]) < .003, "RSA " + nm); A(abs(bl(jacL, lay)-jac_lock[k]) < .01, "Jac " + nm)
can = list(range(27, 35)); incl = bl(ovL, can); excl = np.mean([ovL[L] for L in can if L != 31])
conv = "incl" if abs(incl-9.76) <= abs(excl-9.76) else "excl"
A(min(abs(incl-9.76), abs(excl-9.76)) < .03, "top10 canonical 9.76")
A(abs(bl(ovL, range(0, 12))-6.93) < .05 and abs(bl(ovL, range(12, 22))-7.09) < .05 and abs(bl(ovL, range(22, 27))-8.83) < .05, "top10 early/mid/seam")
ov10band = lambda lay: (np.mean([ovL[L] for L in lay if L != 31]) if conv == "excl" else bl(ovL, lay))
print("ALL ANCHORS PASS. top-10 band convention = %s L31 (incl %.3f / excl %.3f)" % (conv, incl, excl))

rows = [["band", "rsa_to_L31", "top5_jaccard", "top10_overlap", "pc1_var_pct", "valence_r", "arousal_r", "norm_cv"]]
for nm, lay in BANDS:
    rows.append([nm, round(bl(rsaL, lay), 3), round(bl(jacL, lay), 2), round(ov10band(lay), 2),
                 round(bm(pc1var, lay), 2), round(bm(valr, lay), 3), round(bm(aror_abs, lay), 3), round(bm(cov, lay), 2)])
os.makedirs("geometry_out", exist_ok=True)
with open("geometry_out/band_summary_table.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
w = "%-16s %9s %8s %8s %9s %8s %8s %7s"
print(w % tuple(rows[0]))
for r in rows[1:]: print(w % tuple(str(x) for x in r))
print("\nwrote geometry_out/band_summary_table.csv")
