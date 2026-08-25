#!/usr/bin/env python3
"""
analyze_sun_choi_engagement.py

Pre-submission analyses engaging Sun et al. 2026 (arXiv 2604.03147, "Valence-Arousal
Subspace in LLMs") and Choi & Weber (2604.07382, parabola), plus permutation p-values
for Fig 8. All on existing artifacts; numpy only (no sklearn).

Self-validating: asserts the global-171 L31 PCA reproduces Fig 8 / P3.4 (0.7203 / 0.6675).

Inputs:  data/vectors/emotion_vectors_171.pt (cleaned + raw), data/ratings/russell_mehrabian_1977.csv,
         data/ratings/warriner_cache.csv  (independent VAD lexicon, norm-source robustness);
         OPTIONAL data/ratings/NRC-VAD-Lexicon.txt (Sun et al.'s exact source; drop in to add).
Run from project root:  python3 scripts/analyze_sun_choi_engagement.py
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

import sys, csv, os, numpy as np
sys.path.insert(0, "outputs")  # load_pt lives in the scratch outputs dir during dev; adjust if needed
try:
    from load_pt import load_pt
except Exception:
    # fallback: torch on the Mac
    import torch
    def load_pt(p): return torch.load(p, map_location="cpu", weights_only=False)

SEED = 0
LAYER = 31
np.random.seed(SEED)

p = load_pt("data/vectors/emotion_vectors_171.pt")
em = list(p["emotions"])
Vc = np.asarray(p["vectors"], dtype=np.float64)[:, LAYER, :]       # cleaned
Vr = np.asarray(p["vectors_raw"], dtype=np.float64)[:, LAYER, :]   # raw

def r(a, b): return float(np.corrcoef(a, b)[0, 1])
def pca_scores(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U * S, (S**2 / (S**2).sum())

# ---- R-M 45 overlap ----
rm = {}
for row in csv.DictReader(open("data/ratings/russell_mehrabian_1977.csv")):
    rm[row["Term"].strip().lower()] = (float(row["Pleasure Mean"]), float(row["Arousal Mean"]))
matched = [e for e in em if e in rm]
idx45 = [em.index(e) for e in matched]
plea = np.array([rm[e][0] for e in matched]); arou = np.array([rm[e][1] for e in matched])
assert len(matched) == 45, f"expected 45 R-M overlap, got {len(matched)}"

# ===== BASELINE (Fig 8 / P3.4): global-171 PCA, correlate on 45 =====
sc, var = pca_scores(Vc)
r1 = abs(r(sc[idx45, 0], plea)); r2 = abs(r(sc[idx45, 1], arou))
assert abs(r1 - 0.7203) < 0.005 and abs(r2 - 0.6675) < 0.005, "baseline drift vs Fig 8"
print(f"BASELINE global-171 L31 cleaned: PC1<->pleasure={r1:.4f}  PC2<->arousal={r2:.4f}  "
      f"(PC1var {var[0]*100:.3f}% PC2var {var[1]*100:.3f}%)")

# ===== #2 best-single-PC + supervised multi-PC =====
arousal_rs = [abs(r(sc[idx45, k], arou)) for k in range(10)]
bk_a = int(np.argmax(arousal_rs))
def loo_multiPC(scores_rows, target, K):
    n = len(target)
    X = np.column_stack([np.ones(n)] + [scores_rows[:, k] for k in range(K)])
    preds = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        beta, *_ = np.linalg.lstsq(X[m], target[m], rcond=None)
        preds[i] = X[i] @ beta
    return r(preds, target)
print(f"#2 arousal best single PC = PC{bk_a+1} |r|={arousal_rs[bk_a]:.4f}; "
      f"multi-PC LOO fit PC1-5 r={loo_multiPC(sc[idx45], arou, 5):.4f}, "
      f"PC1-10 r={loo_multiPC(sc[idx45], arou, 10):.4f}")

# ===== #3 raw basis =====
scr, _ = pca_scores(Vr)
print(f"#3 raw L31: PC1<->pleasure={abs(r(scr[idx45,0],plea)):.4f}  "
      f"PC2<->arousal={abs(r(scr[idx45,1],arou)):.4f}  "
      f"arousal best PC=PC{int(np.argmax([abs(r(scr[idx45,k],arou)) for k in range(10)]))+1}")

# ===== #5 parabola |PC1| vs PC2 across all 171 =====
pc1, pc2 = sc[:, 0], sc[:, 1]
if r(sc[idx45, 1], arou) < 0: pc2 = -pc2
print(f"#5 parabola r(|PC1|,PC2) over 171 = {r(np.abs(pc1), pc2):.4f}")

# ===== #6 permutation p =====
def perm_p(a, target, robs, nperm=5000):
    cnt = sum(abs(r(a, np.random.permutation(target))) >= abs(robs) for _ in range(nperm))
    return (cnt + 1) / (nperm + 1)
print(f"#6 perm-p (5000): pleasure r={r1:.4f} p={perm_p(sc[idx45,0],plea,r1):.5f}; "
      f"arousal r={r2:.4f} p={perm_p(sc[idx45,1],arou,r2):.5f}")

# ===== #1 subsample test: refit PCA on 27 of 45 (their small-n regime) =====
V45 = Vc[idx45]
def sub_best(Vs, ar):
    s, _ = pca_scores(Vs)
    rs = [abs(r(s[:, k], ar)) for k in range(min(10, s.shape[1]))]
    bk = int(np.argmax(rs))
    return bk, rs[bk], (abs(r(s[:, 1], ar)) if s.shape[1] > 1 else np.nan)
N = 500; pc2best = 0; pc2rs = []; counts = {}
for _ in range(N):
    sel = np.random.choice(45, 27, replace=False)
    bk, br, pc2r = sub_best(V45[sel], arou[sel])
    counts[bk] = counts.get(bk, 0) + 1; pc2best += (bk == 1); pc2rs.append(pc2r)
bk45, br45, pc2r45 = sub_best(V45, arou)
print(f"#1 subsample n=27 (500 draws): arousal-best-PC dist "
      + ", ".join(f"PC{k+1}:{v}" for k, v in sorted(counts.items()))
      + f"; PC2-best {100*pc2best/N:.0f}%; median|r(PC2,arousal)|={np.median(pc2rs):.3f}")
print(f"   full-45 refit: arousal best PC=PC{bk45+1} |r|={br45:.3f}, PC2 |r|={pc2r45:.3f} "
      f"(vs 171-fit PC2 {r2:.3f}) -> arousal leaves PC2 at small n")

# ===== #4 norm-source robustness: Warriner (independent VAD lexicon) =====
def norm_check(path, wordcol, valcol, aroucol, name):
    rows = list(csv.DictReader(open(path)))
    W = {rw[wordcol].strip().lower(): (float(rw[valcol]), float(rw[aroucol]))
         for rw in rows if rw.get(valcol) and rw.get(aroucol)}
    mm = [e for e in em if e in W]; ix = [em.index(e) for e in mm]
    v = np.array([W[e][0] for e in mm]); a = np.array([W[e][1] for e in mm])
    ba = int(np.argmax([abs(r(sc[ix, k], a)) for k in range(10)]))
    print(f"#4 {name} (n={len(mm)}): PC1<->valence={abs(r(sc[ix,0],v)):.4f}  "
          f"PC2<->arousal={abs(r(sc[ix,1],a)):.4f}  arousal best PC=PC{ba+1}")
norm_check("data/ratings/warriner_cache.csv", "Word", "V.Mean.Sum", "A.Mean.Sum", "Warriner")
# NRC-VAD (Sun et al.'s source). Canonicalize to columns word,valence,arousal,dominance.
# v2.1 (2025) preferred -> nrc_vad_v2_1.csv ; v1 (2018) fallback -> nrc_vad_2018.csv
_nrc = next((q for q in ["data/ratings/nrc_vad_v2_1.csv", "data/ratings/nrc_vad_2018.csv"]
             if os.path.exists(q)), None)
if _nrc:
    _ver = "v2.1" if "v2_1" in _nrc else "v1 (2018)"
    norm_check(_nrc, "word", "valence", "arousal", f"NRC-VAD {_ver} (Sun et al.'s source)")
else:
    print("   NRC-VAD not on disk; drop data/ratings/nrc_vad_v2_1.csv "
          "(cols: word,valence,arousal,dominance) to add Sun et al.'s exact source")
