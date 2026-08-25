#!/usr/bin/env python3
"""
analyze_pc1_valence_residual.py — producer for the PC1 valence-residual analysis
("Population valence balance of the emotion set") and its resolution / audit
(PC1 valence-residual test).

Question: does the 171-emotion set lean negative on PC1 (the valence axis)
BEYOND what each emotion's own valence rating predicts? If PC1's negative-side
count is fully explained by the word list's valence composition, the tilt is a
stimulus-set property, not a property of Gemma's geometry.

Method (matches the locked reference spec):
  1. Cleaned L31 emotion vectors from data/vectors/emotion_vectors_171.pt ('vectors').
  2. Global PCA over all 171 (center columns, SVD). PC1 scores.
     PC1 oriented by the locked anchor: positive pole = mean of
     {optimistic, kind, cheerful} (sign-flip if that mean is negative).
  3. NRC-VAD v2.1 valence (data/ratings/nrc_vad_v2_1.csv), matched to the
     emotion names by lowercase/strip, single-token names only -> n = 161.
  4. OLS regress PC1 on NRC-VAD valence (WITH intercept); analyze residuals.

Prints, and self-validates against, the locked reference numbers:
  Full 171 (population shape):   PC1 var 26.732%; 93 neg / 78 pos; raw PC1
                                 skew +0.28; PC1 median -56.5.
  Residual test (NRC-VAD 161):   median residual +8.9; residual skew +0.20;
                                 Wilcoxon signed-rank (two-sided, H0 median 0)
                                 p = 0.99; valence-predicted-negative 101/161;
                                 PC1-negative 88/161.

Run from the PROJECT ROOT (paper env has torch; a numpy .pt loader is used as a
fallback if torch is absent):
    python3 scripts/analyze_pc1_valence_residual.py
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

import csv
import sys

import numpy as np
from scipy import stats

# --- load cleaned L31 vectors (same convention as analyze_sun_choi_engagement.py) ---
sys.path.insert(0, "outputs")  # load_pt lives in the scratch outputs dir during dev
try:
    from load_pt import load_pt
except Exception:
    import torch
    def load_pt(p):
        return torch.load(p, map_location="cpu", weights_only=False)

LAYER = 31
ANCHOR_POS = ["optimistic", "kind", "cheerful"]   # locked positive-pole anchor

p = load_pt("data/vectors/emotion_vectors_171.pt")
emotions = [str(e) for e in p["emotions"]]
V = np.asarray(p["vectors"], dtype=np.float64)     # 'vectors' = CLEANED
X = V[:, LAYER, :]                                  # (171, d_model), L31
n = X.shape[0]
assert n == 171, f"expected 171 emotions, got {n}"

# --- global PCA over all 171, PC1 scores (center + SVD) ---
Xc = X - X.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
scores = U * S                                      # PC scores (171 x k)
var_ratio = (S ** 2) / (S ** 2).sum()
pc1 = scores[:, 0].copy()

# orient PC1 so the anchor set is on the positive pole
idx = {e: i for i, e in enumerate(emotions)}
anchor_mean = np.mean([pc1[idx[a]] for a in ANCHOR_POS if a in idx])
if anchor_mean < 0:
    pc1 = -pc1

print(f"PC1 variance share: {var_ratio[0] * 100:.3f}%   (locked 26.732%)")

# --- full-171 population shape ---
neg = int((pc1 < 0).sum())
pos = int((pc1 > 0).sum())
raw_skew = float(stats.skew(pc1))
pc1_median = float(np.median(pc1))
print(f"\n[full 171] PC1 negative-side: {neg}   positive-side: {pos}   (locked 93 / 78)")
print(f"[full 171] raw PC1 skew: {raw_skew:+.2f}   (locked +0.28)")
print(f"[full 171] PC1 median:   {pc1_median:+.1f}   (locked -56.5)")
order = np.argsort(pc1)
tops = [(emotions[i], pc1[i]) for i in order[::-1][:3]]
bots = [(emotions[i], pc1[i]) for i in order[:3]]
print("[full 171] most positive:", ", ".join(f"{e} {v:+.0f}" for e, v in tops),
      " | most negative:", ", ".join(f"{e} {v:+.0f}" for e, v in bots))

# --- NRC-VAD v2.1 valence, single-token overlap ---
val = {}
with open("data/ratings/nrc_vad_v2_1.csv") as f:
    for r in csv.DictReader(f):
        val[r["word"].strip().lower()] = float(r["valence"])

matched = [(i, val[e.strip().lower()]) for i, e in enumerate(emotions)
           if " " not in e.strip() and e.strip().lower() in val]
mi = np.array([i for i, _ in matched])
x = np.array([v for _, v in matched])       # NRC-VAD valence
y = pc1[mi]                                  # PC1 position
m = len(matched)
print(f"\n[residual test] NRC-VAD overlap n = {m}   (locked 161)")

# --- OLS with intercept: PC1 ~ valence ---
A = np.column_stack([np.ones(m), x])
beta, *_ = np.linalg.lstsq(A, y, rcond=None)
fitted = A @ beta
resid = y - fitted

med_resid = float(np.median(resid))
resid_skew = float(stats.skew(resid))
w_p = float(stats.wilcoxon(resid, alternative="two-sided").pvalue)
pred_neg = int((fitted < 0).sum())
act_neg = int((y < 0).sum())

print(f"[residual test] median residual: {med_resid:+.1f}   (locked +8.9)")
print(f"[residual test] residual skew:   {resid_skew:+.2f}   (locked +0.20)")
print(f"[residual test] Wilcoxon signed-rank p (two-sided): {w_p:.2f}   (locked 0.99)")
print(f"[residual test] valence-predicted-negative (fitted<0): {pred_neg}/{m}   (locked 101/161)")
print(f"[residual test] PC1-negative (y<0):                    {act_neg}/{m}   (locked 88/161)")
print(f"[residual test] mean residual (OLS intercept -> ~0): {np.mean(resid):+.2e}")

# --- self-validation against the locked values ---
checks = {
    "PC1 var 26.732": abs(var_ratio[0] * 100 - 26.732) < 0.01,
    "count 93/78": (neg, pos) == (93, 78),
    "raw skew +0.28": abs(raw_skew - 0.28) < 0.02,
    "PC1 median -56.5": abs(pc1_median - (-56.5)) < 1.0,
    "n=161": m == 161,
    "median resid +8.9": abs(med_resid - 8.9) < 0.2,
    "resid skew +0.20": abs(resid_skew - 0.20) < 0.02,
    "wilcoxon p=0.99": abs(w_p - 0.99) < 0.02,
    "pred-neg 101": pred_neg == 101,
    "act-neg 88": act_neg == 88,
}
print("\n=== self-validation ===")
for k, ok in checks.items():
    print(f"  [{'OK ' if ok else 'FAIL'}] {k}")
if all(checks.values()):
    print("\nAll locked reference values reproduced.")
else:
    print("\n*** MISMATCH: a locked value did not reproduce — investigate before relying on output. ***")
    sys.exit(1)
