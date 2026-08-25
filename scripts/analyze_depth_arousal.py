#!/usr/bin/env python3
"""Resolve the depth-arousal question as analysis, not figure-polish.

Four questions, printed in order:

  Q1  SIGN RECONCILIATION. The per-panel r (signed) and the depth-curve |r|
      disagreed (-0.35 vs 0.15 at L10). Cause: PC sign orientation. We compute,
      at every layer, the arousal correlation under three sign conventions and
      show they reconcile:
        (a) signed r with anchors oriented per-layer  (what panels used)
        (b) |r|                                       (what the curve used)
        (c) signed r with a SINGLE fixed global sign  (the correct convention)
      If |(a)| != |(b)| the anchor orientation was flipping; (c) is the fix.

  Q2  WHAT IS L10?  Under the fixed convention, is arousal at L10 absent
      (r~0), weak (small +), or inverted (negative)? Reported with a
      bootstrap CI so "inverted" is only claimed if the CI excludes 0.

  Q3  PC2/PC3 MIXING.  Is early arousal "unstable because it's split across
      PC2 and PC3"? We compute arousal's correlation with PC2, PC3, and with
      the best 2-D combination of {PC2,PC3} (canonical / max), per layer. If
      the combined correlation is high early while PC2 alone is low, the
      mixing claim is demonstrated.

  Q4  VARIANCE-ARTIFACT CONTROL.  Does arousal lag merely because PC2 is a
      weaker component everywhere? We report PC2's variance share by layer
      alongside the arousal correlation; if variance is flat while correlation
      climbs, lag is developmental, not a variance artifact.

Reads: emotion_vectors_171.pt, russell_mehrabian_1977.csv.  No GPU, no plot.
Run:   python3 scripts/analyze_depth_arousal.py
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

import numpy as np
import torch

from hero_common import load_vectors, load_rm, zscore, POS, NEG, HI, LO

RNG = np.random.default_rng(0)


def pca_full(V):
    """Return components (d, k) and variance ratios, unsigned."""
    Vc = V - V.mean(0, keepdims=True)
    U, S, Wt = np.linalg.svd(Vc, full_matrices=False)
    scores = U * S                      # (n, k)
    var = (S ** 2) / (S ** 2).sum()
    return scores, var


def matched(emotions, rm):
    idx = {e: i for i, e in enumerate(emotions)}
    m = [t for t in rm if t in idx]
    ar = np.array([rm[t][1] for t in m])
    va = np.array([rm[t][0] for t in m])
    ii = np.array([idx[t] for t in m])
    return ii, va, ar


def corr(a, b):
    return float(np.corrcoef(zscore(a), zscore(b))[0, 1])


def anchor_sign(scores, emotions, comp_idx, hi_names, lo_names):
    idx = {e: i for i, e in enumerate(emotions)}
    comp = scores[:, comp_idx]
    m = lambda names: np.mean([comp[idx[x]] for x in names if x in idx])
    return 1.0 if m(hi_names) >= m(lo_names) else -1.0


def best_2d_corr(s2, s3, target):
    """Max correlation of any unit combination a*s2+b*s3 with target =
    sqrt(r2^2 + r3^2) when s2,s3 orthonormalized. Return that and the angle."""
    z2, z3, zt = zscore(s2), zscore(s3), zscore(target)
    # orthonormalize z3 against z2
    z3o = z3 - (np.dot(z3, z2) / np.dot(z2, z2)) * z2
    r2 = corr(z2, zt)
    r3o = corr(z3o, zt)
    return float(np.sqrt(r2 ** 2 + r3o ** 2)), r2, r3o


def main():
    emotions, _, p = load_vectors()
    rm = load_rm()
    n_layers = p["vectors"].shape[1]
    ii, va, ar = matched(emotions, rm)

    # global fixed sign: orient PC2 at the analysis layer (L31) once, then keep
    V31 = p["vectors"][:, 31, :].float().numpy()
    s31, _ = pca_full(V31)
    fixed_sign = anchor_sign(s31, emotions, 1, HI, LO)

    print("=" * 74)
    print("Q1  SIGN RECONCILIATION  (arousal corr at each layer, 3 conventions)")
    print("    a=per-layer-anchor signed   b=|r|   c=fixed-global-sign signed")
    print("=" * 74)
    print(" L   var%%PC2  (a)signed  (b)|r|   (c)fixed   flip?")
    rows = []
    for L in range(n_layers):
        V = p["vectors"][:, L, :].float().numpy()
        sc, var = pca_full(V)
        s2 = sc[:, 1][ii]
        r_unsigned = corr(s2, ar)                      # sign depends on SVD gauge
        sgn_layer = anchor_sign(sc, emotions, 1, HI, LO)
        a = r_unsigned * sgn_layer                     # per-layer anchor sign
        b = abs(r_unsigned)
        c = r_unsigned * fixed_sign                    # single fixed sign
        flip = "" if abs(abs(a) - b) < 1e-9 else "  <-- |a|!=b"
        rows.append((L, var[1] * 100, a, b, c))
        if L in (0, 5, 10, 15, 20, 22, 26, 31, 40, 45):
            print("%2d   %6.2f   %+7.3f   %5.3f   %+7.3f %s" % (L, var[1]*100, a, b, c, flip))

    print()
    print("=" * 74)
    print("Q2  WHAT IS L10?  fixed-sign arousal corr with bootstrap 95%% CI")
    print("=" * 74)
    for L in (5, 10, 15, 22, 26, 31, 40):
        V = p["vectors"][:, L, :].float().numpy()
        sc, _ = pca_full(V)
        s2 = sc[:, 1][ii] * fixed_sign
        r = corr(s2, ar)
        boot = []
        n = len(ar)
        for _ in range(2000):
            k = RNG.integers(0, n, n)
            boot.append(corr(s2[k], ar[k]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = ("INVERTED" if hi < 0 else "POSITIVE" if lo > 0 else "INDISTINGUISHABLE FROM 0")
        print("L%2d  r=%+.3f  CI[%+.3f, %+.3f]  -> %s" % (L, r, lo, hi, verdict))

    print()
    print("=" * 74)
    print("Q3  PC2/PC3 MIXING  (arousal corr: PC2 alone, PC3 alone, best 2-D)")
    print("=" * 74)
    print(" L    PC2     PC3    best2D   (mixing if best2D >> |PC2| early)")
    for L in (0, 5, 10, 15, 20, 22, 26, 31, 40):
        V = p["vectors"][:, L, :].float().numpy()
        sc, _ = pca_full(V)
        s2 = sc[:, 1][ii]; s3 = sc[:, 2][ii]
        best, r2, r3o = best_2d_corr(s2, s3, ar)
        print("%2d   %+.3f  %+.3f   %.3f" % (L, r2, r3o, best))

    print()
    print("=" * 74)
    print("Q4  VARIANCE-ARTIFACT CONTROL  (PC2 var share vs arousal corr by layer)")
    print("=" * 74)
    print(" L    var%%PC2   |arousal r|   (developmental if var flat, r rising)")
    for L in (0, 5, 10, 15, 20, 22, 26, 31, 40, 45):
        V = p["vectors"][:, L, :].float().numpy()
        sc, var = pca_full(V)
        s2 = sc[:, 1][ii] * fixed_sign
        print("%2d   %6.2f     %.3f" % (L, var[1]*100, abs(corr(s2, ar))))

    print()
    print("INTERPRETATION GUIDE (read against the printed numbers):")
    print(" * Q1: if (c)fixed is smooth and |(c)| tracks (b), the sign bug is the")
    print("   per-layer anchor flipping; report (c).")
    print(" * Q2: L10 verdict is whatever the CI says -- no adjective beyond it.")
    print(" * Q3: if best2D is high at L10 while PC2 alone is ~0/negative, arousal")
    print("   is present but SPLIT across PC2/PC3 early -> 'mixing' is demonstrated.")
    print(" * Q4: if var%PC2 is roughly flat across depth while |r| climbs, the lag")
    print("   is developmental, not a weak-component artifact.")


if __name__ == "__main__":
    main()
