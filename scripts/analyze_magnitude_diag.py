#!/usr/bin/env python3
"""Magnitude diagnostic: in-distribution class separation vs natural-text spread.
Anthropic's parallel: they read low activation magnitudes on natural documents
as overfitting evidence for their LR probe (essay, Figure 16 discussion). This
applies the same standard to the 171 story-derived emotion vectors.
Identity used (exact, from diff-of-means construction): the mean projection of
emotion e's stories onto v_hat_e exceeds the grand-mean projection by exactly
||v_e||. Natural-text mean/std per emotion are exact from the histogram row.
Diagnostic: separation_sigma = ||v_e|| / std_natural(e).
Writes data/heldout/maxact/magnitude_diagnostic.json.
Run from project root: python3 scripts/analyze_magnitude_diag.py"""
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

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path("data/heldout/maxact")
VECTORS = Path("data/vectors/emotion_vectors_171.pt")
LAYER = 31
PANEL = ["happy", "sad", "angry", "afraid", "calm", "desperate",
         "nervous", "surprised", "proud", "guilty", "inspired", "loving"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = torch.load(VECTORS, map_location="cpu", weights_only=False)
    em = list(p["emotions"])
    V = p["vectors"][:, LAYER, :].float()
    norms = V.norm(dim=-1).numpy()          # cleaned-vector norms = class separation

    z = np.load(BASE / "histograms.npz", allow_pickle=True)
    counts = z["counts"].astype(np.float64)  # (171, bins)
    lo, hi, bins = float(z["lo"]), float(z["hi"]), int(z["bins"])
    centers = lo + (np.arange(bins) + 0.5) * (hi - lo) / bins
    hist_emotions = list(z["emotions"])
    assert hist_emotions == em, "emotion order mismatch between vectors and histograms"

    n = counts.sum(axis=1)
    mean = (counts * centers).sum(axis=1) / n
    var = (counts * (centers[None, :] - mean[:, None]) ** 2).sum(axis=1) / n
    std = np.sqrt(var)
    sep = norms / std

    order = np.argsort(sep)
    rows = {em[i]: {"norm_class_separation": round(float(norms[i]), 1),
                    "natural_mean": round(float(mean[i]), 1),
                    "natural_std": round(float(std[i]), 1),
                    "separation_sigma": round(float(sep[i]), 2)} for i in range(len(em))}

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"vectors_sha256": sha256_file(VECTORS),
                   "histograms_sha256": sha256_file(BASE / "histograms.npz")},
        "layer": LAYER,
        "units": "unit-normalized projections; separation = ||v_e|| by the diff-of-means identity",
        "identity_note": "mean story projection for emotion e minus grand-mean projection equals ||v_e|| exactly; absolute story-side level is not recoverable without the raw activation files (not retained), but the class separation is, and it is the diagnostic quantity",
        "caveats": [
            "in-distribution side is by construction (vectors were built from these stories); this measures scale calibration on natural text, not independent validity",
            "story-side pooling was tokens 50->end; natural side skipped BOS only",
            "natural moments computed from 2.0-wide histogram bins (exact to bin width)",
        ],
        "summary": {
            "separation_sigma_min": round(float(sep.min()), 2),
            "separation_sigma_median": round(float(np.median(sep)), 2),
            "separation_sigma_max": round(float(sep.max()), 2),
            "n_below_1_sigma": int((sep < 1).sum()),
            "n_below_2_sigma": int((sep < 2).sum()),
        },
        "per_emotion": rows,
    }
    jpath = BASE / "magnitude_diagnostic.json"
    jpath.write_text(json.dumps(out, indent=1))

    print("separation (sigma): min %.2f  median %.2f  max %.2f | below 1σ: %d, below 2σ: %d of 171"
          % (sep.min(), np.median(sep), sep.max(), (sep < 1).sum(), (sep < 2).sum()))
    print("\nweakest 8:")
    for i in order[:8]:
        print("  %-16s sep %5.2fσ  (norm %5.0f, nat mean %6.1f, nat std %5.1f)"
              % (em[i], sep[i], norms[i], mean[i], std[i]))
    print("strongest 8:")
    for i in order[::-1][:8]:
        print("  %-16s sep %5.2fσ  (norm %5.0f, nat mean %6.1f, nat std %5.1f)"
              % (em[i], sep[i], norms[i], mean[i], std[i]))
    print("\npanel 12:")
    for e in PANEL:
        r = rows[e]
        print("  %-11s sep %5.2fσ  (norm %5.0f, nat mean %6.1f, nat std %5.1f)"
              % (e, r["separation_sigma"], r["norm_class_separation"], r["natural_mean"], r["natural_std"]))
    print("\nwrote %s" % jpath)


if __name__ == "__main__":
    main()
