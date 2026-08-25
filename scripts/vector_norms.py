#!/usr/bin/env python3
"""
vector_norms.py — RECONSTRUCTION of the (lost) vector-norm producer for §5.4 / P1.4.

The original producer script did not survive; its output data files did. This
reconstruction regenerates them deterministically from data/vectors/emotion_vectors_171.pt.

Norm definition: the L2 norm of each emotion's difference-of-means vector at each
layer, computed PER OWN BASIS — cleaned norms with the cleaned PCA, raw norms with
the raw PCA. PC signs are oriented so PC1 correlates positively with
valence anchors (happy/joyful/content vs sad/afraid/disgusted) and PC2 with arousal
anchors (outraged/irate/furious vs fulfilled/peaceful/serene) — the same orientation
convention as the Fig 8 rebuild.

Regenerates the SIX keeper outputs:
  vector_norms_all_emotions_layers.csv   (vector_set, emotion, layer, norm)
  vector_norms_layer_summary.csv         (per-layer mean/median/std/min/max/cv)
  vector_norms_emotion_summary.csv       (per-emotion across-layer stats + L31 rank)
  vector_norms_L31_ranked.csv            (L31 norms, ranked high->low)
  vector_norms_pca_correlations_L31.csv  (L31 norm vs own-basis global-PCA PC1/PC2)
  vector_norms_summary.json              (L31 stats, top/bottom 15, layer peak/lowest)

Does NOT regenerate vector_norms_cluster_summary.csv: that file used the retired
5-member warm partition (the cluster framing is retired; see §5.5).

Verify: run with --out /tmp/vn and diff against geometry_out/vector_norms_* .
Expected (cleaned L31): mean_norm ~491.65, max playful ~883.50, min troubled ~260.17,
layer peak mean-norm at L45; norm_L31 vs PC1 Pearson r ~ +0.406.

Usage:
    python scripts/vector_norms.py                 # writes into geometry_out/
    python scripts/vector_norms.py --out /tmp/vn   # writes elsewhere (for verification)
"""
from __future__ import annotations

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

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

VEC_PATH = Path("data/vectors/emotion_vectors_171.pt")
LAYER = 31
SETS = [("cleaned", "vectors"), ("raw", "vectors_raw")]
POS_ANCHORS = ["happy", "joyful", "content"]
NEG_ANCHORS = ["sad", "afraid", "disgusted"]
HI_AROUSAL = ["outraged", "irate", "furious"]
LO_AROUSAL = ["fulfilled", "peaceful", "serene"]


def orient(coord: np.ndarray, idx: dict, hi: list[str], lo: list[str]) -> np.ndarray:
    """Flip sign so mean(hi) > mean(lo)."""
    if np.mean([coord[idx[e]] for e in hi]) < np.mean([coord[idx[e]] for e in lo]):
        return -coord
    return coord


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default=str(VEC_PATH))
    ap.add_argument("--out", default="geometry_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.vectors, map_location="cpu", weights_only=False)
    emotions = list(blob["emotions"])
    idx = {e: i for i, e in enumerate(emotions)}

    all_rows, layer_rows, emo_rows, l31_rows, pca_rows = [], [], [], [], []
    summary: dict = {}

    for setname, key in SETS:
        V = blob[key].float().numpy()            # (171, 46, 4608), float32
        norms = np.linalg.norm(V, axis=2)        # (171, 46), float32
        n_layers = norms.shape[1]
        l31 = norms[:, LAYER]
        order = np.argsort(-l31)                  # high -> low
        rank = {int(i): r + 1 for r, i in enumerate(order)}

        for e in emotions:
            i = idx[e]
            for L in range(n_layers):
                all_rows.append((setname, e, L, float(norms[i, L])))

        for L in range(n_layers):
            col = norms[:, L]
            m = float(col.mean())
            layer_rows.append((setname, L, m, float(np.median(col)), float(col.std()),
                               float(col.min()), float(col.max()), float(col.std() / m)))

        for e in emotions:
            i = idx[e]
            emo_rows.append((setname, e, float(norms[i].mean()), float(np.median(norms[i])),
                             float(norms[i].max()), int(np.argmax(norms[i])),
                             float(l31[i]), rank[i]))
            l31_rows.append((setname, e, LAYER, float(l31[i]), rank[i]))

        # own-basis global PCA at L31
        X = V[:, LAYER, :]
        Xc = X - X.mean(0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        coords = U[:, :3] * S[:3]
        evr = (S ** 2) / np.sum(S ** 2)
        pc1 = orient(coords[:, 0], idx, POS_ANCHORS, NEG_ANCHORS)
        pc2 = orient(coords[:, 1], idx, HI_AROUSAL, LO_AROUSAL)
        for e in emotions:
            i = idx[e]
            pca_rows.append((setname, e, float(l31[i]), float(pc1[i]), float(pc2[i]),
                             float(abs(pc1[i])), float(abs(pc2[i])),
                             float(evr[0]), float(evr[1])))

        top15 = [{"emotion": emotions[int(i)], "norm": float(l31[int(i)])} for i in order[:15]]
        bot15 = [{"emotion": emotions[int(i)], "norm": float(l31[int(i)])} for i in order[::-1][:15]]
        layer_means = norms.mean(0)
        summary[setname] = {
            "shape": list(V.shape),
            "L31": {
                "mean_norm": float(l31.mean()),
                "median_norm": float(np.median(l31)),
                "std_norm": float(l31.std()),
                "min": {"emotion": emotions[int(order[-1])], "norm": float(l31.min())},
                "max": {"emotion": emotions[int(order[0])], "norm": float(l31.max())},
                "top_15": top15,
                "bottom_15": bot15,
            },
            "layer_peak_mean_norm": {"layer": int(np.argmax(layer_means)),
                                     "mean_norm": float(layer_means.max())},
            "layer_lowest_mean_norm": {"layer": int(np.argmin(layer_means)),
                                       "mean_norm": float(layer_means.min())},
        }

    def write_csv(name, header, rows):
        with open(out / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write_csv("vector_norms_all_emotions_layers.csv",
              ["vector_set", "emotion", "layer", "norm"], all_rows)
    write_csv("vector_norms_layer_summary.csv",
              ["vector_set", "layer", "mean_norm", "median_norm", "std_norm",
               "min_norm", "max_norm", "cv_norm"], layer_rows)
    write_csv("vector_norms_emotion_summary.csv",
              ["vector_set", "emotion", "mean_norm_all_layers", "median_norm_all_layers",
               "max_norm_all_layers", "argmax_layer", "norm_L31", "rank_L31_high_to_low"], emo_rows)
    write_csv("vector_norms_L31_ranked.csv",
              ["vector_set", "emotion", "layer", "norm", "rank_high_to_low"], l31_rows)
    write_csv("vector_norms_pca_correlations_L31.csv",
              ["vector_set", "emotion", "norm_L31", "pc1", "pc2", "abs_pc1", "abs_pc2",
               "pca_explained_pc1", "pca_explained_pc2"], pca_rows)
    with open(out / "vector_norms_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # quick self-report of the load-bearing numbers
    cl = summary["cleaned"]["L31"]
    pc1c = np.array([r[3] for r in pca_rows if r[0] == "cleaned"])
    nrm = np.array([r[2] for r in pca_rows if r[0] == "cleaned"])
    r = float(np.corrcoef(nrm, pc1c)[0, 1])
    print(f"cleaned L31 mean_norm={cl['mean_norm']:.2f} max={cl['max']['emotion']}"
          f"({cl['max']['norm']:.2f}) min={cl['min']['emotion']}({cl['min']['norm']:.2f})")
    print(f"layer peak mean-norm at L{summary['cleaned']['layer_peak_mean_norm']['layer']}")
    print(f"norm_L31 vs PC1 (cleaned) Pearson r = {r:+.3f}   (expected ~ +0.406)")
    print(f"wrote 6 files to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
