#!/usr/bin/env python3
"""
peremotion_purity.py — producer for the Appendix B cohesion table.

Per-emotion cluster purity: the fraction of an emotion's 10 nearest cosine
neighbours (in the cleaned L31 space) that fall inside its own cluster, where
"cluster" is the native seed-0 k=10 partition (geometry_out/L31_kmeans_partition.csv) — the
same partition as Fig 6 / Table 12. Also writes a per-emotion `core_stability` column:
the fraction of 50 k-means seeds in which the emotion co-clusters with a majority
of its canonical cluster-mates (a cross-seed "does it stay with its core" measure).

Basis: cleaned L31 vectors, canonical partition. Deterministic for purity (uses the
fixed partition — no k-means); core_stability runs a 50-seed sweep.

Output: geometry_out/peremotion_purity.csv  (emotion, cluster, purity, core_stability)

Self-validates the summary stats against the locked P2.4 canonical numbers and
exits nonzero on mismatch:
  mean purity 0.756; 46 mutually-nearest pairs;
  per-cluster mean purity — Hostile Anger 0.953, Fear & Overwhelm 0.890,
  Compassionate Warmth (warm-affiliative) 0.575.
  NB: purity is computed on the canonical native partition (geometry_out/L31_kmeans_partition.csv),
  the same partition as Fig 6 / Table 12. Its warm cluster has 8 members, so per-member
  purity caps at 7/10 = 0.7 and the cluster mean is ~0.575 (canonical). The retired
  prior-basis partition gave a 6-member warm cluster with mean ~0.433; that partition is
  superseded under the single-basis policy and is not used here.

Run from the PROJECT ROOT (needs torch to load the .pt and scikit-learn for the
seed sweep):  python scripts/peremotion_purity.py
"""
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = ROOT / "data/vectors/emotion_vectors_171.pt"
PARTITION_PATH = ROOT / "geometry_out/L31_kmeans_partition.csv"
OUT = ROOT / "geometry_out/peremotion_purity.csv"

# --- basis guard: refuse to run on non-canonical vectors ---------------------------
# The canonical filename carries the canonical bytes (bytes swapped, name kept). Without
# this guard the script silently computes prior-basis numbers if a stale copy is present.
VECTORS_SHA256_EXPECTED = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"  # canonical
_sha = hashlib.sha256(VECTORS_PATH.read_bytes()).hexdigest()
assert _sha == VECTORS_SHA256_EXPECTED, (
    f"emotion_vectors_171.pt basis mismatch: got {_sha[:12]}…, expected canonical "
    f"{VECTORS_SHA256_EXPECTED[:12]}… — refusing to compute purity on a non-canonical basis")

LAYER = 31          # replication analysis layer
K = 10              # canonical number of clusters
TOPN = 10           # nearest-neighbour horizon for purity
N_SEEDS = 50        # cross-seed sweep for core_stability

# --- load, matching analyze_cluster_robustness.py convention ---
blob = torch.load(VECTORS_PATH, map_location="cpu", weights_only=False)
emotions = list(blob["emotions"])
V = blob["vectors"].float().cpu().numpy()[:, LAYER, :]        # (171, d) CLEANED canonical
n = len(emotions)
idx = {e: i for i, e in enumerate(emotions)}

# canonical single-file native partition (geometry_out/L31_kmeans_partition.csv) — the
# one file every consumer (Fig 6, Table 12, coords) reads; sha-gated to block a stale CSV.
PARTITION_SHA256_EXPECTED = "79a3341e0f18373d6e511d8254892b8a4836d132796674d29195cef3f2eed656"  # native
_psha = hashlib.sha256(PARTITION_PATH.read_bytes()).hexdigest()
assert _psha == PARTITION_SHA256_EXPECTED, (
    f"L31_kmeans_partition.csv basis mismatch: got {_psha[:12]}…, expected native "
    f"{PARTITION_SHA256_EXPECTED[:12]}… — refusing to compute purity on a non-canonical partition")
import csv as _csv
cmap = {row["emotion"]: int(row["cluster"]) for row in _csv.DictReader(PARTITION_PATH.open())}
clab = np.array([cmap[e] for e in emotions])

# --- cosine geometry ---
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
S = Vn @ Vn.T
np.fill_diagonal(S, -np.inf)                                  # exclude self

# --- per-emotion top-10 in-cluster purity (deterministic) ---
purity = np.empty(n)
for i in range(n):
    nn = np.argpartition(-S[i], TOPN)[:TOPN]                  # 10 nearest neighbours
    purity[i] = float(np.mean(clab[nn] == clab[i]))

# --- mutually-nearest pairs (each other's top-1), counted once ---
top1 = np.argmax(S, axis=1)
mutual = sum(1 for i in range(n) if top1[top1[i]] == i and i < top1[i])
# self-documenting robustness: tightest NN1-vs-NN2 cosine gap. The pair count cannot
# flip under cross-machine BLAS float64 noise (~1e-15) unless this margin is that small.
_srt = np.sort(S, axis=1)
nn_margin = float((_srt[:, -1] - _srt[:, -2]).min())

# --- core_stability across 50 seeds (canonical s4_clusters.py convention) ---
# K-means runs on the UNNORMALIZED float64 vectors (cosine is used only for purity/NN
# above). stab[i] = mean over 50 seeds of the fraction of i's seed-0 cluster-mates that
# land in i's cluster that seed. Seed 0 included (≡ the canonical partition).
Vf = V.astype(np.float64)
seed_labels = np.array([
    KMeans(n_clusters=K, random_state=s, n_init=10).fit_predict(Vf)
    for s in range(N_SEEDS)
])                                                            # (N_SEEDS, n)
core = np.empty(n)
for i in range(n):
    mates = np.array([j for j in np.where(clab == clab[i])[0] if j != i])
    if mates.size == 0:
        core[i] = float("nan")
        continue
    core[i] = float(np.mean([
        np.mean(seed_labels[s][mates] == seed_labels[s][i]) for s in range(N_SEEDS)
    ]))

# --- write ---
df = pd.DataFrame({"emotion": emotions, "cluster": clab, "purity": purity,
                   "core_stability": core})
df.to_csv(OUT, index=False)
print(f"wrote {OUT} ({n} rows)")

# --- self-validation against locked P2.4 canonical stats ---
mean_purity = float(np.mean(purity))
def cluster_mean(anchor):                                     # cluster identified by an anchor emotion
    c = clab[idx[anchor]]
    return float(np.mean(purity[clab == c]))
anger = cluster_mean("angry")          # Hostile Anger
fear = cluster_mean("afraid")          # Fear & Overwhelm
warm = cluster_mean("compassionate")   # Compassionate Warmth (warm-affiliative)

ranked = df.sort_values("core_stability")
least6 = ranked.head(6)["emotion"].tolist()
print(f"mean purity {mean_purity:.3f} | mutual-NN pairs {mutual} "
      f"(min NN1-NN2 cosine margin {nn_margin:.2e}) | "
      f"anger {anger:.3f}  fear {fear:.3f}  warm {warm:.3f}")
# core_stability is a 50-seed k-means sweep and is sensitive to the scikit-learn version
# (only seed 0 is ARI-locked); the bottom-of-list ordering drifts across versions, so it is
# reported for information, NOT gated. Under sklearn 1.8.0 the bottom-6 is
# {amused, playful, impatient, restless, tired, sleepy}; under other builds it varies.
print("least-stable 6 (core_stability, informational — not gated, sklearn-version-sensitive):", least6)

checks = {
    "mean purity 0.756":          abs(mean_purity - 0.756) < 0.003,
    "mutual-NN pairs == 46":      mutual == 46,
    "Hostile Anger 0.953":        abs(anger - 0.953) < 0.01,
    "Fear & Overwhelm 0.890":     abs(fear - 0.890) < 0.01,
    "Compassionate Warmth 0.575": abs(warm - 0.575) < 0.01,
}
print("=== self-validation (locked P2.4 canonical) ===")
for name, ok in checks.items():
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
if not all(checks.values()):
    print("\n*** MISMATCH: regenerated cohesion stats do not match locked P2.4 canonical. "
          "Stop and investigate before releasing. ***")
    sys.exit(1)
print("\nAll locked P2.4 canonical cohesion stats reproduced.")
