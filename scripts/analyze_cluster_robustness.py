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

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

ROOT = Path(".")
OUT = ROOT / "geometry_out"
OUT.mkdir(exist_ok=True)

VECTORS_PATH = ROOT / "data/vectors/emotion_vectors_171.pt"
GEOM_PATH = OUT / "geometry_summary.json"

LAYER = 31
K_CANONICAL = 10
K_MIN, K_MAX = 6, 15
N_SEEDS = 100
RANDOM_SEEDS = list(range(N_SEEDS))

# ---------------------------------------------------------------------
# Load vectors
# ---------------------------------------------------------------------

blob = torch.load(VECTORS_PATH, map_location="cpu", weights_only=False)
emotions = list(blob["emotions"])
V = blob["vectors"].float().cpu().numpy()[:, LAYER, :]  # cleaned canonical vectors

# K-means on cosine geometry is usually better approximated by normalizing
# vectors first and then using Euclidean k-means.
Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

print(f"Loaded {len(emotions)} emotions from {VECTORS_PATH}")
print(f"Using cleaned vectors at layer {LAYER}: {V.shape}")
print("Clustering normalized vectors to approximate cosine geometry.")

# ---------------------------------------------------------------------
# Load canonical clusters from geometry_summary.json if available
# ---------------------------------------------------------------------

canonical_names = {}
canonical_labels = None
canonical_cluster_map = {}

if GEOM_PATH.exists():
    geom = json.load(open(GEOM_PATH))
    clusters = geom.get("kmeans", {}).get("clusters", [])
    if clusters:
        canonical_labels = np.full(len(emotions), -1, dtype=int)
        for cidx, c in enumerate(clusters):
            cname = c.get("name", f"cluster_{cidx}")
            members = c.get("members", [])
            for e in members:
                if e in emotions:
                    i = emotions.index(e)
                    canonical_labels[i] = cidx
                    canonical_names[cidx] = cname
                    canonical_cluster_map[e] = cname

        if (canonical_labels >= 0).all():
            print(f"Loaded canonical k={len(set(canonical_labels))} labels from geometry_summary.json")
        else:
            print("WARNING: canonical labels incomplete; will still compute k-means stability.")
            canonical_labels = None

if canonical_labels is None:
    print("No complete canonical cluster labels found. Using seed=0 k-means as canonical.")
    km0 = KMeans(n_clusters=K_CANONICAL, random_state=0, n_init=50)
    canonical_labels = km0.fit_predict(Vn)
    canonical_names = {i: f"cluster_{i}" for i in range(K_CANONICAL)}
    canonical_cluster_map = {e: canonical_names[canonical_labels[i]] for i, e in enumerate(emotions)}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def fit_kmeans(k, seed):
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(Vn)
    return km, labels

def cluster_members(labels):
    d = {}
    for i, lab in enumerate(labels):
        d.setdefault(int(lab), []).append(emotions[i])
    return d

def pairwise_same_cluster_matrix(labels):
    labels = np.asarray(labels)
    return labels[:, None] == labels[None, :]

# ---------------------------------------------------------------------
# 1. k=10 seed stability
# ---------------------------------------------------------------------

seed_rows = []
seed_labels = []

for seed in RANDOM_SEEDS:
    km, labels = fit_kmeans(K_CANONICAL, seed)
    seed_labels.append(labels)

    ari_to_canonical = adjusted_rand_score(canonical_labels, labels)
    sil = silhouette_score(Vn, labels, metric="euclidean")

    seed_rows.append({
        "k": K_CANONICAL,
        "seed": seed,
        "ari_to_canonical": float(ari_to_canonical),
        "silhouette_euclidean_normalized": float(sil),
    })

seed_df = pd.DataFrame(seed_rows)

# Pairwise ARI among seeds
pair_rows = []
for a in range(len(seed_labels)):
    for b in range(a + 1, len(seed_labels)):
        pair_rows.append({
            "seed_a": a,
            "seed_b": b,
            "ari": float(adjusted_rand_score(seed_labels[a], seed_labels[b])),
        })
pair_df = pd.DataFrame(pair_rows)

# Co-association stability: how often each pair of emotions clusters together across seeds.
coassoc = np.zeros((len(emotions), len(emotions)), dtype=float)
for labels in seed_labels:
    coassoc += pairwise_same_cluster_matrix(labels).astype(float)
coassoc /= len(seed_labels)

coassoc_rows = []
for i in range(len(emotions)):
    for j in range(i + 1, len(emotions)):
        same_canonical = bool(canonical_labels[i] == canonical_labels[j])
        coassoc_rows.append({
            "emotion_a": emotions[i],
            "emotion_b": emotions[j],
            "canonical_same_cluster": same_canonical,
            "coassociation_rate_k10_100seeds": float(coassoc[i, j]),
            "canonical_cluster_a": canonical_cluster_map.get(emotions[i], ""),
            "canonical_cluster_b": canonical_cluster_map.get(emotions[j], ""),
        })
coassoc_df = pd.DataFrame(coassoc_rows)

# ---------------------------------------------------------------------
# 2. k sweep
# ---------------------------------------------------------------------

ksweep_rows = []
ksweep_labels = {}

for k in range(K_MIN, K_MAX + 1):
    labels_for_k = []
    for seed in RANDOM_SEEDS:
        km, labels = fit_kmeans(k, seed)
        labels_for_k.append(labels)

    ksweep_labels[k] = labels_for_k

    # pairwise ARI across seeds for this k
    aris = []
    for a in range(len(labels_for_k)):
        for b in range(a + 1, len(labels_for_k)):
            aris.append(adjusted_rand_score(labels_for_k[a], labels_for_k[b]))

    # similarity to canonical k=10 labels is defined even if k differs
    ari_to_canon = [adjusted_rand_score(canonical_labels, labels) for labels in labels_for_k]
    sils = [silhouette_score(Vn, labels, metric="euclidean") for labels in labels_for_k]

    ksweep_rows.append({
        "k": k,
        "mean_pairwise_ari_across_seeds": float(np.mean(aris)),
        "median_pairwise_ari_across_seeds": float(np.median(aris)),
        "min_pairwise_ari_across_seeds": float(np.min(aris)),
        "max_pairwise_ari_across_seeds": float(np.max(aris)),
        "mean_ari_to_canonical_k10": float(np.mean(ari_to_canon)),
        "median_ari_to_canonical_k10": float(np.median(ari_to_canon)),
        "mean_silhouette": float(np.mean(sils)),
        "median_silhouette": float(np.median(sils)),
    })

ksweep_df = pd.DataFrame(ksweep_rows)

# ---------------------------------------------------------------------
# 3. Canonical boundary analysis
# ---------------------------------------------------------------------

# Use canonical labels to compute centroids.
unique_labs = sorted(set(canonical_labels))
centroids = []
for lab in unique_labs:
    centroids.append(Vn[canonical_labels == lab].mean(axis=0))
centroids = np.vstack(centroids)
centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

sim_to_centroids = cosine_similarity(Vn, centroids)

boundary_rows = []
for i, e in enumerate(emotions):
    own_lab = int(canonical_labels[i])
    own_idx = unique_labs.index(own_lab)
    sims = sim_to_centroids[i].copy()

    own_sim = float(sims[own_idx])
    sims_without_own = sims.copy()
    sims_without_own[own_idx] = -np.inf
    second_idx = int(np.argmax(sims_without_own))
    second_lab = unique_labs[second_idx]
    second_sim = float(sims[second_idx])

    margin = own_sim - second_sim

    boundary_rows.append({
        "emotion": e,
        "canonical_cluster_id": own_lab,
        "canonical_cluster": canonical_names.get(own_lab, f"cluster_{own_lab}"),
        "own_centroid_cosine": own_sim,
        "second_cluster_id": int(second_lab),
        "second_cluster": canonical_names.get(second_lab, f"cluster_{second_lab}"),
        "second_centroid_cosine": second_sim,
        "boundary_margin": float(margin),
        "boundary_score_low_margin_is_boundary": float(-margin),
    })

boundary_df = pd.DataFrame(boundary_rows).sort_values("boundary_margin", ascending=True)

core_df = boundary_df.sort_values("boundary_margin", ascending=False).copy()

# ---------------------------------------------------------------------
# 4. Per-cluster stability summaries
# ---------------------------------------------------------------------

# For each canonical cluster, measure within-cluster coassociation across seed runs:
# among pairs of members in that canonical cluster, how often do they stay together?
cluster_stability_rows = []

for lab in unique_labs:
    members_idx = np.where(canonical_labels == lab)[0]
    members = [emotions[i] for i in members_idx]
    cname = canonical_names.get(lab, f"cluster_{lab}")

    if len(members_idx) < 2:
        within_mean = np.nan
        within_min = np.nan
        within_median = np.nan
    else:
        vals = []
        for a_i in range(len(members_idx)):
            for b_i in range(a_i + 1, len(members_idx)):
                vals.append(coassoc[members_idx[a_i], members_idx[b_i]])
        within_mean = float(np.mean(vals))
        within_median = float(np.median(vals))
        within_min = float(np.min(vals))

    bsub = boundary_df[boundary_df["canonical_cluster_id"] == lab]

    cluster_stability_rows.append({
        "canonical_cluster_id": int(lab),
        "canonical_cluster": cname,
        "n": int(len(members)),
        "within_cluster_mean_coassociation_k10_100seeds": within_mean,
        "within_cluster_median_coassociation_k10_100seeds": within_median,
        "within_cluster_min_coassociation_k10_100seeds": within_min,
        "mean_boundary_margin": float(bsub["boundary_margin"].mean()),
        "median_boundary_margin": float(bsub["boundary_margin"].median()),
        "min_boundary_margin": float(bsub["boundary_margin"].min()),
        "core_5_by_margin": "; ".join(
            bsub.sort_values("boundary_margin", ascending=False)["emotion"].head(5).tolist()
        ),
        "boundary_5_by_margin": "; ".join(
            bsub.sort_values("boundary_margin", ascending=True)["emotion"].head(5).tolist()
        ),
        "members": "; ".join(members),
    })

cluster_stability_df = pd.DataFrame(cluster_stability_rows).sort_values(
    "within_cluster_mean_coassociation_k10_100seeds", ascending=False
)

# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

seed_df.to_csv(OUT / "cluster_robustness_seed_stability_k10.csv", index=False)
pair_df.to_csv(OUT / "cluster_robustness_seed_pairwise_ari_k10.csv", index=False)
ksweep_df.to_csv(OUT / "cluster_robustness_k_sweep.csv", index=False)
coassoc_df.to_csv(OUT / "cluster_robustness_pair_coassociation.csv", index=False)
boundary_df.to_csv(OUT / "cluster_boundary_emotions.csv", index=False)
core_df.to_csv(OUT / "cluster_core_exemplars.csv", index=False)
cluster_stability_df.to_csv(OUT / "cluster_robustness_cluster_summary.csv", index=False)

summary = {
    "layer": LAYER,
    "vector_set": "cleaned",
    "n_emotions": len(emotions),
    "canonical_k": K_CANONICAL,
    "n_seeds": N_SEEDS,
    "seed_stability_k10": {
        "mean_ari_to_canonical": float(seed_df["ari_to_canonical"].mean()),
        "median_ari_to_canonical": float(seed_df["ari_to_canonical"].median()),
        "min_ari_to_canonical": float(seed_df["ari_to_canonical"].min()),
        "max_ari_to_canonical": float(seed_df["ari_to_canonical"].max()),
        "mean_pairwise_seed_ari": float(pair_df["ari"].mean()),
        "median_pairwise_seed_ari": float(pair_df["ari"].median()),
    },
    "best_k_by_mean_silhouette": int(ksweep_df.sort_values("mean_silhouette", ascending=False).iloc[0]["k"]),
    "best_k_by_seed_stability": int(ksweep_df.sort_values("mean_pairwise_ari_across_seeds", ascending=False).iloc[0]["k"]),
    "most_stable_clusters": cluster_stability_df.head(5).to_dict("records"),
    "least_stable_clusters": cluster_stability_df.tail(5).to_dict("records"),
    "top_boundary_emotions": boundary_df.head(25).to_dict("records"),
    "top_core_exemplars": core_df.head(25).to_dict("records"),
}
with open(OUT / "cluster_robustness_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---------------------------------------------------------------------
# Print compact report
# ---------------------------------------------------------------------

print("\n=== k=10 seed stability ===")
print(seed_df[["ari_to_canonical", "silhouette_euclidean_normalized"]].describe().to_string())
print("\nPairwise ARI across k=10 seeds:")
print(pair_df["ari"].describe().to_string())

print("\n=== k sweep ===")
print(ksweep_df.round(4).to_string(index=False))

print("\n=== Cluster stability summary ===")
cols = [
    "canonical_cluster",
    "n",
    "within_cluster_mean_coassociation_k10_100seeds",
    "within_cluster_min_coassociation_k10_100seeds",
    "mean_boundary_margin",
    "boundary_5_by_margin",
    "core_5_by_margin",
]
print(cluster_stability_df[cols].round(4).to_string(index=False))

print("\n=== Top 30 boundary emotions ===")
print(boundary_df[[
    "emotion",
    "canonical_cluster",
    "second_cluster",
    "own_centroid_cosine",
    "second_centroid_cosine",
    "boundary_margin",
]].head(30).round(4).to_string(index=False))

print("\n=== Top 30 core exemplars ===")
print(core_df[[
    "emotion",
    "canonical_cluster",
    "second_cluster",
    "own_centroid_cosine",
    "second_centroid_cosine",
    "boundary_margin",
]].head(30).round(4).to_string(index=False))

print("\nWrote:")
for p in [
    OUT / "cluster_robustness_seed_stability_k10.csv",
    OUT / "cluster_robustness_seed_pairwise_ari_k10.csv",
    OUT / "cluster_robustness_k_sweep.csv",
    OUT / "cluster_robustness_pair_coassociation.csv",
    OUT / "cluster_boundary_emotions.csv",
    OUT / "cluster_core_exemplars.csv",
    OUT / "cluster_robustness_cluster_summary.csv",
    OUT / "cluster_robustness_summary.json",
]:
    print(" ", p)
