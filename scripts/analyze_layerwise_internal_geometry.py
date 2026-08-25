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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.stats import spearmanr, pearsonr

ROOT = Path(".")
OUT = ROOT / "geometry_out"
OUT.mkdir(exist_ok=True)

VECTORS_PATH = ROOT / "data/vectors/emotion_vectors_171.pt"
GEOM_PATH = OUT / "geometry_summary.json"

LAYER_CANONICAL = 31
K = 10
RANDOM_STATE = 0

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

blob = torch.load(VECTORS_PATH, map_location="cpu", weights_only=False)
emotions = list(blob["emotions"])
V = blob["vectors"].float().cpu().numpy()  # cleaned vectors: emotions × layers × d

n_emotions, n_layers, d = V.shape

print(f"Loaded cleaned vectors: {V.shape}")
print(f"Canonical layer: L{LAYER_CANONICAL}")

# ---------------------------------------------------------------------
# Canonical L31 cosine geometry and clusters
# ---------------------------------------------------------------------

def l2_normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

V31 = V[:, LAYER_CANONICAL, :]
V31n = l2_normalize(V31)
cos31 = V31n @ V31n.T

# Flatten upper triangle for RSA-like cosine-geometry comparison.
tri = np.triu_indices(n_emotions, k=1)
cos31_flat = cos31[tri]

# Prefer canonical clusters from geometry_summary if available.
canonical_labels = None
canonical_cluster_names = {}

if GEOM_PATH.exists():
    geom = json.load(open(GEOM_PATH))
    clusters = geom.get("kmeans", {}).get("clusters", [])
    if clusters:
        labels = np.full(n_emotions, -1, dtype=int)
        for cidx, c in enumerate(clusters):
            cname = c.get("name", f"cluster_{cidx}")
            canonical_cluster_names[cidx] = cname
            for e in c.get("members", []):
                if e in emotions:
                    labels[emotions.index(e)] = cidx
        if (labels >= 0).all():
            canonical_labels = labels
            print("Loaded canonical cluster labels from geometry_summary.json")

if canonical_labels is None:
    print("No complete canonical cluster labels found; computing seed=0 L31 k=10 labels.")
    km31 = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=50)
    canonical_labels = km31.fit_predict(V31n)
    canonical_cluster_names = {i: f"cluster_{i}" for i in range(K)}

# ---------------------------------------------------------------------
# Layerwise metrics
# ---------------------------------------------------------------------

rows = []
cluster_rows = []

for layer in range(n_layers):
    X = V[:, layer, :]
    Xn = l2_normalize(X)

    norms = np.linalg.norm(X, axis=1)
    cos = Xn @ Xn.T
    cos_flat = cos[tri]

    # Compare layer's pairwise cosine geometry to canonical L31.
    rsa_spearman, rsa_spearman_p = spearmanr(cos_flat, cos31_flat)
    rsa_pearson, rsa_pearson_p = pearsonr(cos_flat, cos31_flat)

    # PCA.
    pca = PCA(n_components=5, random_state=0)
    scores = pca.fit_transform(X)

    pc1 = scores[:, 0].copy()
    pc2 = scores[:, 1].copy()

    # Orient for interpretability only. Does not affect variance.
    if "happy" in emotions and pc1[emotions.index("happy")] < 0:
        pc1 *= -1
    if "outraged" in emotions and pc2[emotions.index("outraged")] < 0:
        pc2 *= -1

    # k=10 clustering at this layer.
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=50)
    labels = km.fit_predict(Xn)

    ari_to_L31 = adjusted_rand_score(canonical_labels, labels)
    sil = silhouette_score(Xn, labels, metric="euclidean")

    rows.append({
        "layer": layer,
        "mean_norm": float(norms.mean()),
        "median_norm": float(np.median(norms)),
        "std_norm": float(norms.std(ddof=0)),
        "cv_norm": float(norms.std(ddof=0) / norms.mean()) if norms.mean() else np.nan,

        "pca_pc1_var": float(pca.explained_variance_ratio_[0]),
        "pca_pc2_var": float(pca.explained_variance_ratio_[1]),
        "pca_pc1_pc2_var": float(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]),
        "pca_pc3_var": float(pca.explained_variance_ratio_[2]),

        "rsa_spearman_to_L31_cosine_geometry": float(rsa_spearman),
        "rsa_pearson_to_L31_cosine_geometry": float(rsa_pearson),

        "k10_ari_to_L31_canonical": float(ari_to_L31),
        "k10_silhouette": float(sil),

        "pc1_happy_score": float(pc1[emotions.index("happy")]) if "happy" in emotions else np.nan,
        "pc2_outraged_score": float(pc2[emotions.index("outraged")]) if "outraged" in emotions else np.nan,
    })

    # Per-layer cluster sizes for diagnostic use.
    sizes = pd.Series(labels).value_counts().sort_index()
    for lab, size in sizes.items():
        cluster_rows.append({
            "layer": layer,
            "cluster_label_seed0": int(lab),
            "n": int(size),
        })

df = pd.DataFrame(rows)
cluster_size_df = pd.DataFrame(cluster_rows)

df.to_csv(OUT / "layerwise_internal_geometry_summary.csv", index=False)
cluster_size_df.to_csv(OUT / "layerwise_internal_cluster_sizes.csv", index=False)

# ---------------------------------------------------------------------
# Band summaries
# ---------------------------------------------------------------------

bands = {
    "early_0_11": list(range(0, 12)),
    "mid_12_21": list(range(12, 22)),
    "seam_22_26": list(range(22, 27)),
    "canonical_band_27_34": list(range(27, 35)),
    "late_35_40": list(range(35, 41)),
    "final_41_45": list(range(41, 46)),
}

band_rows = []
for name, layers in bands.items():
    g = df[df["layer"].isin(layers)]
    band_rows.append({
        "band": name,
        "layers": f"{min(layers)}-{max(layers)}",
        "mean_norm": float(g["mean_norm"].mean()),
        "mean_pc1_pc2_var": float(g["pca_pc1_pc2_var"].mean()),
        "mean_rsa_spearman_to_L31": float(g["rsa_spearman_to_L31_cosine_geometry"].mean()),
        "mean_k10_ari_to_L31": float(g["k10_ari_to_L31_canonical"].mean()),
        "mean_k10_silhouette": float(g["k10_silhouette"].mean()),
    })

band_df = pd.DataFrame(band_rows)
band_df.to_csv(OUT / "layerwise_internal_geometry_band_summary.csv", index=False)

summary = {
    "vector_set": "cleaned",
    "n_emotions": n_emotions,
    "n_layers": n_layers,
    "canonical_layer": LAYER_CANONICAL,
    "k": K,
    "best_layers": {
        "max_mean_norm": df.loc[df["mean_norm"].idxmax()].to_dict(),
        "max_rsa_spearman_to_L31": df.loc[df["rsa_spearman_to_L31_cosine_geometry"].idxmax()].to_dict(),
        "max_k10_ari_to_L31": df.loc[df["k10_ari_to_L31_canonical"].idxmax()].to_dict(),
        "max_k10_silhouette": df.loc[df["k10_silhouette"].idxmax()].to_dict(),
        "max_pc1_pc2_var": df.loc[df["pca_pc1_pc2_var"].idxmax()].to_dict(),
    },
    "band_summary": band_df.to_dict("records"),
}
with open(OUT / "layerwise_internal_geometry_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---------------------------------------------------------------------
# Print compact report
# ---------------------------------------------------------------------

print("\n=== Selected layers ===")
sel_layers = [0, 6, 12, 18, 22, 24, 26, 31, 37, 38, 40, 41, 42, 43, 44, 45]
cols = [
    "layer",
    "mean_norm",
    "pca_pc1_var",
    "pca_pc2_var",
    "pca_pc1_pc2_var",
    "rsa_spearman_to_L31_cosine_geometry",
    "k10_ari_to_L31_canonical",
    "k10_silhouette",
]
print(df[df["layer"].isin(sel_layers)][cols].round(4).to_string(index=False))

print("\n=== Top layers by similarity to L31 cosine geometry ===")
print(df.sort_values("rsa_spearman_to_L31_cosine_geometry", ascending=False)[cols].head(12).round(4).to_string(index=False))

print("\n=== Top layers by k=10 ARI to L31 canonical clusters ===")
print(df.sort_values("k10_ari_to_L31_canonical", ascending=False)[cols].head(12).round(4).to_string(index=False))

print("\n=== Top layers by k=10 silhouette ===")
print(df.sort_values("k10_silhouette", ascending=False)[cols].head(12).round(4).to_string(index=False))

print("\n=== Top layers by mean vector norm ===")
print(df.sort_values("mean_norm", ascending=False)[cols].head(12).round(4).to_string(index=False))

print("\n=== Band summary ===")
print(band_df.round(4).to_string(index=False))

print("\nWrote:")
for p in [
    OUT / "layerwise_internal_geometry_summary.csv",
    OUT / "layerwise_internal_cluster_sizes.csv",
    OUT / "layerwise_internal_geometry_band_summary.csv",
    OUT / "layerwise_internal_geometry_summary.json",
]:
    print(" ", p)
