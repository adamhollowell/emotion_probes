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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.stats import pearsonr, spearmanr

ROOT = Path(".")
OUT = ROOT / "geometry_out"
OUT.mkdir(exist_ok=True)

VECTORS_PATH = ROOT / "data/vectors/emotion_vectors_171.pt"

# IMPORTANT:
# Use the corrected Russell & Mehrabian 45-emotion file.
# Do NOT use data/ratings/russell_mehrabian_overlap_45.csv.
# Do NOT use russell_mehrabian_overlap_45_DISCARD_BAD.csv.
RM_CANDIDATES = [
    ROOT / "data/ratings/russell_mehrabian_1977.csv",  # canonical (consolidated per PROV.7c)
    ROOT / "data/ratings/russell_mehrabian_45_emotions.csv",
    ROOT / "russell_mehrabian_45_emotions.csv",
    ROOT / "russell_mehrabian_45_emotions(2).csv",
]

BAD_NAME_PATTERNS = [
    "overlap_45",
    "DISCARD_BAD",
    "warriner",
    "llm",
]

CANONICAL_LAYER = 31

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def fail(msg):
    raise SystemExit(f"\nERROR: {msg}\n")

def corr_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return float(pr), float(pp), float(sr), float(sp)

def orient_to_positive(x, y):
    """Flip x so Pearson correlation with y is nonnegative."""
    pr = pearsonr(x, y)[0]
    if pr < 0:
        return -x, -1
    return x, 1

def find_col(df, candidates):
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # fallback: substring match
    for c in df.columns:
        cl = c.lower()
        if any(cand.lower() in cl for cand in candidates):
            return c
    return None

# ---------------------------------------------------------------------
# Locate and validate Russell/Mehrabian file
# ---------------------------------------------------------------------

rm_path = None
for p in RM_CANDIDATES:
    if p.exists():
        rm_path = p
        break

if rm_path is None:
    fail(
        "Could not find corrected Russell & Mehrabian file. Expected one of:\n"
        + "\n".join(f"  {p}" for p in RM_CANDIDATES)
    )

rm_name_lower = rm_path.name.lower()
if any(pat.lower() in rm_name_lower for pat in BAD_NAME_PATTERNS):
    fail(f"Refusing to use suspicious benchmark file: {rm_path}")

print(f"Using Russell & Mehrabian benchmark: {rm_path}")

rm = pd.read_csv(rm_path)
print("RM columns:", list(rm.columns))
print("RM shape:", rm.shape)

emotion_col = find_col(rm, ["emotion", "word", "term", "label"])
pleasure_col = find_col(rm, ["pleasure_mean", "pleasure", "valence_mean", "valence"])
arousal_col = find_col(rm, ["arousal_mean", "arousal"])

if emotion_col is None:
    fail("Could not identify emotion column in Russell & Mehrabian CSV.")
if pleasure_col is None:
    fail("Could not identify pleasure/valence column in Russell & Mehrabian CSV.")
if arousal_col is None:
    fail("Could not identify arousal column in Russell & Mehrabian CSV.")

rm = rm[[emotion_col, pleasure_col, arousal_col]].copy()
rm.columns = ["emotion", "pleasure", "arousal"]
rm["emotion"] = rm["emotion"].astype(str).str.strip().str.lower()

# Validate expected file size and a few known corrected values if available.
if len(rm) != 45:
    fail(f"Expected 45 Russell & Mehrabian emotions, found {len(rm)} rows in {rm_path}")

# These are sanity checks from the corrected file comparison.
# They are intentionally approximate because columns may be rounded.
known_checks = {
    "depressed": {"pleasure": -0.72},
    "embarrassed": {"arousal": 0.54},
    "upset": {"pleasure": -0.63, "arousal": 0.30},
    "weary": {"arousal": -0.33},
}
for emo, vals in known_checks.items():
    if emo in set(rm["emotion"]):
        row = rm[rm["emotion"] == emo].iloc[0]
        for col, expected in vals.items():
            observed = float(row[col])
            if abs(observed - expected) > 0.03:
                fail(
                    f"Sanity check failed for {emo}.{col}: observed {observed}, "
                    f"expected approximately {expected}. This may be the corrupted file."
                )

# ---------------------------------------------------------------------
# Load vectors
# ---------------------------------------------------------------------

blob = torch.load(VECTORS_PATH, map_location="cpu", weights_only=False)
emotions = [str(e).strip().lower() for e in blob["emotions"]]
V = blob["vectors"].float().cpu().numpy()  # emotions × layers × d

n_emotions, n_layers, d = V.shape
print(f"Loaded vectors: {V.shape}")

# Match emotions.
idx_rows = []
missing = []
for _, row in rm.iterrows():
    emo = row["emotion"]
    if emo in emotions:
        idx_rows.append({
            "emotion": emo,
            "vector_idx": emotions.index(emo),
            "pleasure": float(row["pleasure"]),
            "arousal": float(row["arousal"]),
        })
    else:
        missing.append(emo)

if missing:
    print("WARNING: Missing RM emotions from vector file:", missing)

match = pd.DataFrame(idx_rows)
if len(match) < 40:
    fail(f"Only matched {len(match)} emotions; expected close to 45.")

print(f"Matched {len(match)} Russell & Mehrabian emotions to Gemma vectors.")

# ---------------------------------------------------------------------
# Layerwise external alignment
# ---------------------------------------------------------------------

rows = []
loo = LeaveOneOut()

for layer in range(n_layers):
    X = V[match["vector_idx"].values, layer, :]
    yP = match["pleasure"].values
    yA = match["arousal"].values

    # PCA on matched 45 emotions at this layer.
    pca = PCA(n_components=5, random_state=0)
    scores = pca.fit_transform(X)
    pc1 = scores[:, 0].copy()
    pc2 = scores[:, 1].copy()
    pc3 = scores[:, 2].copy()

    # Report signed/raw correlations plus oriented absolute-positive versions.
    pc1_P_pr_raw, pc1_P_pp, pc1_P_sr_raw, pc1_P_sp = corr_pair(pc1, yP)
    pc2_A_pr_raw, pc2_A_pp, pc2_A_sr_raw, pc2_A_sp = corr_pair(pc2, yA)

    pc1o, pc1_flip = orient_to_positive(pc1, yP)
    pc2o, pc2_flip = orient_to_positive(pc2, yA)

    pc1_P_pr, pc1_P_p, pc1_P_sr, pc1_P_s_p = corr_pair(pc1o, yP)
    pc2_A_pr, pc2_A_p, pc2_A_sr, pc2_A_s_p = corr_pair(pc2o, yA)

    # Cross-correlations, useful for detecting swaps/rotation.
    pc1_A_pr, pc1_A_p, pc1_A_sr, pc1_A_s_p = corr_pair(pc1o, yA)
    pc2_P_pr, pc2_P_p, pc2_P_sr, pc2_P_s_p = corr_pair(pc2o, yP)

    # 2D alignment: can PC1+PC2 predict Pleasure and Arousal?
    Z2 = scores[:, :2]
    regP = LinearRegression()
    regA = LinearRegression()

    predP_loocv = cross_val_predict(regP, Z2, yP, cv=loo)
    predA_loocv = cross_val_predict(regA, Z2, yA, cv=loo)

    P_loocv_pr, P_loocv_p, P_loocv_sr, P_loocv_sp = corr_pair(predP_loocv, yP)
    A_loocv_pr, A_loocv_p, A_loocv_sr, A_loocv_sp = corr_pair(predA_loocv, yA)

    regP.fit(Z2, yP)
    regA.fit(Z2, yA)
    P_r2_in_sample = float(regP.score(Z2, yP))
    A_r2_in_sample = float(regA.score(Z2, yA))

    # 3D alignment for comparison only.
    Z3 = scores[:, :3]
    predP3_loocv = cross_val_predict(LinearRegression(), Z3, yP, cv=loo)
    predA3_loocv = cross_val_predict(LinearRegression(), Z3, yA, cv=loo)
    P3_loocv_pr, _, P3_loocv_sr, _ = corr_pair(predP3_loocv, yP)
    A3_loocv_pr, _, A3_loocv_sr, _ = corr_pair(predA3_loocv, yA)

    mean_norm = float(np.linalg.norm(X, axis=1).mean())

    rows.append({
        "layer": layer,
        "n_matched": int(len(match)),
        "mean_norm_matched": mean_norm,

        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
        "pc1_pc2_var": float(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]),
        "pc3_var": float(pca.explained_variance_ratio_[2]),

        "pc1_pleasure_pearson_raw": float(pc1_P_pr_raw),
        "pc1_pleasure_spearman_raw": float(pc1_P_sr_raw),
        "pc2_arousal_pearson_raw": float(pc2_A_pr_raw),
        "pc2_arousal_spearman_raw": float(pc2_A_sr_raw),

        "pc1_flip_for_positive_pleasure": int(pc1_flip),
        "pc2_flip_for_positive_arousal": int(pc2_flip),

        "pc1_pleasure_pearson_oriented": float(pc1_P_pr),
        "pc1_pleasure_pearson_p": float(pc1_P_p),
        "pc1_pleasure_spearman_oriented": float(pc1_P_sr),
        "pc1_pleasure_spearman_p": float(pc1_P_s_p),

        "pc2_arousal_pearson_oriented": float(pc2_A_pr),
        "pc2_arousal_pearson_p": float(pc2_A_p),
        "pc2_arousal_spearman_oriented": float(pc2_A_sr),
        "pc2_arousal_spearman_p": float(pc2_A_s_p),

        "pc1_arousal_pearson_oriented": float(pc1_A_pr),
        "pc1_arousal_spearman_oriented": float(pc1_A_sr),
        "pc2_pleasure_pearson_oriented": float(pc2_P_pr),
        "pc2_pleasure_spearman_oriented": float(pc2_P_sr),

        "loocv_pc12_to_pleasure_pearson": float(P_loocv_pr),
        "loocv_pc12_to_pleasure_spearman": float(P_loocv_sr),
        "loocv_pc12_to_arousal_pearson": float(A_loocv_pr),
        "loocv_pc12_to_arousal_spearman": float(A_loocv_sr),

        "insample_pc12_to_pleasure_r2": P_r2_in_sample,
        "insample_pc12_to_arousal_r2": A_r2_in_sample,

        "loocv_pc123_to_pleasure_pearson": float(P3_loocv_pr),
        "loocv_pc123_to_pleasure_spearman": float(P3_loocv_sr),
        "loocv_pc123_to_arousal_pearson": float(A3_loocv_pr),
        "loocv_pc123_to_arousal_spearman": float(A3_loocv_sr),
    })

df = pd.DataFrame(rows)

# Useful combined scores.
df["simple_axis_alignment_mean_abs_pearson"] = (
    df["pc1_pleasure_pearson_oriented"].abs() + df["pc2_arousal_pearson_oriented"].abs()
) / 2

df["pc12_loocv_affect_alignment_mean_pearson"] = (
    df["loocv_pc12_to_pleasure_pearson"] + df["loocv_pc12_to_arousal_pearson"]
) / 2

df["pc12_loocv_affect_alignment_mean_spearman"] = (
    df["loocv_pc12_to_pleasure_spearman"] + df["loocv_pc12_to_arousal_spearman"]
) / 2

# ---------------------------------------------------------------------
# Band summary
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
        "mean_norm_matched": float(g["mean_norm_matched"].mean()),
        "mean_pc1_pleasure_pearson": float(g["pc1_pleasure_pearson_oriented"].mean()),
        "mean_pc2_arousal_pearson": float(g["pc2_arousal_pearson_oriented"].mean()),
        "mean_simple_axis_alignment": float(g["simple_axis_alignment_mean_abs_pearson"].mean()),
        "mean_loocv_pc12_to_pleasure_pearson": float(g["loocv_pc12_to_pleasure_pearson"].mean()),
        "mean_loocv_pc12_to_arousal_pearson": float(g["loocv_pc12_to_arousal_pearson"].mean()),
        "mean_pc12_loocv_affect_alignment": float(g["pc12_loocv_affect_alignment_mean_pearson"].mean()),
        "mean_pc1_pc2_var": float(g["pc1_pc2_var"].mean()),
    })

band_df = pd.DataFrame(band_rows)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

df.to_csv(OUT / "layerwise_external_affect_alignment.csv", index=False)
band_df.to_csv(OUT / "layerwise_external_affect_alignment_band_summary.csv", index=False)
match.to_csv(OUT / "layerwise_external_affect_alignment_matched_emotions.csv", index=False)

summary = {
    "benchmark_file": str(rm_path),
    "n_rm_rows": int(len(rm)),
    "n_matched": int(len(match)),
    "missing": missing,
    "canonical_layer": CANONICAL_LAYER,
    "best_layers": {
        "best_pc1_pleasure": df.loc[df["pc1_pleasure_pearson_oriented"].idxmax()].to_dict(),
        "best_pc2_arousal": df.loc[df["pc2_arousal_pearson_oriented"].idxmax()].to_dict(),
        "best_simple_axis_alignment": df.loc[df["simple_axis_alignment_mean_abs_pearson"].idxmax()].to_dict(),
        "best_pc12_loocv_affect_alignment": df.loc[df["pc12_loocv_affect_alignment_mean_pearson"].idxmax()].to_dict(),
        "best_pc12_loocv_pleasure": df.loc[df["loocv_pc12_to_pleasure_pearson"].idxmax()].to_dict(),
        "best_pc12_loocv_arousal": df.loc[df["loocv_pc12_to_arousal_pearson"].idxmax()].to_dict(),
    },
    "canonical_layer_row": df[df["layer"] == CANONICAL_LAYER].iloc[0].to_dict(),
    "band_summary": band_df.to_dict("records"),
}
with open(OUT / "layerwise_external_affect_alignment_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---------------------------------------------------------------------
# Print compact report
# ---------------------------------------------------------------------

selected_layers = [0, 6, 12, 18, 22, 24, 26, 27, 31, 34, 37, 38, 40, 42, 44, 45]
cols = [
    "layer",
    "mean_norm_matched",
    "pc1_var",
    "pc2_var",
    "pc1_pc2_var",
    "pc1_pleasure_pearson_oriented",
    "pc1_pleasure_spearman_oriented",
    "pc2_arousal_pearson_oriented",
    "pc2_arousal_spearman_oriented",
    "loocv_pc12_to_pleasure_pearson",
    "loocv_pc12_to_arousal_pearson",
    "pc12_loocv_affect_alignment_mean_pearson",
]

print("\n=== Matched emotions ===")
print(f"Matched n={len(match)}")
if missing:
    print("Missing:", missing)
else:
    print("Missing: none")

print("\n=== Selected layers ===")
print(df[df["layer"].isin(selected_layers)][cols].round(4).to_string(index=False))

print("\n=== Top layers by simple PC1/PC2 axis alignment ===")
print(df.sort_values("simple_axis_alignment_mean_abs_pearson", ascending=False)[cols].head(12).round(4).to_string(index=False))

print("\n=== Top layers by PC1↔Pleasure ===")
print(df.sort_values("pc1_pleasure_pearson_oriented", ascending=False)[cols].head(12).round(4).to_string(index=False))

print("\n=== Top layers by PC2↔Arousal ===")
print(df.sort_values("pc2_arousal_pearson_oriented", ascending=False)[cols].head(12).round(4).to_string(index=False))

print("\n=== Top layers by joint PC1+PC2 LOOCV affect alignment ===")
print(df.sort_values("pc12_loocv_affect_alignment_mean_pearson", ascending=False)[cols].head(12).round(4).to_string(index=False))

print("\n=== Band summary ===")
print(band_df.round(4).to_string(index=False))

print("\nWrote:")
for p in [
    OUT / "layerwise_external_affect_alignment.csv",
    OUT / "layerwise_external_affect_alignment_band_summary.csv",
    OUT / "layerwise_external_affect_alignment_matched_emotions.csv",
    OUT / "layerwise_external_affect_alignment_summary.json",
]:
    print(" ", p)
