"""
Probe computation: turn raw activations into emotion vectors.

The math (per layer):
    1. For each emotion e, mu_e = mean activation across that emotion's stories.
    2. mu_all = mean of mu_e across emotions (unweighted; one vote per emotion).
    3. Raw emotion vector: v_e = mu_e - mu_all.
    4. PCA cleanup: take activations on neutral text, compute top-k principal
       components where k is the smallest count whose cumulative explained
       variance >= config.pca_variance_target. Project those k components out
       of each v_e.
    5. Output: one (PCA-cleaned and raw) vector per (emotion, layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from emotion_probes.activations import ActivationBatch, load_activation_batch
from emotion_probes.config import Config
from emotion_probes.emotions import EMOTIONS_171


# ---- Data structures ----

@dataclass
class EmotionVectors:
    """
    The full set of emotion vectors for one variant.

    ``vectors`` is the PCA-cleaned set; ``vectors_raw`` is before cleanup.
    Shape for both: (n_emotions, n_layers, d_model).
    """
    variant: str
    emotions: list[str]
    vectors: torch.Tensor
    vectors_raw: torch.Tensor
    n_layers: int
    d_model: int

    def get(self, emotion: str, layer: int, raw: bool = False) -> torch.Tensor:
        """One vector. Shape: (d_model,)."""
        idx = self.emotions.index(emotion)
        source = self.vectors_raw if raw else self.vectors
        return source[idx, layer]


# ---- Public API ----

def compute_emotion_vectors(
    activations_dir: Path,
    neutral_activations: torch.Tensor,
    config: Config,
    variant: str = "canonical",
) -> EmotionVectors:
    """
    Build EmotionVectors from saved per-emotion activation files.

    Loads every ``{emotion}__{variant}.pt`` under ``activations_dir`` whose
    emotion is in ``config.emotion_subset`` (or the full 171 if subset is
    None). Computes raw diff-of-means, then projects out the top PCs of
    ``neutral_activations`` per layer.

    ``neutral_activations`` shape: (n_neutral_stories, n_layers, d_model).
    Must have the same n_layers / d_model as the saved activations.
    """
    requested = config.emotion_subset or EMOTIONS_171
    per_emotion: dict[str, torch.Tensor] = {}
    for emotion in requested:
        path = activations_dir / f"{emotion}__{variant}.pt"
        if not path.exists():
            continue
        batch = load_activation_batch(emotion, variant, config)
        per_emotion[emotion] = batch.activations

    if not per_emotion:
        raise FileNotFoundError(
            f"No activation files found in {activations_dir} for variant "
            f"{variant!r}; expected {{emotion}}__{variant}.pt files."
        )

    raw_vectors, emotions = difference_of_means(per_emotion)
    n_layers, d_model = raw_vectors.shape[1], raw_vectors.shape[2]

    if neutral_activations.shape[1:] != (n_layers, d_model):
        raise ValueError(
            f"neutral_activations shape {tuple(neutral_activations.shape)} "
            f"incompatible with emotion activations (n_layers={n_layers}, "
            f"d_model={d_model})"
        )

    cleaned = project_out_top_pcs(
        raw_vectors, neutral_activations, config.pca_variance_target,
    )
    return EmotionVectors(
        variant=variant,
        emotions=emotions,
        vectors=cleaned,
        vectors_raw=raw_vectors,
        n_layers=n_layers,
        d_model=d_model,
    )


def difference_of_means(
    per_emotion_activations: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[str]]:
    """
    Compute raw diff-of-means vectors per emotion per layer.

    Procedure per layer (vectorized across layers):
        mu_e   = mean over stories of activations for emotion e
        mu_all = mean of mu_e across emotions   (unweighted)
        v_e    = mu_e - mu_all

    Returns:
        vectors: (n_emotions, n_layers, d_model)
        emotions: ordered list giving the row labels of ``vectors``.

    The unweighted across-emotion mean matches Anthropic's description
    ("mean activation across different emotions") and prevents emotions
    with more stories from dragging the centroid.
    """
    if not per_emotion_activations:
        raise ValueError("per_emotion_activations is empty")
    emotions = list(per_emotion_activations.keys())
    per_emotion_means = torch.stack(
        [per_emotion_activations[e].mean(dim=0).float() for e in emotions],
        dim=0,
    )  # (n_emotions, n_layers, d_model)
    grand_mean = per_emotion_means.mean(dim=0, keepdim=True)  # (1, n_layers, d_model)
    return per_emotion_means - grand_mean, emotions


def project_out_top_pcs(
    vectors: torch.Tensor,
    neutral_activations: torch.Tensor,
    variance_target: float,
) -> torch.Tensor:
    """
    PCA cleanup per layer.

    For each layer L:
        1. Center neutral_activations[:, L, :] by subtracting its mean.
        2. SVD; treat singular values squared / (n-1) as variance per PC.
        3. Find smallest k whose cumulative variance ratio >= variance_target.
        4. With V_k the top-k right singular vectors (orthonormal columns,
           shape (d_model, k)), set v <- v - V_k @ (V_k.T @ v) for each
           emotion vector at layer L.

    Args:
        vectors: (n_emotions, n_layers, d_model) — raw diff-of-means.
        neutral_activations: (n_neutral, n_layers, d_model).
        variance_target: in (0, 1).

    Returns: tensor of the same shape as ``vectors``, cleaned.
    """
    if not 0.0 < variance_target < 1.0:
        raise ValueError(
            f"variance_target must be in (0, 1); got {variance_target}"
        )
    n_emotions, n_layers, d_model = vectors.shape
    assert neutral_activations.shape[1] == n_layers
    assert neutral_activations.shape[2] == d_model

    cleaned = vectors.clone().float()
    neutral = neutral_activations.float()
    n_neutral = neutral.shape[0]

    for layer in range(n_layers):
        X = neutral[:, layer, :]  # (n_neutral, d_model)
        X_centered = X - X.mean(dim=0, keepdim=True)
        # economy SVD: U (n, r), S (r,), Vh (r, d_model); r = min(n, d_model)
        _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        # Variance explained per PC. n_neutral-1 normalization cancels in the
        # ratio, but include it so explained_variance numerics match sklearn.
        denom = max(n_neutral - 1, 1)
        var = (S ** 2) / denom
        ratio = var / var.sum()
        k = int((torch.cumsum(ratio, dim=0) >= variance_target).nonzero()[0].item()) + 1
        V_k = Vh[:k].T  # (d_model, k), orthonormal columns
        # Project out: v <- v - V_k @ (V_k.T @ v)
        coefs = cleaned[:, layer, :] @ V_k  # (n_emotions, k)
        cleaned[:, layer, :] = cleaned[:, layer, :] - coefs @ V_k.T

    return cleaned


def cosine_similarities(vectors: EmotionVectors, layer: int) -> torch.Tensor:
    """
    Pairwise cosine similarities between emotion vectors at one layer.

    Returns (n_emotions, n_emotions) symmetric with unit diagonal.
    """
    V = vectors.vectors[:, layer, :].float()  # (n_emotions, d_model)
    norms = V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    Vn = V / norms
    return Vn @ Vn.T


def emotion_pca(
    vectors: EmotionVectors,
    layer: int,
    n_components: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PCA across the emotion vectors themselves at one layer.

    The first two PCs should approximately recover valence and arousal
    (the affective circumplex from psychology) — see Anthropic Fig 7-8.

    Returns:
        coords: (n_emotions, n_components) — projections of each emotion
            vector onto the top components.
        explained_variance_ratio: (n_components,)
    """
    V = vectors.vectors[:, layer, :].float()  # (n_emotions, d_model)
    V_centered = V - V.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(V_centered, full_matrices=False)
    coords = U[:, :n_components] * S[:n_components]
    total = (S ** 2).sum().clamp_min(1e-12)
    ratio = (S[:n_components] ** 2) / total
    return coords, ratio


# ---- Saving / loading ----

def save_vectors(vectors: EmotionVectors, config: Config) -> Path:
    """Save EmotionVectors to ``{config.vectors_dir}/{variant}.pt``."""
    path = config.vectors_dir / f"{vectors.variant}.pt"
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "variant": vectors.variant,
            "emotions": vectors.emotions,
            "vectors": vectors.vectors,
            "vectors_raw": vectors.vectors_raw,
            "n_layers": vectors.n_layers,
            "d_model": vectors.d_model,
        },
        tmp,
    )
    tmp.rename(path)
    return path


def load_vectors(variant: str, config: Config) -> EmotionVectors:
    """Load EmotionVectors from disk."""
    path = config.vectors_dir / f"{variant}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return EmotionVectors(**payload)
