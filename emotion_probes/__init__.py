"""
emotion_probes
==============

A replication of Anthropic's "Emotion Vectors" methodology on open models.

Pipeline:
    1. generate_stories  -> labeled emotion-laden text
    2. extract_activations -> residual stream activations per layer
    3. compute_vectors    -> emotion vectors via difference-of-means + PCA cleanup
    4. validate           -> logit lens, top-activating examples
    5. steer              -> causal experiments

Designed so that the same machinery handles:
    - The canonical story-based bare vectors (replicating Anthropic)
    - Object-tagged variants (e.g. loving-toward-X)
    - Dialogue-based present-/other-speaker probes
"""

from emotion_probes.config import Config

__version__ = "0.1.0"
__all__ = ["Config"]
