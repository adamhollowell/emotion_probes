"""
Shared utilities used across modules.
"""

from __future__ import annotations

import torch

from emotion_probes.config import Config


def device_from_config(config: Config) -> torch.device:
    """Resolve config.device to a torch.device, with sane fallbacks."""
    if config.device == "cuda" and not torch.cuda.is_available():
        # Fall back to CPU rather than crashing; useful when developing
        # the package code on a machine without CUDA.
        return torch.device("cpu")
    if config.device == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(config.device)


def dtype_from_config(config: Config) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.dtype]


def primary_layer(config: Config, n_layers: int) -> int:
    """
    Resolve the 'about two-thirds of the way through' layer.

    If config.primary_layer is set, use it. Otherwise default to
    round(2/3 * n_layers).
    """
    if config.primary_layer is not None:
        return config.primary_layer
    return round(2 / 3 * n_layers)
