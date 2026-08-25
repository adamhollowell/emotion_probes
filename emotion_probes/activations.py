"""
Activation extraction: residual stream activations per layer per token.

For each story, run a forward pass, capture residual stream at every layer
(``hook_resid_post``), then mean-pool over token positions from
``skip_first_n_tokens`` to end. One vector per (story, layer).

Built against transformer_lens 3.x ``TransformerBridge``. The ``Config``
flag ``tl_compatibility_mode`` chooses between raw HF numerics and folded
LayerNorm / centered weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformer_lens.model_bridge import TransformerBridge

from emotion_probes.config import Config
from emotion_probes.generation import Story
from emotion_probes.utils import device_from_config, dtype_from_config


# ---- Data structures ----

@dataclass
class ActivationBatch:
    """
    Activations for one (emotion, variant) group, saved as one file.

    Shape: (n_stories, n_layers, d_model).
    """
    emotion: str
    variant: str
    activations: torch.Tensor  # (n_stories, n_layers, d_model)
    story_indices: list[int]  # indices into the input list


# ---- Public API ----

def load_model(config: Config) -> TransformerBridge:
    """
    Boot a ``TransformerBridge`` per config and (optionally) enable the
    legacy HookedTransformer-equivalent numerics regime.
    """
    device = device_from_config(config)
    dtype = dtype_from_config(config)
    bridge = TransformerBridge.boot_transformers(
        config.model_name,
        device=device,
        dtype=dtype,
    )
    if config.tl_compatibility_mode:
        bridge.enable_compatibility_mode(disable_warnings=True)
    bridge.eval()
    return bridge


def extract_activations_for_stories(
    stories: list[Story],
    model: TransformerBridge,
    config: Config,
) -> torch.Tensor:
    """
    Forward each story, mean-pool ``hook_resid_post`` over the trailing
    tokens (from ``config.skip_first_n_tokens`` to end), and stack.

    Returns shape (n_stories, n_layers, d_model) on CPU as float32.
    """
    n_layers = model.cfg.n_layers
    resid_hooks = _residual_stream_hook_names(model)
    out = torch.empty(len(stories), n_layers, model.cfg.d_model, dtype=torch.float32)

    pbar = tqdm(stories, desc="extract", leave=False)
    for i, story in enumerate(pbar):
        out[i] = _mean_resid_post_single(
            text=story.text,
            model=model,
            resid_hooks=resid_hooks,
            skip_first_n=config.skip_first_n_tokens,
        )
    return out


def extract_all_activations(
    stories: list[Story],
    model: TransformerBridge,
    config: Config,
) -> Path:
    """
    Extract activations for all stories, save one .pt file per
    (emotion, variant). Resumable — groups already on disk are skipped.

    Returns ``config.activations_dir``.
    """
    out_dir = config.activations_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bucket stories by (emotion, variant), tracking their original index.
    groups: dict[tuple[str, str], list[tuple[int, Story]]] = {}
    for idx, s in enumerate(stories):
        groups.setdefault((s.emotion, s.variant), []).append((idx, s))

    for (emotion, variant), members in groups.items():
        path = _batch_path(out_dir, emotion, variant)
        if path.exists():
            continue
        indices = [i for i, _ in members]
        member_stories = [s for _, s in members]
        acts = extract_activations_for_stories(member_stories, model, config)
        batch = ActivationBatch(
            emotion=emotion,
            variant=variant,
            activations=acts,
            story_indices=indices,
        )
        _save_batch(batch, path)

    return out_dir


def extract_activations_at_position(
    texts: list[str],
    target_token: str,
    model: TransformerBridge,
    config: Config,
) -> torch.Tensor:
    """
    Grab residual stream activations at the LAST occurrence of
    ``target_token`` (the rightmost match in the token stream).

    Used for the "activation at the Assistant colon" analyses: pass a
    chat-formatted prompt and ``target_token=":"``.

    Returns (n_texts, n_layers, d_model). Raises ``ValueError`` if
    ``target_token`` does not tokenize to a single token, or if the token
    does not appear in a given text.
    """
    tokenizer = model.tokenizer
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(target_ids) != 1:
        raise ValueError(
            f"target_token {target_token!r} tokenizes to {len(target_ids)} "
            "tokens; pass something that maps to exactly one."
        )
    target_id = target_ids[0]
    resid_hooks = _residual_stream_hook_names(model)
    n_layers = model.cfg.n_layers
    out = torch.empty(len(texts), n_layers, model.cfg.d_model, dtype=torch.float32)

    for i, text in enumerate(texts):
        tokens = model.to_tokens(text)  # (1, seq)
        matches = (tokens[0] == target_id).nonzero(as_tuple=True)[0]
        if matches.numel() == 0:
            raise ValueError(
                f"target token id {target_id} ({target_token!r}) not found "
                f"in text {i}"
            )
        pos = int(matches[-1].item())
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=resid_hooks)
        for layer in range(n_layers):
            out[i, layer] = cache[resid_hooks[layer]][0, pos].float().cpu()
    return out


def load_activation_batch(
    emotion: str,
    variant: str,
    config: Config,
) -> ActivationBatch:
    """Load one saved (emotion, variant) group from disk."""
    path = _batch_path(config.activations_dir, emotion, variant)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return ActivationBatch(**payload)


# ---- Internals ----

def _residual_stream_hook_names(model: TransformerBridge) -> list[str]:
    """Hook names for ``hook_resid_post`` at each layer, in layer order."""
    return [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]


def _mean_resid_post_single(
    text: str,
    model: TransformerBridge,
    resid_hooks: list[str],
    skip_first_n: int,
) -> torch.Tensor:
    """Run one forward, mean-pool resid_post from skip_first_n to end."""
    tokens = model.to_tokens(text)  # (1, seq)
    seq_len = tokens.shape[1]
    # If the story is too short, fall back to averaging over whatever is
    # past the BOS. Anthropic's stories all comfortably exceed 50 tokens,
    # but tiny dev runs sometimes won't.
    start = skip_first_n if seq_len > skip_first_n + 1 else min(1, seq_len - 1)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=resid_hooks)
    layer_means = []
    for name in resid_hooks:
        # cache[name]: (1, seq, d_model)
        layer_means.append(cache[name][0, start:].mean(dim=0).float().cpu())
    return torch.stack(layer_means, dim=0)  # (n_layers, d_model)


def _batch_path(out_dir: Path, emotion: str, variant: str) -> Path:
    return out_dir / f"{emotion}__{variant}.pt"


def _save_batch(batch: ActivationBatch, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "emotion": batch.emotion,
            "variant": batch.variant,
            "activations": batch.activations,
            "story_indices": batch.story_indices,
        },
        tmp,
    )
    tmp.rename(path)  # atomic on POSIX — no half-written file on crash
