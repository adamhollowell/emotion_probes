#!/usr/bin/env python3
"""
extract_embedding_null.py — bag-of-embeddings null (lightweight, no forward pass).

Tests whether Gemma-2-27B's affective circumplex is already latent in the STATIC token
embeddings, or is produced by contextual depth. This fills the genuine gap BELOW L0:
in emotion_vectors_171.pt, "L0" is blocks.0.hook_resid_post, which already contains block-0
attention + MLP. The embedding layer (embed_tokens) is strictly shallower.

Design (per external review):
  * Embedding table read DIRECTLY from google/gemma-2-27b via safetensors
    (model.embed_tokens.weight) — no tie assumption, no fold_ln assumption, no W_U.
    Downloads only the safetensors index + the ONE shard holding embed_tokens.weight;
    the 46-block model is never instantiated.
  * The difference-of-means (and optional cleanup) are the CANONICAL functions imported
    from emotion_probes.probes. transformer_lens is stubbed ONLY if not installed, so the
    package loads without the 27B/GPU dep. Nothing is reimplemented -> the math cannot
    drift from emotion_vectors_171.pt (it is the same code).
  * PRIMARY output is RAW diff-of-means (embedding-raw vs L31-raw): no cleanup, no neutral
    PCs, so there is no way to mix layers in the cleanup, and neutral.pt is not needed.
    A CLEANED variant is also written; its PCs are computed from the neutral dialogues'
    OWN embeddings (never carried over from L31).
  * Tokenization mirrors model.to_tokens (BOS prepended; skip_first_n=50 trailing pool)
    and self-checks the first tokens inline.

Run from project ROOT on the Mac (needs: transformers, safetensors, huggingface_hub, torch;
HF gate for google/gemma-2-27b already accepted):
    python3 scripts/extract_embedding_null.py

Output: data/vectors/embedding_null_vectors.pt
    {"emotions":[...171...], "vectors":(171,1,4608) cleaned, "vectors_raw":(171,1,4608) raw,
     "layer":"embed_tokens", "provenance":{...}}
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

import importlib.abc
import importlib.machinery
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import torch

REPO_ID = "google/gemma-2-27b"


# --- Reuse the CANONICAL math from emotion_probes.probes ---------------------------------
# Use the real transformer_lens if it's importable; otherwise fabricate a stub for any
# `transformer_lens*` module so the package imports (the functions we call are pure torch
# and never touch TransformerBridge). This guarantees zero reimplementation of the math.
def _install_tl_stub() -> None:
    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__file__ = "<transformer_lens stub>"   # real string so inspect.getsourcefile is happy
            m.__path__ = []                            # act as a package (submodule imports resolve)

            def _getattr(attr):
                # Let dunders raise normally so inspect / import machinery walk sys.modules
                # without tripping over fabricated attributes; fabricate only real names
                # actually imported by the package (e.g. TransformerBridge).
                if attr.startswith("__") and attr.endswith("__"):
                    raise AttributeError(attr)
                return type(attr, (object,), {})

            m.__getattr__ = _getattr
            return m

        def exec_module(self, module):  # noqa: D401
            pass

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "transformer_lens" or name.startswith("transformer_lens."):
                return importlib.machinery.ModuleSpec(name, _Loader())
            return None

    sys.meta_path.insert(0, _Finder())


try:
    from emotion_probes.probes import difference_of_means, project_out_top_pcs
except ModuleNotFoundError as exc:
    if "transformer_lens" not in str(exc):
        raise
    _install_tl_stub()
    from emotion_probes.probes import difference_of_means, project_out_top_pcs  # noqa: E402

from emotion_probes.config import Config           # noqa: E402  (pure config, no heavy deps)
from emotion_probes.generation import load_stories  # noqa: E402  (json loader)

CFG = Config()
SKIP = CFG.skip_first_n_tokens        # 50 — identical to the L31 extraction
TARGET = CFG.pca_variance_target      # 0.50


def load_embedding_table() -> torch.Tensor:
    """Read model.embed_tokens.weight (vocab, d_model) from the single shard that holds it."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index = json.load(open(hf_hub_download(REPO_ID, "model.safetensors.index.json")))
    shard = index["weight_map"]["model.embed_tokens.weight"]
    with safe_open(hf_hub_download(REPO_ID, shard), framework="pt") as f:
        W = f.get_tensor("model.embed_tokens.weight")
    return W.float()


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO_ID)
    W_E = load_embedding_table()
    print(f"embed_tokens.weight {tuple(W_E.shape)}  skip_first_n={SKIP}  cleanup_target={TARGET}")

    def pooled(text: str) -> torch.Tensor:
        ids = tok(text, add_special_tokens=True)["input_ids"]  # BOS-prepended, like to_tokens
        assert ids and ids[0] == tok.bos_token_id, "first token != BOS (to_tokens mismatch)"
        n = len(ids)
        start = SKIP if n > SKIP + 1 else min(1, n - 1)         # identical start/BOS rule
        return W_E[torch.tensor(ids[start:], dtype=torch.long)].mean(dim=0)  # (d_model,)

    # tokenization self-check (eyeball: should be ['<bos>', 'The', ' rain', ...])
    demo = "The rain fell softly on the quiet street as she walked home alone."
    print("tokenizer self-check:", tok.convert_ids_to_tokens(tok(demo)["input_ids"][:8]))

    stories = [
        s for s in load_stories(Path("data/stories/stories_171_canonical_clean.jsonl"))
        if getattr(s, "variant", "canonical") == "canonical"
    ]
    groups: dict[str, list] = defaultdict(list)
    for s in stories:
        groups[s.emotion].append(s)
    print(f"emotion stories: {len(stories)} across {len(groups)} emotions")

    # per-emotion embedding activations, shape (n_stories, 1, d_model)  [n_layers = 1]
    per_emotion: dict[str, torch.Tensor] = {}
    for e in sorted(groups):
        per_emotion[e] = torch.stack([pooled(s.text) for s in groups[e]]).unsqueeze(1)
        print(f"    {e}: {per_emotion[e].shape[0]} stories")

    neutrals = load_stories(Path("data/stories/neutral_dialogues.jsonl"))
    neutral_acts = torch.stack([pooled(s.text) for s in neutrals]).unsqueeze(1)  # (N,1,d)
    print(f"neutral dialogues: {neutral_acts.shape[0]}")

    # CANONICAL math (imported, not reimplemented)
    raw, emotions = difference_of_means(per_emotion)             # (171, 1, d) — PRIMARY
    cleaned = project_out_top_pcs(raw, neutral_acts, TARGET)     # neutral-EMBEDDING PCs

    out = {
        "emotions": list(emotions),
        "vectors": cleaned,          # cleaned (neutral-embedding PCs projected out)
        "vectors_raw": raw,          # RAW diff-of-means — the assumption-free primary
        "layer": "embed_tokens (static input embedding; no forward pass)",
        "provenance": {
            "embedding_source": "google/gemma-2-27b model.embed_tokens.weight via safetensors "
                                "(no tie/fold assumption; full model never loaded)",
            "tokenization": "AutoTokenizer, BOS-prepended (matches model.to_tokens)",
            "pooling": f"mean over tokens[{SKIP}:] (skip_first_n_tokens), identical start/BOS rule",
            "math": "emotion_probes.probes.difference_of_means / project_out_top_pcs (canonical, unmodified)",
            "cleanup": f"raw is primary; cleaned projects out neutral-EMBEDDING PCs to {TARGET} variance",
            "identical_to_vectors_171_except": "activation source (embed_tokens vs blocks.31.hook_resid_post)",
        },
    }
    out_path = Path("data/vectors/embedding_null_vectors.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"wrote {out_path}  raw={tuple(raw.shape)}  cleaned={tuple(cleaned.shape)}")


if __name__ == "__main__":
    main()
