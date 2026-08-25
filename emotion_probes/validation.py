"""
Validation experiments to check that the emotion vectors mean what we
think they mean.

The three core checks from the paper:
    1. Logit lens: project each emotion vector through W_U and look at
       top tokens. Should be emotion-semantically related.
    2. Top-activating examples on a diverse held-out corpus: passages
       with the highest dot product against the vector should be
       emotion-laden.
    3. Geometry: PCA across emotion vectors should approximately recover
       valence (PC1) and arousal (PC2).

Numerics assumption: the model was loaded with ``tl_compatibility_mode``
on (default in this project). In that mode TransformerBridge folds the
final RMSNorm/LayerNorm into ``W_U``, so ``v @ W_U`` is the correct
logit-lens computation directly. With raw HF weights you'd need to
apply ``ln_final`` to ``v`` first or the logits will be misnormalized.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformer_lens.model_bridge import TransformerBridge

from emotion_probes.probes import EmotionVectors, emotion_pca


# ---- Data structures ----

@dataclass
class LogitLensResult:
    emotion: str
    layer: int
    top_tokens: list[tuple[str, float]]    # (token, logit) descending
    bottom_tokens: list[tuple[str, float]]  # (token, logit) ascending


@dataclass
class TopActivatingExample:
    """One peak of an emotion vector against a corpus text."""
    text_index: int           # index into the corpus_texts list
    peak_position: int        # token offset of the peak
    peak_activation: float    # dot product value at the peak
    window_text: str          # decoded window of tokens around the peak


@dataclass
class TopActivatingExamples:
    emotion: str
    layer: int
    examples: list[TopActivatingExample]  # descending by peak_activation


# ---- Public API ----

def logit_lens(
    vectors: EmotionVectors,
    model: TransformerBridge,
    layer: int,
    top_k: int = 10,
) -> list[LogitLensResult]:
    """
    Project each emotion vector at ``layer`` through the unembedding to
    get logits over the vocabulary, then return the top/bottom-k tokens.

    Reference (Anthropic Table 1): "happy" -> excited, excitement,
    exciting, ...; "angry" -> anger, angry, rage, fury, ...
    Nonsense top tokens almost always indicate something is wrong
    upstream (wrong layer, wrong numerics regime, vectors on a different
    device or dtype than W_U).
    """
    W_U = model.W_U  # (d_model, vocab_size); ln_final folded in via compat mode
    tokenizer = model.tokenizer
    results: list[LogitLensResult] = []
    with torch.no_grad():
        for emotion in vectors.emotions:
            v = vectors.get(emotion, layer).to(W_U.device).to(W_U.dtype)
            logits = v @ W_U  # (vocab_size,)
            top_vals, top_ids = logits.topk(top_k)
            bot_vals, bot_ids = logits.topk(top_k, largest=False)
            top = [
                (tokenizer.decode([int(i)]), float(v))
                for v, i in zip(top_vals, top_ids)
            ]
            bot = [
                (tokenizer.decode([int(i)]), float(v))
                for v, i in zip(bot_vals, bot_ids)
            ]
            results.append(
                LogitLensResult(emotion=emotion, layer=layer, top_tokens=top, bottom_tokens=bot)
            )
    return results


def top_activating_examples(
    vectors: EmotionVectors,
    model: TransformerBridge,
    layer: int,
    corpus_texts: list[str],
    top_k: int = 10,
    window_tokens: int = 50,
    skip_first_n_positions: int = 1,
) -> list[TopActivatingExamples]:
    """
    For each emotion vector at ``layer``, find the corpus passages where
    the residual stream most aligns with that vector.

    Procedure:
        1. Forward each text, capture ``hook_resid_post`` at ``layer``.
        2. Dot product per position with each emotion vector, excluding
           the first ``skip_first_n_positions`` positions per text.
        3. Keep the global top-k peaks per emotion across all texts.
        4. Decode a window of ``window_tokens`` tokens centered on each peak.

    Heap-based so memory stays O(n_emotions * top_k) regardless of corpus
    size — fine for the per-position activations of a 14k-document corpus.

    ``skip_first_n_positions`` defaults to 1 to drop the BOS token. The
    BOS residual has a disproportionately large norm in most models
    (it's the only "context-free" position), so without this skip the
    Cauchy–Schwarz argument means BOS hits crowd out genuinely
    emotion-related peaks. Set to 0 to disable the skip.
    """
    if skip_first_n_positions < 0:
        raise ValueError(
            f"skip_first_n_positions must be >= 0; got {skip_first_n_positions}"
        )
    V = vectors.vectors[:, layer, :].float()  # (n_emotions, d_model)
    emotions = vectors.emotions
    resid_hook = f"blocks.{layer}.hook_resid_post"

    # heaps[i] is a min-heap of size <= top_k of
    # (activation, text_index, peak_position, tokens_tensor)
    heaps: list[list[tuple]] = [[] for _ in emotions]

    for text_idx, text in enumerate(tqdm(corpus_texts, desc="top-activating", leave=False)):
        tokens = model.to_tokens(text)  # (1, seq)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[resid_hook])
        resid = cache[resid_hook][0].float().cpu()  # (seq, d_model)
        seq_len = resid.shape[0]
        # Skip leading positions but always leave at least one valid index.
        start = min(skip_first_n_positions, max(0, seq_len - 1))
        # (seq - start, n_emotions)
        activations = resid[start:] @ V.T
        toks_cpu = tokens[0].cpu()
        for e_idx in range(len(emotions)):
            acts = activations[:, e_idx]
            n_pick = min(top_k, acts.shape[0])
            if n_pick == 0:
                continue
            top_v, top_pos_rel = acts.topk(n_pick)
            for val, pos_rel in zip(top_v.tolist(), top_pos_rel.tolist()):
                pos_abs = pos_rel + start
                # heapq compares tuples lexicographically; first elem is
                # the activation, ties broken by text_idx then position.
                entry = (val, text_idx, pos_abs, toks_cpu)
                if len(heaps[e_idx]) < top_k:
                    heapq.heappush(heaps[e_idx], entry)
                elif val > heaps[e_idx][0][0]:
                    heapq.heappushpop(heaps[e_idx], entry)

    half = max(1, window_tokens // 2)
    results: list[TopActivatingExamples] = []
    for e_idx, emotion in enumerate(emotions):
        examples: list[TopActivatingExample] = []
        # Sort heap entries by activation descending.
        sorted_entries = sorted(heaps[e_idx], key=lambda x: -x[0])
        for activation, text_idx, pos, toks in sorted_entries:
            start = max(0, pos - half)
            end = min(toks.shape[0], pos + half)
            window = model.tokenizer.decode(toks[start:end].tolist())
            examples.append(
                TopActivatingExample(
                    text_index=text_idx,
                    peak_position=pos,
                    peak_activation=activation,
                    window_text=window,
                )
            )
        results.append(
            TopActivatingExamples(emotion=emotion, layer=layer, examples=examples)
        )
    return results


def validate_geometry(vectors: EmotionVectors, layer: int) -> dict:
    """
    PCA across emotion vectors at one layer, with per-PC rankings of the
    emotions. The paper finds PC1 ~ valence, PC2 ~ arousal.

    Returns:
        {
            "coords": (n_emotions, 2) projected coordinates,
            "explained_variance_ratio": (2,),
            "pc1_ranking": [(emotion, score)] sorted descending,
            "pc2_ranking": [(emotion, score)] sorted descending,
        }

    Sanity check on the rankings: PC1 should split positive-valence
    emotions (happy, loving, calm, ...) from negative ones (sad, afraid,
    angry, ...). PC2 should split high-arousal (angry, surprised, ...)
    from low-arousal (calm, sad, ...).
    """
    coords, ratio = emotion_pca(vectors, layer, n_components=2)
    pc1 = sorted(
        zip(vectors.emotions, coords[:, 0].tolist()), key=lambda x: -x[1],
    )
    pc2 = sorted(
        zip(vectors.emotions, coords[:, 1].tolist()), key=lambda x: -x[1],
    )
    return {
        "coords": coords,
        "explained_variance_ratio": ratio,
        "pc1_ranking": pc1,
        "pc2_ranking": pc2,
    }
