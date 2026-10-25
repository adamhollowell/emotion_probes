#!/usr/bin/env python3
"""C3 embedding re-base — neutral-dialogue token IDs (handoff §2).

Tokenizes the 500 "Human:/Assistant:" neutral dialogues once (family-wide
tokenizer) and saves raw token IDs for the C3 embedding re-base. Conventions
copied EXACTLY from gemma_grid/scripts/c3_tokenize_corpus.py:
  * AutoTokenizer google/gemma-2-27b, add_special_tokens=True (BOS-prepended,
    matches model.to_tokens); BOS asserted on every text.
  * RAW ids saved — pooling (tokens[50:], short-text fallback min(1,n-1)) is
    applied downstream, one source of truth in the analysis code.

Needs only the tokenizer (no GPU). Output: gemma_grid/data/c3_tokens_neutral.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ID = "google/gemma-2-27b"     # same tokenizer family-wide; mirror c3_tokenize_corpus
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "stories" / "neutral_dialogues.jsonl"  # parent repo corpus, sha256 3838a8f2
OUT = ROOT / "gemma_grid" / "data" / "c3_tokens_neutral.npz"


def main():
    import hashlib
    from transformers import AutoTokenizer

    src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    tok = AutoTokenizer.from_pretrained(REPO_ID)
    bos = tok.bos_token_id
    print("tokenizer self-check:", tok.convert_ids_to_tokens(
        tok("Human: hello\n\nAssistant: hi")["input_ids"][:8]))

    texts, labels = [], []
    with open(SRC) as f:
        for line in f:
            if not line.strip():
                continue
            texts.append(json.loads(line)["text"])
            labels.append("__neutral_100T__")
    n_neutral = len(texts)
    print(f"{n_neutral} neutral dialogues (src sha256 {src_sha[:16]})")
    assert n_neutral == 500, f"expected 500 neutral dialogues, got {n_neutral} — STOP"

    ids_all, offsets = [], [0]
    enc = tok(texts, add_special_tokens=True)["input_ids"]
    for i, ids in enumerate(enc):
        if not ids or ids[0] != bos:
            sys.exit(f"text {i}: first token != BOS (to_tokens mismatch) — STOP")
        ids_all.extend(ids)
        offsets.append(len(ids_all))
    ids_all = np.asarray(ids_all, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int64)
    lens = np.diff(offsets)
    print(f"total tokens {len(ids_all):,} | len min/median/max "
          f"{lens.min()}/{int(np.median(lens))}/{lens.max()}")
    short = int((lens <= 51).sum())
    print(f"dialogues at/below 51 tokens (pooling fallback would trigger): {short} "
          f"(handoff expects ~37)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, ids=ids_all, offsets=offsets, labels=np.asarray(labels),
        n_neutral=n_neutral, bos_token_id=bos,
        provenance=json.dumps({
            "tokenizer_repo": REPO_ID,
            "tokenizer_class": type(tok).__name__,
            "add_special_tokens": True,
            "src": SRC.name,
            "src_sha256": src_sha,
            "note": "raw ids incl. BOS; pooling (tokens[50:], fallback min(1,n-1)) applied downstream",
        }))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1e6:.1f} MB) — done.")


if __name__ == "__main__":
    main()
