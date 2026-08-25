#!/usr/bin/env python3
"""
tokenize_manifest_v2.py -- Gemma-tokenize the rebuilt corpus, locally (Mac, py3.9+).

Produces data/heldout/token_manifest_v2.json with:
  - per-document Gemma token counts (includes BOS, matching model.to_tokens)
  - per-source document and token totals  -> the P6.2 baselines
  - cap cost table: tokens processed under caps 1024/2048/4096/8192/uncapped,
    and docs truncated by each                -> exact data for the Decision-3 revisit

Needs only the Gemma tokenizer (small download), not the model. HF_TOKEN required
because the tokenizer lives in the gated google/gemma-2-27b repo.

Usage (from project root):
    export HF_TOKEN=hf_...
    pip3 install --user transformers sentencepiece
    python3 scripts/tokenize_manifest_v2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

CAPS = [1024, 2048, 4096, 8192]


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: export HF_TOKEN=hf_... first (Gemma tokenizer is in a gated repo).")

    root = Path.cwd()
    corpus_path = root / "data" / "heldout" / "heldout_corpus_v2.jsonl"
    out_path = root / "data" / "heldout" / "token_manifest_v2.json"
    if not corpus_path.exists():
        sys.exit("ERROR: missing %s (run the builder first)" % corpus_path)
    if out_path.exists():
        sys.exit("ERROR: %s already exists. Refusing to overwrite." % out_path)

    from transformers import AutoTokenizer
    print("Loading Gemma tokenizer (gated repo; token in use)...")
    tok = AutoTokenizer.from_pretrained("google/gemma-2-27b", token=token)

    per_doc = {}
    per_source_docs = {}
    per_source_tokens = {}
    t0 = time.time()
    n = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids = tok(rec["text"], add_special_tokens=True).input_ids  # includes BOS
            cnt = len(ids)
            per_doc[rec["doc_id"]] = cnt
            src = rec["source"]
            per_source_docs[src] = per_source_docs.get(src, 0) + 1
            per_source_tokens[src] = per_source_tokens.get(src, 0) + cnt
            n += 1
            if n % 2000 == 0:
                print("  %6d docs (%.0f docs/s)" % (n, n / max(time.time() - t0, 1e-9)))

    total_tokens = sum(per_doc.values())
    cap_table = {}
    for cap in CAPS:
        processed = sum(min(c, cap) for c in per_doc.values())
        truncated = sum(1 for c in per_doc.values() if c > cap)
        cap_table[str(cap)] = {
            "tokens_processed": processed,
            "docs_truncated": truncated,
            "fraction_of_uncapped_tokens": round(processed / total_tokens, 4),
        }

    manifest = {
        "tokenizer": "google/gemma-2-27b (AutoTokenizer, add_special_tokens=True; count includes BOS)",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_docs": n,
        "total_tokens_uncapped": total_tokens,
        "per_source_docs": per_source_docs,
        "per_source_tokens": per_source_tokens,
        "cap_table": cap_table,
        "per_doc": per_doc,
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.rename(out_path)

    print("\n=== token manifest summary ===")
    print("docs: %d   uncapped tokens: %d" % (n, total_tokens))
    for src in sorted(per_source_docs):
        print("  %-14s docs=%6d  tokens=%10d" % (src, per_source_docs[src], per_source_tokens[src]))
    print("cap table:")
    for cap in CAPS:
        c = cap_table[str(cap)]
        print("  cap %5d: %11d tokens processed (%.1f%% of uncapped), %5d docs truncated"
              % (cap, c["tokens_processed"], 100 * c["fraction_of_uncapped_tokens"], c["docs_truncated"]))
    print("\nwrote %s" % out_path)
    print("DECISION-3 REVISIT: compare cap 2048 vs 4096 rows above; paste this summary back to Claude.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
