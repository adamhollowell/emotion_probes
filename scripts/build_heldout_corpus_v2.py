#!/usr/bin/env python3
"""
build_heldout_corpus_v2.py -- rebuild the held-out evaluation corpus, locally.

Builds the held-out corpus locally (CPU, Python 3.9+) -- no GPU required. The
corpus is checksummed and written to disk before any extraction runs, so its SHA is
fixed independently of the GPU stage.

Sources and counts (Decision 1, Option A -- P6.2 window-mix proportions,
disclosed as chosen under the composition hypothesis):

    lmsys          31,532   lmsys/lmsys-chat-1m            (gated; HF_TOKEN)
    pile            9,124   monology/pile-uncopyrighted    (Decision 2 substitution)
    common_corpus   3,593   PleIAs/common_corpus
    isotonic        3,025   Isotonic/human_assistant_conversation
    total          47,274

Construction choices (all disclosed in corpus_manifest_v2.json):
  - Streaming with a seeded shuffle buffer (deterministic given seed + dataset
    revision). Seed 20260706.
  - Multilingual content kept; no text cleaning (matches Anthropic Fig-1 panels,
    which show French passages and OCR-corrupted text).
  - LMSYS/Isotonic conversations flattened to plain text, turns joined by blank
    lines, no role labels.
  - min 100 chars; stored text truncated at 32,000 chars (far beyond any
    2,048-token window; bounds file size and tokenizer time).
  - Exact-duplicate rejection (normalized-text hash).
  - DISJOINTNESS GATE: any candidate sharing >= 1 normalized 12-word shingle
    with stories_171_canonical_clean.jsonl is rejected and counted.

Output (atomic writes; refuses to overwrite):
    data/heldout/heldout_corpus_v2.jsonl   one record per line:
        {"doc_id": "lmsys_000000", "source": "lmsys", "text": "..."}
    data/heldout/corpus_manifest_v2.json   seeds, counts, rejections, sha256

Usage (from project root):
    export HF_TOKEN=hf_...
    python3 scripts/build_heldout_corpus_v2.py --preview   # inspect 2 samples/source, no writes
    python3 scripts/build_heldout_corpus_v2.py             # full build (approx 1-3 h, network-bound)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

BUILDER_VERSION = "2.0.0"
SEED = 20260706
SHINGLE_WORDS = 12
MIN_CHARS = 100
STORE_CHAR_CAP = 32_000
SHUFFLE_BUFFER = 10_000

TARGETS = [
    # (source_name, hf_dataset, count)
    ("lmsys", "lmsys/lmsys-chat-1m", 31_532),
    ("pile", "monology/pile-uncopyrighted", 9_124),
    ("common_corpus", "PleIAs/common_corpus", 3_593),
    ("isotonic", "Isotonic/human_assistant_conversation", 3_025),
]

WS = re.compile(r"\s+")


def normalize(text):
    return WS.sub(" ", text.lower()).strip()


def shingle_hashes(text, n=SHINGLE_WORDS):
    """8-byte blake2b hashes of every n-word shingle of normalized text."""
    words = normalize(text).split(" ")
    out = set()
    if len(words) < n:
        if words and words != [""]:
            joined = " ".join(words)
            out.add(int.from_bytes(hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest(), "big"))
        return out
    for i in range(len(words) - n + 1):
        joined = " ".join(words[i:i + n])
        out.add(int.from_bytes(hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest(), "big"))
    return out


def extract_text(source, row):
    """Flatten one dataset row to plain text. Returns None if unusable."""
    if source == "lmsys":
        conv = row.get("conversation")
        if not conv:
            return None
        parts = []
        for turn in conv:
            c = turn.get("content") if isinstance(turn, dict) else None
            if c:
                parts.append(c.strip())
        return "\n\n".join(parts) if parts else None
    if source in ("pile", "common_corpus"):
        t = row.get("text")
        return t if isinstance(t, str) else None
    if source == "isotonic":
        # Field names on this dataset have varied; try known pairs, then any strings.
        for a, b in (("prompt", "response"), ("human", "assistant"), ("input", "output")):
            if isinstance(row.get(a), str) and isinstance(row.get(b), str):
                return row[a].strip() + "\n\n" + row[b].strip()
        if isinstance(row.get("text"), str):
            return row["text"]
        strings = [v.strip() for v in row.values() if isinstance(v, str) and v.strip()]
        return "\n\n".join(strings) if strings else None
    raise ValueError("unknown source: " + source)


def load_story_shingles(stories_path):
    print("Loading stories for disjointness gate: %s" % stories_path)
    shingles = set()
    n = 0
    with open(stories_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            shingles |= shingle_hashes(rec["text"])
            n += 1
    print("  %d stories -> %d unique %d-word shingle hashes" % (n, len(shingles), SHINGLE_WORDS))
    return shingles


def stream_source(source, hf_name, token):
    from datasets import load_dataset
    ds = load_dataset(hf_name, split="train", streaming=True, token=token)
    return ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true",
                    help="print 2 extracted samples per source and exit; no writes")
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: export HF_TOKEN=hf_... first (lmsys is gated).")

    root = args.project_root
    stories_path = root / "data" / "stories" / "stories_171_canonical_clean.jsonl"
    out_dir = root / "data" / "heldout"
    out_path = out_dir / "heldout_corpus_v2.jsonl"
    manifest_path = out_dir / "corpus_manifest_v2.json"

    if args.preview:
        for source, hf_name, _count in TARGETS:
            print("\n=== %s (%s) ===" % (source, hf_name))
            try:
                stream = stream_source(source, hf_name, token)
                shown = 0
                for row in stream:
                    text = extract_text(source, row)
                    if text and len(text) >= MIN_CHARS:
                        print("--- sample %d (%d chars) ---" % (shown + 1, len(text)))
                        print(text[:400].replace("\n", " | "))
                        shown += 1
                    if shown == 2:
                        break
            except Exception as e:
                print("  PREVIEW FAILED: %r" % e)
                print("  (field names may differ; paste this error back to Claude)")
        print("\nPreview complete. No files written.")
        return 0

    if not stories_path.exists():
        sys.exit("ERROR: missing %s" % stories_path)
    if out_path.exists():
        sys.exit("ERROR: %s already exists. Refusing to overwrite. Move it aside first." % out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    story_shingles = load_story_shingles(stories_path)

    tmp_path = out_path.with_suffix(".jsonl.tmp")
    counts = {}
    rejections = {"gate": {}, "dupe": {}, "too_short": {}, "unusable": {}}
    seen_hashes = set()
    t_start = time.time()

    with open(tmp_path, "w", encoding="utf-8") as out:
        for source, hf_name, target_n in TARGETS:
            print("\n=== %s: target %d from %s ===" % (source, target_n, hf_name))
            for d in rejections.values():
                d[source] = 0
            accepted = 0
            examined = 0
            stream = stream_source(source, hf_name, token)
            t0 = time.time()
            for row in stream:
                examined += 1
                text = extract_text(source, row)
                if text is None:
                    rejections["unusable"][source] += 1
                elif len(text) < MIN_CHARS:
                    rejections["too_short"][source] += 1
                else:
                    text = text[:STORE_CHAR_CAP]
                    nh = hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).digest()
                    if nh in seen_hashes:
                        rejections["dupe"][source] += 1
                    elif shingle_hashes(text) & story_shingles:
                        rejections["gate"][source] += 1
                    else:
                        seen_hashes.add(nh)
                        rec = {"doc_id": "%s_%06d" % (source, accepted),
                               "source": source, "text": text}
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        accepted += 1
                        if accepted % 1000 == 0:
                            rate = accepted / max(time.time() - t0, 1e-9)
                            print("  %6d / %d accepted (%.0f docs/s, %d examined)"
                                  % (accepted, target_n, rate, examined))
                        if accepted == target_n:
                            break
            counts[source] = accepted
            print("  DONE %s: %d accepted, %d examined, gate=%d dupe=%d short=%d unusable=%d (%.1f min)"
                  % (source, accepted, examined, rejections["gate"][source],
                     rejections["dupe"][source], rejections["too_short"][source],
                     rejections["unusable"][source], (time.time() - t0) / 60))
            if accepted < target_n:
                print("  WARNING: stream exhausted before target; shortfall %d" % (target_n - accepted))

    tmp_path.rename(out_path)
    print("\nCorpus written: %s" % out_path)
    print("Computing SHA-256 (the fingerprint any rebuild must reproduce)...")
    digest = sha256_file(out_path)
    print("  sha256: %s" % digest)

    manifest = {
        "builder_version": BUILDER_VERSION,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "shuffle_buffer": SHUFFLE_BUFFER,
        "shingle_words": SHINGLE_WORDS,
        "min_chars": MIN_CHARS,
        "store_char_cap": STORE_CHAR_CAP,
        "sources": {s: {"hf_dataset": h, "target": n, "accepted": counts.get(s, 0)}
                    for s, h, n in TARGETS},
        "rejections": rejections,
        "total_accepted": sum(counts.values()),
        "disjointness_gate": "reject on >=1 shared normalized %d-word shingle with stories_171_canonical_clean.jsonl" % SHINGLE_WORDS,
        "disclosures": [
            "Proportions set to the P6.2 window mix under the composition hypothesis (Decision 1A).",
            "monology/pile-uncopyrighted substitutes for the original Pile (Decision 2A).",
            "Conversations flattened without role labels, turns joined by blank lines.",
            "Multilingual kept; no cleaning; stored text truncated at %d chars." % STORE_CHAR_CAP,
        ],
        "corpus_sha256": digest,
        "corpus_file": str(out_path.name),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Manifest written: %s" % manifest_path)
    print("\nTotal: %d docs in %.1f min. Next: checksum, then tokenize."
          % (sum(counts.values()), (time.time() - t_start) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
