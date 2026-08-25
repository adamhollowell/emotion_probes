#!/usr/bin/env python3
"""Deep audit of heldout_corpus_v2.jsonl: counts, ids, lengths, samples.
Run from project root: python3 scripts/audit_corpus_v2.py"""
import json
import statistics

PATH = "data/heldout/heldout_corpus_v2.jsonl"
counts, lengths, ids = {}, {}, set()
bad = empty = 0
with open(PATH, encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        s = r["source"]
        counts[s] = counts.get(s, 0) + 1
        lengths.setdefault(s, []).append(len(r["text"]))
        ids.add(r["doc_id"])
        if not r["text"].strip():
            empty += 1

total = sum(counts.values())
print("total records:", total, "| unique doc_ids:", len(ids),
      "| bad json:", bad, "| empty texts:", empty)
for s in ("lmsys", "pile", "common_corpus", "isotonic"):
    L = lengths[s]
    at_cap = sum(1 for x in L if x >= 32000)
    print("%-14s n=%6d  chars: min=%d median=%d mean=%d max=%d  at_32k_cap=%d"
          % (s, counts[s], min(L), int(statistics.median(L)),
             int(statistics.mean(L)), max(L), at_cap))

seen = set()
with open(PATH, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["source"] not in seen:
            seen.add(r["source"])
            print("\n--- %s / %s ---" % (r["source"], r["doc_id"]))
            print(r["text"][:250].replace("\n", " | "))
        if len(seen) == 4:
            break
print("\nAUDIT COMPLETE")
