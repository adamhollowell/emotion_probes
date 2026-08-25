#!/usr/bin/env python3
"""P6.1 successor: classify peak tokens of top-activating windows (v2 corpus).
Run from project root: python3 scripts/analyze_maxact_peaks.py
Classification rule is explicit below -- verify it matches the P6.1
operationalization before recording numbers.

Source: the raw windows (data/heldout/maxact/maxactivating_v2.json) carry third-party
window text and are NOT shipped. The derived geometry_out/peak_tokens.json (isolated
peak tokens + categories only; built by scripts/build_peak_tokens.py) IS shipped, and
this script reads it when the raw file is absent -- so the 52.6% rank-1 floor and the
48.4% top-20 rate reproduce from shipped files alone."""
import json
import os
import re
from collections import Counter

PATH = "data/heldout/maxact/maxactivating_v2.json"
DERIVED = "geometry_out/peak_tokens.json"

# ---- classification rule (edit here if the original rule differed) ----
FUNCTION_WORDS = set("""
a an the and or but if then else of to in on at by for with from as is are was
were be been being am do does did have has had will would can could shall
should may might must not no nor so than that this these those it its he she
they them his her their we you your i me my our us who whom which what when
where why how there here all any both each few more most other some such only
own same too very s t just don now
""".split())

RE_NUMERIC = re.compile(r"^\s*[\d.,:%/\-]+\s*$")
RE_PUNCT = re.compile(r"^\s*[^\w\s]+\s*$")


def classify(token):
    t = token.strip()
    if t == "":
        return "whitespace/newline"
    if RE_NUMERIC.match(t) and any(c.isdigit() for c in t):
        return "numeric"
    if RE_PUNCT.match(t):
        return "punctuation/symbol"
    if t.lower() in FUNCTION_WORDS:
        return "function_word"
    return "content"


def report(rank_slice, label, by_emotion):
    cats = Counter()
    rows = []
    for emo in sorted(by_emotion):
        for w in by_emotion[emo][rank_slice]:
            c = classify(w["peak_token"])
            cats[c] += 1
            rows.append((emo, repr(w["peak_token"]), c, w["source"]))
    total = sum(cats.values())
    nonconcept = total - cats["content"]
    print("\n=== %s (n=%d) ===" % (label, total))
    for c, n in cats.most_common():
        print("  %-20s %5d  (%.1f%%)" % (c, n, 100 * n / total))
    print("  NON-CONCEPTUAL RATE: %d/%d = %.1f%%  (prior rank-1 figure: >=41%%)"
          % (nonconcept, total, 100 * nonconcept / total))
    return rows


def _load():
    """Prefer the raw windows; fall back to the shipped derived peak_tokens.json.
    Both expose by_emotion[emo][i] with a 'peak_token' and 'source' field, so report()
    is agnostic to which was loaded."""
    if os.path.exists(PATH):
        return json.load(open(PATH))["by_emotion"], PATH
    if os.path.exists(DERIVED):
        return json.load(open(DERIVED))["by_emotion"], DERIVED
    raise SystemExit(
        "Neither %s nor %s found. Rebuild the corpus + sweep to produce the raw windows, "
        "or fetch the shipped %s." % (PATH, DERIVED, DERIVED))


def main():
    be, src = _load()
    print("source:", src)
    rows = report(slice(0, 1), "RANK-1 peaks, 171 emotions", be)
    report(slice(0, 20), "ALL top-20 peaks", be)

    print("\n=== every rank-1 peak token (eyeball check) ===")
    for emo, tok, cat, src in rows:
        print("  %-16s %-24s %-20s %s" % (emo, tok, cat, src))


if __name__ == "__main__":
    main()
