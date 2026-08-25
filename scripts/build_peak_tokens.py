#!/usr/bin/env python3
"""Derive geometry_out/peak_tokens.json from the (non-redistributed) max-activation
windows in data/heldout/maxact/maxactivating_v2.json.

maxactivating_v2.json carries full third-party window text and is NOT shipped. This
script extracts only the isolated PEAK TOKEN of each of the top-20 windows per emotion
(plus its structural category, source, and doc_id) — no window spans, positions, or
projection values — and writes the small derived artifact the P6.1 floor consumes.
Run scripts/analyze_maxact_peaks.py afterward to recompute the 52.6% floor.

Run from the repo root (needs the rebuilt corpus + sweep to have produced
maxactivating_v2.json; see the README recipe):
    python scripts/build_peak_tokens.py
"""
import json, os, re, time, hashlib
from collections import Counter

SRC = "data/heldout/maxact/maxactivating_v2.json"
OUT = "geometry_out/peak_tokens.json"
VEC = "data/vectors/emotion_vectors_171.pt"

# classification rule — kept byte-identical to scripts/analyze_maxact_peaks.py classify()
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


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    be = json.load(open(SRC))["by_emotion"]
    out = {}
    for emo in sorted(be):
        out[emo] = [
            {"rank": r, "peak_token": w["peak_token"], "category": classify(w["peak_token"]),
             "source": w["source"], "doc_id": w["doc_id"]}
            for r, w in enumerate(be[emo][:20], 1)
        ]
    n = len(out)
    r1 = Counter(out[e][0]["category"] for e in out)
    t20 = Counter(c["category"] for e in out for c in out[e])
    r1_nc = n - r1["content"]
    t20_total = sum(t20.values()); t20_nc = t20_total - t20["content"]
    comp = {"numeric": round(100 * r1["numeric"] / n, 1),
            "punct": round(100 * r1["punctuation/symbol"] / n, 1),
            "whitespace": round(100 * r1["whitespace/newline"] / n, 1),
            "function": round(100 * r1["function_word"] / n, 1),
            "content": round(100 * r1["content"] / n, 1)}
    doc = {
        "item": "P6.1 structural-floor peak tokens (derived from maxactivating_v2.json)",
        "desc": ("Per emotion, the top-20 max-activating windows' single PEAK TOKEN and its "
                 "structural category. Window text, positions, and projection values are NOT "
                 "included; only the isolated peak token per window is retained. This is the exact "
                 "input the P6.1 floor classifier consumes: run scripts/analyze_maxact_peaks.py to "
                 "recompute the 52.6% rank-1 floor and 48.4% top-20 rate."),
        "classifier": ("scripts/analyze_maxact_peaks.py classify(): whitespace/newline | numeric | "
                       "punctuation/symbol | function_word | content"),
        "floor_rank1_nonconcept": f"{r1_nc}/{n} = {round(100 * r1_nc / n, 1)}%",
        "floor_top20_nonconcept": f"{t20_nc}/{t20_total} = {round(100 * t20_nc / t20_total, 1)}%",
        "rank1_composition_pct": comp,
        "n_emotions": n, "per_window": 20,
        "by_emotion": out,
        "_provenance": {
            "basis": "canonical", "vectors": VEC,
            "derived_from": "data/heldout/maxact/maxactivating_v2.json (not redistributed; third-party window text)",
            "maxactivating_v2_sha256": _sha(SRC),
            "producer": "scripts/build_peak_tokens.py",
            "note": "Isolated peak tokens only; no third-party window spans. Reproduces P6.1 floor from shipped files.",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    json.dump(doc, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} | rank-1 floor {r1_nc}/{n} = {100 * r1_nc / n:.1f}% | top-20 {100 * t20_nc / t20_total:.1f}%")


if __name__ == "__main__":
    main()
