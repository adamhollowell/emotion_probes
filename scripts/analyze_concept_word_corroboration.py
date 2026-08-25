#!/usr/bin/env python3
"""P6.1 per-token corroboration, ALL 171 emotions (v2).
Among each emotion's unique top-20 winning docs whose text contains the emotion
word/phrase, where does the document argmax land?
Rules (printed in output): containment = whole-word/phrase regex match on the
detokenized text, case-insensitive, string-level (not sense-checked; polysemous
emotions like content/kind/patient may match non-emotion senses, which biases
TOWARD on-word hits, so a near-zero result is conservative). On-word = argmax
token (stripped, lowered) equals the emotion, or equals a hyphen/space part of
it with length > 3. Inspect any printed hits for word sense before counting.
Run from project root: python3 scripts/analyze_concept_word_corroboration.py"""
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

import json
import re
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("data/heldout/maxact")


def main():
    r = json.load(open(BASE / "pass1_results.json"))
    # Insertion order = pass-2 write order = emotion_vectors_171.pt emotion order,
    # which is the column order of the stored proj arrays. Do NOT sort.
    emotions = list(r["by_emotion"].keys())
    e_index = {e: i for i, e in enumerate(emotions)}

    # doc -> set of emotions whose top-20 include it
    need = {}
    for e in emotions:
        for w in r["by_emotion"][e]:
            need.setdefault(w["doc_id"], set()).add(e)
    print("unique winning docs: %d" % len(need))

    patterns = {e: re.compile(r"(?<![A-Za-z])" + re.escape(e) + r"(?![A-Za-z])", re.I)
                for e in emotions}
    parts = {e: {p for p in re.split(r"[\s\-]+", e) if len(p) > 3} | {e} for e in emotions}

    pt_dir = BASE / "pertoken"
    tar = tarfile.open(BASE / "pertoken.tar") if (BASE / "pertoken.tar").exists() else None

    def load_npz(doc_id):
        p = pt_dir / (doc_id + ".npz")
        if p.exists():
            return np.load(p, allow_pickle=True)
        if tar is None:
            raise FileNotFoundError("no pertoken/ dir entry and no pertoken.tar for " + doc_id)
        return np.load(tar.extractfile("pertoken/%s.npz" % doc_id), allow_pickle=True)

    n_contain = Counter()
    on_word = Counter()
    on_word_hits = []

    for k, (doc_id, emos) in enumerate(sorted(need.items())):
        z = load_npz(doc_id)
        toks = [str(t) for t in z["tokens"]]
        text = "".join(toks).lower()
        proj = z["proj"].astype(np.float32)
        proj[0, :] = -1e9  # BOS skip, matching the sweep
        for e in emos:
            if not patterns[e].search(text):
                continue
            n_contain[e] += 1
            am = int(proj[:, e_index[e]].argmax())
            am_tok = toks[am].strip().lower()
            if am_tok in parts[e]:
                on_word[e] += 1
                on_word_hits.append((e, repr(toks[am]), doc_id))
        if (k + 1) % 200 == 0:
            print("  ...%d/%d docs" % (k + 1, len(need)))

    tc, tw = sum(n_contain.values()), sum(on_word.values())
    print("\n=== POPULATION RESULT (all 171 emotions) ===")
    print("emotion-word-containing winning docs: %d (across %d emotions with >=1 such doc)"
          % (tc, sum(1 for e in emotions if n_contain[e])))
    print("argmax lands on the emotion word: %d / %d" % (tw, tc))
    if on_word_hits:
        print("\non-word hits (inspect each for word sense):")
        for e, t, d in on_word_hits:
            print("  %-16s %-16s %s" % (e, t, d))
    top = n_contain.most_common(8)
    print("\nmost word-containing emotions: " + ", ".join("%s(%d)" % x for x in top))


if __name__ == "__main__":
    main()
