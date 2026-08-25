#!/usr/bin/env python3
"""P6.2: top-window source composition vs recorded baselines (v2 corpus).
Writes the citable output artifact source_distribution.{json,csv} into
data/heldout/maxact/ and prints a summary.
Includes: window mixes (rank-1 / top-20 / unique winners), document- and
swept-token baselines, observed-vs-expected tables, per-emotion source shares
(all 171), dominant-source list, rank-share profile, and PC1/PC2 circumplex
placement of the non-lmsys-dominant set (register-vs-geometry check).
Run from project root: python3 scripts/analyze_source_heterogeneity.py"""
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

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

BASE = Path("data/heldout/maxact")
RESULTS = BASE / "pass1_results.json"
VECTORS = Path("data/vectors/emotion_vectors_171.pt")
SOURCES = ("lmsys", "pile", "common_corpus", "isotonic")
LAYER = 31
POS = ["happy", "joyful", "content", "blissful", "delighted", "cheerful", "optimistic", "loving"]
NEG = ["afraid", "sad", "angry", "miserable", "panicked", "terrified", "depressed", "heartbroken"]
HI = ["enraged", "panicked", "ecstatic", "furious", "terrified", "hysterical", "thrilled", "alarmed"]
LO = ["calm", "serene", "sleepy", "relaxed", "listless", "peaceful", "droopy", "at ease"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def src_of(doc_id):
    return doc_id.rsplit("_", 1)[0]


def mix(counter):
    total = sum(counter.values())
    return {s: round(100.0 * counter.get(s, 0) / total, 2) for s in SOURCES}, total


def oriented_pcs(em, V):
    Vc = V - V.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Vc, full_matrices=False)
    pc1, pc2 = U[:, 0] * S[0], U[:, 1] * S[1]
    am = lambda c, ns: float(np.mean([c[em.index(x)] for x in ns if x in em]))
    if am(pc1, POS) < am(pc1, NEG):
        pc1 = -pc1
    if am(pc2, HI) < am(pc2, LO):
        pc2 = -pc2
    return pc1, pc2


def main():
    r = json.loads(RESULTS.read_text())
    be = r["by_emotion"]
    emotions = list(be.keys())
    tpd = r["tokens_per_doc_observed"]

    doc_counts = Counter(src_of(d) for d in tpd)
    tok_counts = Counter()
    for d, n in tpd.items():
        tok_counts[src_of(d)] += n

    rank1 = Counter(src_of(be[e][0]["doc_id"]) for e in emotions)
    top20 = Counter(src_of(w["doc_id"]) for e in emotions for w in be[e])
    uniq = Counter(src_of(d) for d in {w["doc_id"] for e in emotions for w in be[e]})

    md, n_docs = mix(doc_counts)
    mt, n_toks = mix(tok_counts)
    m1, _ = mix(rank1)
    m20, n_w = mix(top20)
    mu, n_u = mix(uniq)

    per_emotion = {}
    dominant = Counter()
    non_lmsys = []
    for e in emotions:
        c = Counter(src_of(w["doc_id"]) for w in be[e])
        dom = c.most_common(1)[0][0]
        dominant[dom] += 1
        per_emotion[e] = {"dominant": dom, **{s: c.get(s, 0) for s in SOURCES}}
        if dom != "lmsys":
            non_lmsys.append(e)

    rank_profile = {}
    for rank in range(20):
        c = Counter(src_of(be[e][rank]["doc_id"]) for e in emotions)
        rank_profile[rank + 1] = {s: round(100.0 * c.get(s, 0) / len(emotions), 1) for s in SOURCES}

    p = torch.load(VECTORS, map_location="cpu", weights_only=False)
    em_v = list(p["emotions"])
    V = p["vectors"][:, LAYER, :].float().numpy()
    pc1, pc2 = oriented_pcs(em_v, V)
    med1, med2 = float(np.median(pc1)), float(np.median(pc2))
    geometry_check = {}
    for e in non_lmsys:
        i = em_v.index(e)
        geometry_check[e] = {
            "pc1_valence": round(float(pc1[i]), 1),
            "below_median_valence": bool(pc1[i] < med1),
            "pc2_arousal": round(float(pc2[i]), 1),
            "below_median_arousal": bool(pc2[i] < med2),
        }

    expected = {s: {"observed_top20": top20.get(s, 0),
                    "expected_doc_baseline": round(n_w * md[s] / 100, 1),
                    "expected_token_baseline": round(n_w * mt[s] / 100, 1)} for s in SOURCES}

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"pass1_results_sha256": sha256_file(RESULTS),
                   "vectors_sha256": sha256_file(VECTORS)},
        "layer": LAYER,
        "n": {"docs": n_docs, "swept_tokens": n_toks, "top20_windows": n_w, "unique_winning_docs": n_u},
        "baseline_document_pct": md,
        "baseline_swept_token_pct": mt,
        "windows_rank1_pct": m1,
        "windows_top20_pct": m20,
        "windows_unique_docs_pct": mu,
        "observed_vs_expected_top20": expected,
        "dominant_source_counts": dict(dominant),
        "non_lmsys_dominant": sorted(non_lmsys),
        "geometry_check_non_lmsys": geometry_check,
        "rank_share_profile_pct": rank_profile,
        "per_emotion": per_emotion,
    }
    jpath = BASE / "source_distribution.json"
    jpath.write_text(json.dumps(out, indent=1))

    cpath = BASE / "source_distribution.csv"
    with open(cpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["emotion", "dominant"] + list(SOURCES))
        for e in emotions:
            row = per_emotion[e]
            w.writerow([e, row["dominant"]] + [row[s] for s in SOURCES])

    print("baseline docs  :", md)
    print("baseline tokens:", mt)
    print("rank-1 windows :", m1)
    print("top-20 windows :", m20)
    print("dominant sources:", dict(dominant))
    print("non-lmsys-dominant (%d): %s" % (len(non_lmsys), ", ".join(sorted(non_lmsys))))
    below1 = sum(g["below_median_valence"] for g in geometry_check.values())
    below2 = sum(g["below_median_arousal"] for g in geometry_check.values())
    print("geometry check: %d/%d below median valence (PC1), %d/%d below median arousal (PC2)"
          % (below1, len(geometry_check), below2, len(geometry_check)))
    print("wrote %s and %s" % (jpath, cpath))


if __name__ == "__main__":
    main()
