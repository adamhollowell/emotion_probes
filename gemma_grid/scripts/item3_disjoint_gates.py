#!/usr/bin/env python3
"""Gates for the item 3 corrected execution and the disjoint-topic transport
(dispatch DISPATCH_V8_TO_REPO_ITEM3_RERUN_AND_DISJOINT_TOPIC_2026-10-19.md,
sha256 09fd38ba...; pre-statement prestatements/DISJOINT_TOPIC_TRANSPORT.md, sha256 aab3eef2...).

Diagnostics only. No transport outcome is computed here.

Gate 1. Slice-file neutral rows against every neutral activation file for the cell.
Gate 2. Cleanup basis rebuilt from the cell's 100T neutral file at the analysis layer,
        against the stored <cell>__cleanup_basis.pt (same k, subspace overlap 1).
        Checked for both the plain np.linalg.svd rebuild and the reference clean_layer
        SVD path (_svd_thin, gesdd with gesvd fallback) that Parts A and B actually use.
Gate 3. Topic join of every slice row to its corpus story (explicit per-emotion convention rule).

Output: gemma_grid/geometry_out/item3_disjoint_gates.json (+ .MANIFEST.json)
Run with lens_env: lens_env/bin/python gemma_grid/scripts/item3_disjoint_gates.py

2026-10-19 Claude Code revision (before first run; original sha256 e8d7a253...):
 - library-version check and thread pinning at start;
 - input manifest (sha256 of every input read) + output sha256;
 - gate 3 convention rule made literal: position = every value in 0..1199;
   absolute = every value inside the emotion's contiguous block of corpus lines
   (block contiguity asserted); index entry length and emotion set checked;
 - gate 2 also checks the reference clean_layer SVD path and reports overlap even if k differs;
 - gate 1 records dtypes and compares in native dtype.
Ledger gate-3 amendment 2026-10-19 applied after the first run stopped on "guilty" (the corpus is
interleaved in 12-story emotion-topic blocks, so the range reading of "block" was ambiguous):
set-equality convention rule, ascending check, row r = r-th corpus line of E check, and a
same-topic vs different-topic cosine sanity gate on story rows (no outcome metric).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import glob, hashlib, json, sys, time, platform
from collections import Counter, defaultdict
import numpy as np
import scipy, sklearn
import torch

EXPECTED = {"python": "3.9.6", "numpy": "2.0.2", "scipy": "1.13.1", "sklearn": "1.6.1", "torch": "2.8.0"}


def check_versions():
    got = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__, "torch": torch.__version__.split("+")[0]}
    bad = {k: (got[k], EXPECTED[k]) for k in EXPECTED if got[k] != EXPECTED[k]}
    if bad:
        sys.exit(f"STOP: library version mismatch {bad}")
    return got


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gemma_grid/scripts"))
OUT = os.path.join(REPO, "gemma_grid/geometry_out/item3_disjoint_gates.json")
LAYER = {"gemma-2-2b-base": 17, "gemma-2-2b-it": 17, "gemma-2-9b-base": 28, "gemma-2-9b-it": 28,
         "gemma-2-27b-base": 31, "gemma-2-27b-it": 31}
CELLS = list(LAYER)
ANCHOR = "gemma-2-27b-base"
CORPUS = os.path.join(REPO, "data/stories/stories_171_canonical_clean.jsonl")
INDEX = os.path.join(REPO, "gemma_grid/handoff/story_indices_all_cells.json")
TOPICS = os.path.join(REPO, "gemma_grid/data/story_topics.json")
TARGET = 0.5
N_PER_EMO = 1200
INPUTS = {}   # rel path -> sha256


def sha(p, n=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, REPO)


def record(p):
    r = rel(p)
    if r not in INPUTS:
        INPUTS[r] = sha(p)
    return INPUTS[r]


def slice_path(c):
    return os.path.join(REPO, f"gemma_grid/data/{c}/vectors/{c}__per_story_slice.pt")


def neutral_100T_path(c):
    return os.path.join(REPO, "gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__neutral.pt") if c == ANCHOR else \
        os.path.join(REPO, f"gemma_grid/data/{c}/vectors/{c}__neutral.pt")


def basis_path(c):
    return os.path.join(REPO, f"gemma_grid/data/{c}/vectors/{c}__cleanup_basis.pt")


def load(p):
    record(p)
    return torch.load(p, map_location="cpu", weights_only=False, mmap=True)


def acts_of(blob):
    if isinstance(blob, dict):
        for k in ("acts", "neutral", "activations"):
            if k in blob:
                return blob[k]
    if torch.is_tensor(blob):
        return blob
    raise KeyError(f"no activation tensor in keys {list(blob)[:10] if isinstance(blob, dict) else type(blob)}")


def slice_layers(sl):
    v = sl.get("slice_layers")
    if v is None:
        v = sl.get("layers")
    return [int(x) for x in v]


def candidate_neutral_files(c):
    # glob skips dot-prefixed hidden temp files by default
    files = sorted(glob.glob(os.path.join(REPO, f"gemma_grid/data/{c}/vectors/{c}__neutral*.pt")))
    if c == ANCHOR:
        files += [os.path.join(REPO, f) for f in ("neutral.pt", "neutral_HA.pt", "neutral_PA.pt", "neutral_50TM.pt")
                  if os.path.exists(os.path.join(REPO, f))]
    return files


def label(path):
    b = os.path.basename(path)
    if "_PA" in b:
        return "original corpus"
    if "100T" in b or b.endswith("__neutral.pt"):  # released name of the 100T neutral file
        return "100T (canonical)"
    if "_HA" in b:
        return "HA (superseded)"
    if "50TM" in b:
        return "50TM (superseded)"
    return "unlabeled"


def gate1(c):
    sl = load(slice_path(c))
    layers = slice_layers(sl)
    sn_t = sl["neutral"]
    res = {"slice_file": rel(slice_path(c)), "slice_layers": layers, "slice_neutral_shape": list(sn_t.shape),
           "slice_neutral_dtype": str(sn_t.dtype), "comparisons": []}
    for f in candidate_neutral_files(c):
        blob = load(f)
        A = acts_of(blob)
        row = {"file": rel(f), "kind": label(f), "dtype": str(A.dtype), "shape": list(A.shape),
               "src_sha256": blob.get("src_sha256") if isinstance(blob, dict) else None, "per_layer": {}}
        bit_all = True
        for i, l in enumerate(layers):
            a_t = sn_t[:, i, :]
            b_t = A[:, l, :]
            if tuple(a_t.shape) != tuple(b_t.shape):
                row["per_layer"][str(l)] = {"shape_mismatch": [list(a_t.shape), list(b_t.shape)]}
                bit_all = False
                continue
            bit = bool(a_t.dtype == b_t.dtype and torch.equal(a_t, b_t))
            bit_all &= bit
            a = a_t.double().numpy(); b = b_t.double().numpy()
            row["per_layer"][str(l)] = {"rel_diff": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                                        "bit_identical": bit}
        row["bit_identical_all_layers"] = bit_all
        res["comparisons"].append(row)
        del A, blob
    matches = [r["file"] for r in res["comparisons"] if r["bit_identical_all_layers"]]
    res["bit_identical_to"] = matches
    res["matches_any_HA_or_100T"] = any(r["bit_identical_all_layers"] and r["kind"] in ("100T (canonical)", "HA (superseded)")
                                        for r in res["comparisons"])
    res["original_corpus_file_present"] = any(r["kind"] == "original corpus" for r in res["comparisons"])
    del sl
    return res


def k_of(S):
    r = S ** 2 / (S ** 2).sum()
    return int(np.argmax(np.cumsum(r) >= TARGET)) + 1


def overlap(Va, Vb):
    return float(np.linalg.norm(Va @ Vb.T) ** 2 / max(Va.shape[0], Vb.shape[0]))


def gate2(c):
    import analysisB_seam_onset_bootstrap_gesvdfix as B   # reference clean_layer's SVD path
    L = LAYER[c]
    N = acts_of(load(neutral_100T_path(c)))[:, L, :].double().numpy()
    Xc = N - N.mean(0, keepdims=True)
    _, S, Vh = np.linalg.svd(Xc, full_matrices=False)
    k = k_of(S); Vk = Vh[:k]
    _, S2, Vh2 = B._svd_thin(Xc)
    k2 = k_of(S2); Vk2 = Vh2[:k2]
    st = load(basis_path(c))
    Sk = np.asarray(st["bases_per_layer"][L], dtype=np.float64)
    ks = int(st["k_per_layer"][L])
    out = {"neutral_file": rel(neutral_100T_path(c)), "neutral_sha256": record(neutral_100T_path(c)),
           "stored_basis": rel(basis_path(c)), "stored_basis_sha256": record(basis_path(c)),
           "stored_analysis_layer": int(st["analysis_layer"]), "stored_target": float(st["target"]),
           "layer": L, "k_rebuilt": k, "k_rebuilt_clean_layer_path": k2, "k_stored": ks,
           "stored_basis_dtype": str(np.asarray(st["bases_per_layer"][L]).dtype),
           "subspace_overlap": overlap(Vk, Sk),
           "subspace_overlap_clean_layer_path": overlap(Vk2, Sk),
           "overlap_definition": "||Vk Sk^T||_F^2 / max(k_rebuilt, k_stored); 1 iff identical subspaces",
           "tolerance": "|overlap-1| < 1e-6 (stored basis is float32)"}
    out["pass"] = bool(k == ks == k2 and abs(out["subspace_overlap"] - 1.0) < 1e-6
                       and abs(out["subspace_overlap_clean_layer_path"] - 1.0) < 1e-6
                       and int(st["analysis_layer"]) == L and float(st["target"]) == TARGET)
    return out


def corpus_rows():
    if not os.path.exists(CORPUS):  # release: corpus withheld; emotion and topic per story ship without text
        record(TOPICS)
        t = json.load(open(TOPICS))
        return [{"emotion": t["emotions"][a], "topic": t["topics"][b]} for a, b in t["rows"]]
    record(CORPUS)
    return [json.loads(l) for l in open(CORPUS)]


def build_join(rows, idx, slice_keys_by_cell):
    """Apply the dispatch's per-emotion convention rule. Returns (report, mapping) where
    mapping[cell][emotion] = list of 1200 corpus line numbers in slice-row order."""
    by_emo = defaultdict(list)
    for i, r in enumerate(rows):
        by_emo[r["emotion"]].append(i)
    contiguous = {e: (v == list(range(v[0], v[0] + len(v)))) for e, v in by_emo.items()}
    topics_all = sorted({r["topic"] for r in rows})
    emo_topics = {e: {rows[i]["topic"] for i in by_emo[e]} for e in by_emo}
    rep = {"n_corpus_rows": len(rows), "n_emotions_in_corpus": len(by_emo), "n_topics": len(topics_all),
           "every_emotion_block_contiguous": all(contiguous.values()),
           "every_emotion_1200_rows": all(len(v) == N_PER_EMO for v in by_emo.values()),
           "every_emotion_same_100_topics": len(topics_all) == 100 and all(t == set(topics_all) for t in emo_topics.values()),
           "cells": {}}
    mappings = {}
    for c in CELLS:
        sl_keys = slice_keys_by_cell[c]
        cell = {"conventions": Counter(), "stops": [], "checks": {}}
        mapping = {}
        if c == ANCHOR:
            cell["index_entry_present"] = c in idx
            for e in sl_keys:
                mapping[e] = by_emo[e][:]
            cell["conventions"]["corpus order (anchor loader convention)"] = len(sl_keys)
        else:
            si = idx[c]["story_indices"]
            cell["index_emotions_equal_slice_emotions"] = set(si) == set(sl_keys)
            positions = set(range(N_PER_EMO))
            for e in sl_keys:
                v = [int(x) for x in si[e]]
                if len(v) != N_PER_EMO:
                    cell["stops"].append({"emotion": e, "reason": f"index entry has {len(v)} values, not 1200"})
                    continue
                if any(v[j + 1] <= v[j] for j in range(len(v) - 1)):
                    cell["stops"].append({"emotion": e, "reason": "index entry not strictly ascending"})
                    continue
                sv = set(v)
                pos = sv == positions                    # Ledger amendment 2026-10-19: set equality
                absl = sv == set(by_emo[e])
                if pos and absl:
                    cell["stops"].append({"emotion": e, "reason": "values satisfy both conventions"})
                    continue
                if not pos and not absl:
                    cell["stops"].append({"emotion": e, "reason": "values satisfy neither convention",
                                          "min": min(v), "max": max(v)})
                    continue
                if pos:
                    mapping[e] = [by_emo[e][x] for x in v]
                    cell["conventions"]["within-emotion position"] += 1
                else:
                    mapping[e] = list(v)
                    cell["conventions"]["absolute corpus line"] += 1
        cell["conventions"] = dict(cell["conventions"])
        if not cell["stops"]:
            used = [i for e in mapping for i in mapping[e]]
            cell["checks"]["emotion_matches"] = all(rows[i]["emotion"] == e for e in mapping for i in mapping[e])
            cell["checks"]["no_story_twice"] = len(used) == len(set(used))
            cell["checks"]["every_row_assigned"] = len(mapping) == 171 and all(len(mapping[e]) == N_PER_EMO for e in sl_keys)
            pair = Counter((e, rows[i]["topic"]) for e in mapping for i in mapping[e])
            cell["checks"]["twelve_per_emotion_topic_pair"] = len(pair) == 17100 and set(pair.values()) == {12}
            cell["checks"]["every_emotion_same_100_topics"] = all(
                {rows[i]["topic"] for i in mapping[e]} == set(topics_all) for e in mapping)
            # Ledger amendment: slice row r of emotion E is the r-th corpus line with emotion E
            cell["checks"]["row_r_is_rth_corpus_line_of_emotion"] = all(mapping[e] == by_emo[e] for e in mapping)
            if c != ANCHOR:
                cell["checks"]["index_emotions_equal_slice_emotions"] = cell.pop("index_emotions_equal_slice_emotions")
            cell["pass"] = all(cell["checks"].values())
            mappings[c] = mapping
        else:
            cell["pass"] = False
        rep["cells"][c] = cell
    # contiguity is reported only: the corpus is interleaved in 12-story emotion-topic blocks (Ledger amendment)
    rep["pass"] = (all(v["pass"] for v in rep["cells"].values()) and rep["every_emotion_same_100_topics"]
                   and rep["every_emotion_1200_rows"])
    return rep, mappings


def topic_sanity(c, mapping, rows):
    """Ledger amendment sanity gate, all 171 emotions (>= 20 required): mean cosine between
    same-topic story activations (raw slice story rows at the analysis layer, self-pairs excluded)
    vs different-topic. Gap must be positive in every emotion. Neutral rows not read."""
    sl = load(slice_path(c))
    li = slice_layers(sl).index(LAYER[c])
    per = {}
    for e, lines in mapping.items():
        X = sl["per_emotion"][e][:, li, :].double().numpy()
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        C = Xn @ Xn.T
        t = np.array([rows[i]["topic"] for i in lines])
        same = t[:, None] == t[None, :]
        np.fill_diagonal(same, False)
        diff = t[:, None] != t[None, :]
        ms, md = float(C[same].mean()), float(C[diff].mean())
        per[e] = {"mean_cos_same_topic": ms, "mean_cos_diff_topic": md, "gap": ms - md}
    del sl
    gaps = [v["gap"] for v in per.values()]
    return {"activations": "raw per-story slice rows at analysis layer (uncleaned, uncentered); self-pairs excluded",
            "n_emotions_sampled": len(per), "emotions_sampled": "all 171",
            "min_gap": float(min(gaps)), "n_nonpositive_gap": int(sum(g <= 0 for g in gaps)),
            "pass": bool(len(per) >= 20 and all(g > 0 for g in gaps)), "per_emotion": per}


def slice_keys(c):
    sl = load(slice_path(c))
    k = list(sl["per_emotion"].keys())
    del sl
    return k


def gate3():
    rows = corpus_rows()
    record(INDEX)
    idx = json.load(open(INDEX))
    keys = {c: slice_keys(c) for c in CELLS}
    rep, mappings = build_join(rows, idx, keys)
    rep["topic_sanity_gate"] = {}
    for c in CELLS:
        if c in mappings:
            print(f"  [sanity] {c}", flush=True)
            rep["topic_sanity_gate"][c] = topic_sanity(c, mappings[c], rows)
    rep["topic_sanity_pass"] = len(rep["topic_sanity_gate"]) == len(CELLS) and all(
        v["pass"] for v in rep["topic_sanity_gate"].values())
    rep["pass"] = rep["pass"] and rep["topic_sanity_pass"]
    rep.update({"corpus": rel(CORPUS), "corpus_sha256": record(CORPUS), "index_file": rel(INDEX),
                "index_sha256": record(INDEX),
                "convention_rule": "Ledger gate-3 amendment 2026-10-19 (set equality): per cell and emotion E, the "
                                   "index list (1,200 entries, ascending) is ABSOLUTE iff its set equals the set of "
                                   "corpus lines with emotion E; POSITIONS iff its set equals {0..1199}, position k = "
                                   "k-th corpus line with emotion E in file order; neither or both -> stop. Anchor "
                                   "follows corpus order.",
                "limit": "The slice files carry no story identifier, so a slice row's identity rests on the index file "
                         "(and, for the anchor, on the corpus-order loader convention) and cannot be verified beyond it."})
    return rep


def main():
    t0 = time.time()
    versions = check_versions()
    rep = {"record": "Gates for item 3 corrected execution and disjoint-topic transport. Diagnostics only.",
           "dispatch_sha256": "09fd38ba352a2eebf573362dc8b848f5a5afc74dad3685626b1569b96089018b",
           "prestatement_sha256": "aab3eef2dcb6d4ef1af38d4e49b2cc81cf502e23e56804f0a3c62dfdfa6c2d6b",
           "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "versions": versions,
           "producer": rel(os.path.abspath(__file__)), "producer_sha256": sha(os.path.abspath(__file__)),
           "gate1_slice_neutral_rows": {}, "gate2_cleanup_basis": {}}
    for c in CELLS:
        print(f"[gate1] {c}", flush=True)
        rep["gate1_slice_neutral_rows"][c] = gate1(c)
        print(f"  bit-identical to: {rep['gate1_slice_neutral_rows'][c]['bit_identical_to']}", flush=True)
    for c in CELLS:
        print(f"[gate2] {c}", flush=True)
        g = gate2(c)
        rep["gate2_cleanup_basis"][c] = g
        print(f"  k {g['k_rebuilt']}/{g['k_rebuilt_clean_layer_path']} vs {g['k_stored']}, overlap "
              f"{g['subspace_overlap']:.12f}/{g['subspace_overlap_clean_layer_path']:.12f}, pass {g['pass']}", flush=True)
    rep["gate2_pass"] = all(v["pass"] for v in rep["gate2_cleanup_basis"].values())
    print("[gate3] topic join", flush=True)
    g3 = gate3()
    rep["gate3_topic_join"] = g3
    for c, v in g3["cells"].items():
        print(f"  {c}: {v['conventions']} stops={len(v['stops'])} checks={v['checks']} pass={v['pass']}", flush=True)
    rep["gate3_pass"] = g3["pass"]
    rep["seconds"] = round(time.time() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    man = {"output": rel(OUT), "output_sha256": sha(OUT), "producer": rep["producer"],
           "producer_sha256": rep["producer_sha256"], "versions": versions, "inputs_sha256": INPUTS,
           "run_utc": rep["run_utc"]}
    with open(OUT + ".MANIFEST.json", "w") as f:
        json.dump(man, f, indent=2)
    print(f"[report] {OUT}  sha256 {man['output_sha256']}")
    print(f"gate2 {'PASS' if rep['gate2_pass'] else 'FAIL'}  gate3 {'PASS' if g3['pass'] else 'FAIL'}")
    return 0 if (rep["gate2_pass"] and g3["pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
