#!/usr/bin/env python3
"""Part B of DISPATCH_V8 (sha256 09fd38ba...): disjoint-topic transport, exactly as the pre-statement
prestatements/DISJOINT_TOPIC_TRANSPORT.md (sha256 aab3eef2...) specifies. Exploratory. Producer does not grade.

Mechanics (handoff §6):
 - 16 fits = ordered pairs in pmap_transport.json["pairs"], in that order; battery transport()
   (frozen estimator, frozen RandomState(0) five-fold assignment, scoring against all 171 targets).
 - Partition: sorted 100 topic strings, RandomState(20261019).permutation(100), first 50 -> set A.
 - Random halves: per emotion a fresh RandomState(20261019).permutation(1200) over slice order,
   first 600 -> R1, rest -> R2.
 - Half-vector: mean over the emotion's 600 stories in the half, minus the half's grand mean (mean over
   171 emotions), cleaned with clean_layer(., cell's 100T neutral at the analysis layer) (gate-2-passed
   input). Story rows only; the slice `neutral` key is never read.
 - Topic of each slice row from the gated join (Ledger gate-3 amendment: slice row r of emotion E is
   the r-th corpus line with emotion E).
 - Bag-of-words: c3_tokens.npz spans from token 50 (fallback start=min(1,n-1) if n<=51, as
   emotion_probes/activations.py::_mean_resid_post_single), decoded with the gemma-2-27b tokenizer,
   lowercased, [a-z]+, vocabulary over all 205,200 spans, df>=5, L2-normalized span vectors, empty spans
   excluded and counted. Variant (ii) drops single-[a-z]+-token emotion names. Direct retrieval
   (cosine ranking, no map, both directions) only: the FITTED BoW form is WITHDRAWN by
   prestatements/DISJOINT_TOPIC_TRANSPORT_AMENDMENT_1.md (filed 2026-10-19) and is never computed.

Outputs (geometry_out/), each with .MANIFEST.json:
  disjoint_topic_part1_crosscell_disjoint.json
  disjoint_topic_part2_matched_halves.json
  disjoint_topic_part3_withincell.json
  disjoint_topic_bow.json
No story text is written. Run: .venv/bin/python gemma_grid/scripts/disjoint_topic_transport.py
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, hashlib, time, platform, re
from collections import Counter
import numpy as np

# ---- battery producer's authorized gesdd->gesvd fallback (as pmap_battery_serial.py) ----
_ORIG_SVD = np.linalg.svd
def _svd_fallback(a, full_matrices=True, compute_uv=True, hermitian=False):
    try:
        return _ORIG_SVD(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as _sp
        if not compute_uv:
            return _sp(a, full_matrices=full_matrices, compute_uv=False, lapack_driver="gesvd")
        return _sp(a, full_matrices=full_matrices, lapack_driver="gesvd")
np.linalg.svd = _svd_fallback

import scipy, scipy.sparse as sp, sklearn, torch, tokenizers
ENV_NAME = ".venv"
EXPECTED = {"python": "3.12.13", "numpy": "2.4.6", "scipy": "1.17.1", "sklearn": "1.8.0", "torch": "2.12.0",
            "tokenizers": "0.22.2"}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOGPU = os.path.join(REPO, "gemma_grid/scripts")
sys.path.insert(0, NOGPU)
sys.path.insert(0, os.path.join(REPO, "gemma_grid/scripts"))
import pmap_transport as P
import pmap_battery as M
import analysisB_seam_onset_bootstrap_gesvdfix as B
import item3_disjoint_gates as G      # build_join (Ledger-amended gate-3 rule)

GEO = os.path.join(REPO, "gemma_grid/geometry_out")
ART = os.path.join(GEO, "pmap_transport.json")
GATES = os.path.join(GEO, "item3_disjoint_gates.json")
CORPUS = os.path.join(REPO, "data/stories/stories_171_canonical_clean.jsonl")
INDEX = os.path.join(REPO, "gemma_grid/handoff/story_indices_all_cells.json")
TOKENS = os.path.join(REPO, "gemma_grid/data/c3_tokens.npz")
PRESTATEMENT = os.path.join(REPO, "gemma_grid/prestatements/DISJOINT_TOPIC_TRANSPORT.md")
AMENDMENT_1 = os.path.join(REPO, "gemma_grid/prestatements/DISJOINT_TOPIC_TRANSPORT_AMENDMENT_1.md")
TOK_REPO, TOK_REV = "google/gemma-2-27b", "938270f5272feb02779b55c2bb2fffdd0f53ff0c"
OUTS = {k: os.path.join(GEO, f"disjoint_topic_{k}.json") for k in
        ("part1_crosscell_disjoint", "part2_matched_halves", "part3_withincell", "bow")}
ANCHOR = "gemma-2-27b-base"
SEED = 20261019
N_EMO, N_PER, N_STORIES, SKIP = 171, 1200, 205200, 50
DF_MIN = 5
INPUTS = {}


def sha(p, n=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    r = os.path.relpath(p, REPO)
    return p if r.startswith("..") else r


def record(p):
    r = rel(p)
    if r not in INPUTS:
        INPUTS[r] = sha(p)
    return INPUTS[r]


def check_versions():
    got = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__, "torch": torch.__version__.split("+")[0],
           "tokenizers": tokenizers.__version__}
    bad = {k: (got[k], EXPECTED[k]) for k in EXPECTED if got[k] != EXPECTED[k]}
    if os.path.realpath(sys.prefix) != os.path.realpath(os.path.join(REPO, ENV_NAME)):
        bad["environment"] = (sys.prefix, ENV_NAME)
    if bad:
        sys.exit(f"STOP: environment/library version mismatch {bad}")
    cfg = np.show_config(mode="dicts")
    got["environment"] = ENV_NAME
    got["sys_prefix"] = sys.prefix
    got["numpy_blas_lapack"] = {k: cfg["Build Dependencies"][k] for k in ("blas", "lapack")}
    got["blas_threads"] = {v: os.environ[v] for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                       "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
    return got


def slice_path(c):
    return os.path.join(REPO, f"gemma_grid/data/{c}/vectors/{c}__per_story_slice.pt")


def neutral_100T_path(c):
    return os.path.join(REPO, "gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__neutral.pt") if c == ANCHOR else \
        os.path.join(REPO, f"gemma_grid/data/{c}/vectors/{c}__neutral.pt")


def criterion(m):
    return {**m, "criterion_median_rank_le_2": bool(m["median_rank"] <= 2)}


# ------------------------------------------------------------------ model half-vectors
def cell_halves(c, mapping, rows, setA, emos):
    """Return dict of four cleaned 171xd half matrices {A,B,R1,R2} plus diagnostics."""
    sl = torch.load(slice_path(c), map_location="cpu", weights_only=False, mmap=True)
    lay = sl.get("slice_layers") or sl.get("layers")
    li = list(lay).index(M.LAYER[c])
    pe = sl["per_emotion"]
    H = {k: [] for k in ("A", "B", "R1", "R2")}
    for e in emos:
        X = np.asarray(pe[e][:, li, :], dtype=np.float64)                  # (1200, d) story rows only
        assert X.shape[0] == N_PER
        inA = np.array([rows[i]["topic"] in setA for i in mapping[e]])
        assert inA.sum() == 600, (c, e, inA.sum())
        perm = np.random.RandomState(SEED).permutation(N_PER)
        H["A"].append(X[inA].mean(0)); H["B"].append(X[~inA].mean(0))
        H["R1"].append(X[perm[:600]].mean(0)); H["R2"].append(X[perm[600:]].mean(0))
    del sl, pe
    npath = neutral_100T_path(c)
    record(npath)
    N100 = torch.load(npath, map_location="cpu", weights_only=False, mmap=True)["acts"][:, M.LAYER[c], :]
    N100 = N100.double().numpy()
    out = {}
    for k, rows_k in H.items():
        V = np.stack(rows_k)
        V = V - V.mean(0, keepdims=True)                                      # half's grand mean
        out[k] = B.clean_layer(V, N100)
    return out


# ------------------------------------------------------------------ bag-of-words
def direct_retrieval(Q, T):
    """Rank all target rows by cosine to each query row (no map); estimator's rank convention."""
    S = P.cosine_rows(Q, T)
    n = Q.shape[0]
    ranks = np.zeros(n, int); coss = np.zeros(n)
    for j in range(n):
        order = np.argsort(-S[j])
        ranks[j] = int(np.where(order == j)[0][0]) + 1
        coss[j] = float(S[j, j])
    return M.metrics(ranks, coss)


def bow(rows, emos, setA):
    from huggingface_hub import hf_hub_download
    tok_path = hf_hub_download(TOK_REPO, "tokenizer.json", revision=TOK_REV)
    tok_sha = sha(tok_path)
    tok = tokenizers.Tokenizer.from_file(tok_path)
    record(TOKENS)
    z = np.load(TOKENS, allow_pickle=False)
    ids, off, labels = z["ids"], z["offsets"], z["labels"]
    assert int(z["n_stories"]) == N_STORIES and int(z["bos_token_id"]) == 2
    corpus_emo = np.array([r["emotion"] for r in rows])
    assert len(corpus_emo) == N_STORIES and np.array_equal(labels[:N_STORIES], corpus_emo), \
        "STOP: c3_tokens labels do not equal corpus emotion column row for row"
    spans, n_fallback, n_special = [], 0, 0
    special = {0, 1, 2, 3}   # pad, eos, bos, unk ids in the gemma-2 vocab
    for i in range(N_STORIES):
        s = ids[off[i]:off[i + 1]]
        n = len(s)
        start = SKIP if n > SKIP + 1 else min(1, n - 1)
        n_fallback += int(not n > SKIP + 1)
        sp_ = s[start:].tolist()
        n_special += sum(1 for t in sp_ if t in special)
        spans.append(sp_)
    texts = tok.decode_batch(spans, skip_special_tokens=False)
    del spans
    R = re.compile(r"[a-z]+")
    docs = [Counter(R.findall(t.lower())) for t in texts]
    del texts
    df = Counter()
    for d in docs:
        df.update(d.keys())
    vocab_all = sorted(w for w, c in df.items() if c >= DF_MIN)
    single = sorted({e for e in emos if re.fullmatch(r"[a-z]+", e)})
    multi = sorted(set(emos) - set(single))
    vocab_ii = [w for w in vocab_all if w not in set(single)]

    topic_arr = np.array([r["topic"] for r in rows])
    inA_story = np.isin(topic_arr, sorted(setA))
    emo_index = {e: np.where(corpus_emo == e)[0] for e in emos}

    res = {"span_rule": "token ids from index 50 onward; if the sequence has <= 51 tokens, start = min(1, n-1) "
                        "(main pipeline fallback, activations.py::_mean_resid_post_single); ids include BOS",
           "n_spans_using_fallback": n_fallback,
           "n_special_token_ids_in_spans": n_special,
           "tokenizer": {"repo": TOK_REPO, "revision": TOK_REV, "file": "tokenizer.json", "sha256": tok_sha,
                         "decode": "tokenizers.Tokenizer.decode_batch(skip_special_tokens=False)"},
           "features": "lowercased [a-z]+ word counts; vocabulary over all 205,200 spans; words in < 5 spans "
                       "dropped; each span vector L2-normalized; empty spans excluded and counted",
           "raw_vocab_size": len(df), "exclusion_list_variant_ii": single,
           "exclusion_list_in_vocab": [w for w in single if w in set(vocab_all)],
           "emotion_names_not_single_token_kept": multi,
           "fitted_retrieval": "WITHDRAWN before it was run by Amendment 1 (see amendment_1); never computed",
           "variants": {}}
    for vname, vocab in (("i_all_words", vocab_all), ("ii_exact_label_removed", vocab_ii)):
        col = {w: j for j, w in enumerate(vocab)}
        indptr, indices, data = [0], [], []
        for d in docs:
            for w, c in d.items():
                j = col.get(w)
                if j is not None:
                    indices.append(j); data.append(c)
            indptr.append(len(indices))
        X = sp.csr_matrix((np.asarray(data, np.float64), np.asarray(indices, np.int64), np.asarray(indptr)),
                          shape=(N_STORIES, len(vocab)))
        nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        empty = nrm == 0
        X = sp.diags(np.where(empty, 0.0, 1.0 / np.where(empty, 1.0, nrm))) @ X
        HA, HB, nA, nB = [], [], [], []
        for e in emos:
            ix = emo_index[e]
            a = ix[inA_story[ix] & ~empty[ix]]; b = ix[~inA_story[ix] & ~empty[ix]]
            nA.append(len(a)); nB.append(len(b))
            HA.append(np.asarray(X[a].mean(0)).ravel()); HB.append(np.asarray(X[b].mean(0)).ravel())
        HA, HB = np.stack(HA), np.stack(HB)
        HA -= HA.mean(0, keepdims=True); HB -= HB.mean(0, keepdims=True)
        zA = [emos[i] for i in np.where(np.linalg.norm(HA, axis=1) == 0)[0]]
        zB = [emos[i] for i in np.where(np.linalg.norm(HB, axis=1) == 0)[0]]
        keep = np.array([e not in set(zA) | set(zB) for e in emos])
        empty_by_emo = {e: int(empty[emo_index[e]].sum()) for e in emos if empty[emo_index[e]].any()}
        res["variants"][vname] = {
            "vocab_size": len(vocab),
            "n_empty_spans_excluded": int(empty.sum()),
            "n_empty_spans_excluded_by_emotion": empty_by_emo,
            "n_spans_per_emotion_half": {"set_A_min": int(min(nA)), "set_A_max": int(max(nA)),
                                         "set_B_min": int(min(nB)), "set_B_max": int(max(nB))},
            "zero_norm_centered_vectors": {"set_A": zA, "set_B": zB},
            "n_emotions_scored": int(keep.sum()),
            "zero_norm_handling": "an emotion with a zero-norm centered vector in either half is excluded from "
                                  "both query and candidate sets",
            "direct": {"A_to_B": direct_retrieval(HA[keep], HB[keep]),
                       "B_to_A": direct_retrieval(HB[keep], HA[keep])}}
        print(f"[bow] {vname} V={len(vocab)} empty={int(empty.sum())} zero-norm A={zA} B={zB} "
              f"direct A->B {res['variants'][vname]['direct']['A_to_B']}", flush=True)
        del X
    return res


def write(path, obj, versions, run_utc):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    man = {"output": rel(path), "output_sha256": sha(path), "producer": rel(os.path.abspath(__file__)),
           "producer_sha256": sha(os.path.abspath(__file__)), "versions": versions,
           "inputs_sha256": dict(INPUTS), "run_utc": run_utc}
    with open(path + ".MANIFEST.json", "w") as f:
        json.dump(man, f, indent=2)
    return man["output_sha256"]


def main():
    versions = check_versions()
    run_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for p in (os.path.join(NOGPU, "pmap_transport.py"), os.path.join(NOGPU, "pmap_battery.py"),
              os.path.join(NOGPU, "analysisB_seam_onset_bootstrap_gesvdfix.py"),
              os.path.join(REPO, "gemma_grid/scripts/item3_disjoint_gates.py"), ART, GATES, INDEX):
        record(p)
    assert record(PRESTATEMENT) == "aab3eef2dcb6d4ef1af38d4e49b2cc81cf502e23e56804f0a3c62dfdfa6c2d6b", "STOP: pre-statement hash"
    amend_sha = record(AMENDMENT_1)
    gates = json.load(open(GATES))
    assert gates["gate2_pass"] and gates["gate3_pass"], "STOP: gates 2/3 not passed"
    M.load_all()
    emos = list(M.EMOS)
    pairs = list(json.load(open(ART))["pairs"].keys())
    assert len(pairs) == 16

    rows = G.corpus_rows(); record(CORPUS if os.path.exists(CORPUS) else G.TOPICS)
    idx = json.load(open(INDEX))
    keys = {}
    for c in M.CELLS:
        record(slice_path(c))
        keys[c] = list(torch.load(slice_path(c), map_location="cpu", weights_only=False, mmap=True)["per_emotion"].keys())
    jrep, mappings = G.build_join(rows, idx, keys)
    assert jrep["pass"], "STOP: topic join did not pass"
    topics_sorted = sorted({r["topic"] for r in rows})
    assert len(topics_sorted) == 100
    tperm = np.random.RandomState(SEED).permutation(100)
    set_A = [topics_sorted[i] for i in tperm[:50]]
    set_B = [topics_sorted[i] for i in tperm[50:]]
    setA = set(set_A)
    common = {"dispatch_sha256": "09fd38ba352a2eebf573362dc8b848f5a5afc74dad3685626b1569b96089018b",
              "prestatement_sha256": "aab3eef2dcb6d4ef1af38d4e49b2cc81cf502e23e56804f0a3c62dfdfa6c2d6b",
              "amendment_1": {"file": rel(AMENDMENT_1), "sha256": amend_sha,
                              "effect": "fitted bag-of-words retrieval withdrawn; direct form reported alone"},
              "level": "pre-stated, exploratory; producer does not grade",
              "partition": {"rule": "sorted 100 corpus topic strings; numpy RandomState(20261019).permutation(100); "
                                    "first 50 positions -> set A", "seed": SEED, "set_A_topics": set_A},
              "random_halves": {"rule": "per emotion, fresh RandomState(20261019).permutation(1200) over slice order; "
                                        "first 600 -> R1, rest -> R2", "seed": SEED},
              "half_vector": "mean over the emotion's 600 stories in the half, minus the half's grand mean over 171 "
                             "emotions, cleaned by clean_layer with the cell's 100T neutral acts at the analysis "
                             "layer (gate-2-passed); story rows only, slice neutral rows never read",
              "estimator": {"module": "pmap_transport.py (unmodified) via pmap_battery.transport",
                            "map": "semi-orthogonal Procrustes + global scale", "fold_seed": P.FOLD_SEED,
                            "n_folds": P.N_FOLDS, "scoring": "all 171 targets"},
              "criterion": "median rank <= 2 across all 171 predictions (inherited from section 4.3; reported as "
                           "a fact, not a grade)",
              "analysis_layers": dict(M.LAYER), "versions": versions, "run_utc": run_utc,
              "join": "Ledger gate-3 amendment 2026-10-19; gates file item3_disjoint_gates.json"}

    halves = {}
    for c in M.CELLS:
        t0 = time.time()
        halves[c] = cell_halves(c, mappings[c], rows, setA, emos)
        print(f"[halves] {c} ({time.time()-t0:.0f}s)", flush=True)

    def run_cross(a, b, label):
        fits = []
        for key in pairs:
            s, t = key.split("->")
            for src_h, tgt_h in ((a, b), (b, a)):
                t0 = time.time()
                m = criterion(M.transport(halves[s][src_h], halves[t][tgt_h]))
                fits.append({"pair": key, "assignment": f"source {src_h} -> target {tgt_h}", **m})
                print(f"[{label}] {key} {src_h}->{tgt_h} {m} ({time.time()-t0:.0f}s)", flush=True)
        return fits

    p1 = run_cross("A", "B", "part1")
    out1 = {"record": "Part 1. Cross-cell, disjoint topics (source set A, target set B, then swapped). 32 fits.",
            **common, "fits": p1,
            "criterion_check": {"n_fits": len(p1), "n_meeting": sum(f["criterion_median_rank_le_2"] for f in p1),
                                "fits_not_meeting": [{"pair": f["pair"], "assignment": f["assignment"],
                                                      "median_rank": f["median_rank"]}
                                                     for f in p1 if not f["criterion_median_rank_le_2"]]}}
    h1 = write(OUTS["part1_crosscell_disjoint"], out1, versions, run_utc)

    p2 = run_cross("R1", "R2", "part2")
    out2 = {"record": "Part 2. Cross-cell, matched same-topic half-sample control (source R1, target R2, then "
                      "swapped). 32 fits. Descriptive reference.", **common, "fits": p2,
            "criterion_check": {"n_fits": len(p2), "n_meeting": sum(f["criterion_median_rank_le_2"] for f in p2)}}
    h2 = write(OUTS["part2_matched_halves"], out2, versions, run_utc)

    p3 = []
    for c in M.CELLS:
        for a, b in (("A", "B"), ("B", "A")):
            m = criterion(M.transport(halves[c][a], halves[c][b]))
            p3.append({"cell": c, "assignment": f"set {a} -> set {b}", **m})
            print(f"[part3] {c} {a}->{b} {m}", flush=True)
    out3 = {"record": "Part 3. Within-cell topic-split reference (set A -> set B and set B -> set A). 12 fits. "
                      "Descriptive reference.", **common, "fits": p3,
            "criterion_check": {"n_fits": len(p3), "n_meeting": sum(f["criterion_median_rank_le_2"] for f in p3)}}
    h3 = write(OUTS["part3_withincell"], out3, versions, run_utc)
    del halves

    t0 = time.time()
    if not os.path.exists(CORPUS):  # the bag-of-words baseline reads the story text, which the release withholds
        print("[bow] skipped: needs the story corpus, which the release does not include")
        for k, h in zip(OUTS, (h1, h2, h3)):
            print(f"[out] {rel(OUTS[k])} sha256 {h}")
        return 0
    bw = bow(rows, emos, setA)
    out4 = {"record": "Bag-of-words baseline, variants (i) and (ii). Direct retrieval both directions. Fitted "
                      "retrieval withdrawn by Amendment 1 (never computed).", **common, **bw,
            "half_vectors": "per emotion, mean over that emotion's set-A (set-B) non-empty spans, centered on the "
                            "half's grand mean over 171 emotions"}
    for k in ("random_halves", "half_vector", "estimator", "analysis_layers", "join"):
        out4.pop(k, None)
    h4 = write(OUTS["bow"], out4, versions, run_utc)
    print(f"[bow] done ({time.time()-t0:.0f}s)")
    for k, h in zip(OUTS, (h1, h2, h3, h4)):
        print(f"[out] {rel(OUTS[k])} sha256 {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
