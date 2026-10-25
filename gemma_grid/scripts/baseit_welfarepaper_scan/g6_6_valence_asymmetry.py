#!/usr/bin/env python3
"""Request H item (1) / G6.6 -- valence-asymmetry onset on the 100T lens.

Produces geometry_out/valence_asymmetry_onset.json: the 100T re-run of the
earlier recipe artifact valence_asymmetry_onset.json. Value-insertion
only (SR.13): no grading, no ledger edit, flip package untouched.

Recipe (matches the earlier artifact exactly), per cell:
  * norm lens over each cell's own W_U on CLEANED 100T vectors, all layers:
        logits = ((v/rms(v)) * (1+ln_w)) @ W_U,  rms eps = 1e-6
        own-word self-rank_e = 1 + #(logits_row > own_logit)
    over the fair-140 single-token (' word') emotion set (own-rank is per-emotion
    independent of the other rows, so the stack is restricted to the fair set --
    faithful and cheaper than the full 171).
  * split the fair set by the valence groups serialized in
    geometry_out/fair_emotions_v1.json -> g6_6_split (positive_50 /
    negative_87; n=137, the 3 fair-no-NRC excluded); per group compute
        onset = first layer whose GROUP median own-rank < 10000,
        L0    = group median own-rank at layer 0,
        final = group median own-rank at the last layer.
  * emit {cell: {pos:{onset,L0,final}, neg:{onset,L0,final}}} for all 6 cells.

Basis: CLEANED 100T all the way (anchor sha-gated, untouched); per-cell W_U from
<cell>__lens_ingredients.pt (anchor = repo-root lens_ingredients.pt). W_U is
basis-immune. Reuses the norm-lens own-rank convention of phase4_lens.py verbatim.
"""
import json
import zipfile

import numpy as np
import torch

import common as C
from common import nf
from p_lens_axis import wu_chunks, get_ln, load_vocab, V_SIZE

EPS = 1e-6
OUT = C.REPO / "gemma_grid" / "geometry_out" / "valence_asymmetry_onset.json"
FAIR = C.REPO / "gemma_grid" / "geometry_out" / "fair_emotions_v1.json"
SHA_LEDGER = C.REPO / "gemma_grid" / "neutrals_100T" / "run_phase4" / "100T_LOCAL_SHA256.txt"


def build_emo_tid(vocab):
    """{emotion: token_id} for single-token ' word' forms (family-shared tokenizer)."""
    inv = {}
    for i in range(V_SIZE):
        inv.setdefault(vocab[i], i)
    em = nf.load_vectors(C.REPO / C.CELLS["gemma-2-27b-base"]["vec"])[0]
    return {e: inv[" " + e] for e in em if (" " + e) in inv}


def gather_own_wu(cell, d, tids):
    """(len(tids), d) float32 rows of W_U for the given token ids."""
    uniq = sorted(set(int(t) for t in tids))
    if cell == "gemma-2-27b-base":
        zf = zipfile.ZipFile(C.REPO / C.CELLS[cell]["lens"])
        name = [n for n in zf.namelist() if n.endswith("data/0")][0]
        row_b = d * 4
        rows = {}
        with zf.open(name) as f:
            pos = 0
            for t in uniq:  # increasing order -> forward-only reads
                f.read(t * row_b - pos)
                rows[t] = np.frombuffer(f.read(row_b), np.float32).reshape(d).copy()
                pos = (t + 1) * row_b
        return {t: rows[t] for t in uniq}
    l = torch.load(C.REPO / C.CELLS[cell]["lens"], map_location="cpu",
                   weights_only=False, mmap=True)
    WU = l["W_U"]  # (d, vocab)
    return {t: WU[:, t].float().numpy().astype(np.float32) for t in uniq}


def cell_ranks(cell, emo_tid):
    """Return (em, L, ranks) where ranks[layer][emotion] = own-word self-rank."""
    em, V, layer = C.load_cell(cell)          # V float32 (n, L, d), basis-guarded
    n, L, d = V.shape
    ln = get_ln(cell, d)                       # (d,) float64
    fair = [e for e in em if e in emo_tid]
    assert len(fair) == 140, f"{cell}: fair count {len(fair)} != 140"
    fi = np.array([em.index(e) for e in fair])
    tids = np.array([emo_tid[e] for e in fair])

    # norm-lens stack for fair emotions, all layers: (L, nf, d) -> (L*nf, d)
    Vf = V[:, :, :].astype(np.float64)[:, :, :]  # (n,L,d)
    Vf = Vf[fi]                                   # (nf, L, d)
    rms = np.sqrt((Vf ** 2).mean(axis=2, keepdims=True) + EPS)
    Vn = (Vf / rms) * (1.0 + ln)                  # (nf, L, d)
    nf_ = len(fair)
    stack = Vn.transpose(1, 0, 2).reshape(L * nf_, d).astype(np.float32)  # layer-major
    row_tid = np.tile(tids, L)                    # own tid per row (layer-major)

    # own logits (single gather of the <=140 unique own W_U rows)
    own_w = gather_own_wu(cell, d, tids)
    W_own = np.stack([own_w[int(t)] for t in tids])          # (nf, d)
    own_layer = (Vn * W_own[:, None, :]).sum(axis=2)         # (nf, L)  per emotion,layer
    own = own_layer.T.reshape(L * nf_).astype(np.float32)    # layer-major, aligned to stack

    # streaming count: #(logit > own) over the whole vocab
    counts = np.zeros(L * nf_, np.int64)
    S_t = torch.from_numpy(stack)
    torch.set_num_threads(4)
    for c0, c1, Wb in wu_chunks(cell, d):
        logits = (S_t @ torch.from_numpy(np.ascontiguousarray(Wb)).T).numpy()  # (R, chunk)
        counts += (logits > own[:, None]).sum(axis=1)
    ranks = (counts + 1).reshape(L, nf_)          # (L, nf) layer-major
    return fair, L, ranks


def group_stats(fair, ranks, L, group_emotions):
    idx = [fair.index(e) for e in group_emotions if e in fair]
    med = np.median(ranks[:, idx], axis=1)        # (L,) group median per layer
    onset = next((l for l in range(L) if med[l] < 10000), None)
    return {"onset": (int(onset) if onset is not None else None),
            "L0": int(round(float(med[0]))),
            "final": int(round(float(med[-1])))}, med


def sha256_and_log(path):
    h = C.sha256_file(path)
    line = f"{h}  gemma_grid/geometry_out/{path.name}\n"
    if SHA_LEDGER.exists():  # working hash ledger, not part of the release
        with open(SHA_LEDGER, "a") as f:
            f.write(line)
    return h


def main():
    split = json.loads(FAIR.read_text())["g6_6_split"]
    pos, neg = split["positive_50"], split["negative_87"]
    vocab = load_vocab()
    emo_tid = build_emo_tid(vocab)

    result = {}
    xcheck = {}
    for cell in C.CELLS:
        print(f"[g6.6] {cell} ...", flush=True)
        fair, L, ranks = cell_ranks(cell, emo_tid)
        pstat, _ = group_stats(fair, ranks, L, pos)
        nstat, _ = group_stats(fair, ranks, L, neg)
        result[cell] = {"pos": pstat, "neg": nstat}
        # cross-check: combined pos+neg (137) median onset vs lens_summary onset
        comb_idx = [fair.index(e) for e in (pos + neg) if e in fair]
        comb_med = np.median(ranks[:, comb_idx], axis=1)
        comb_onset = next((l for l in range(L) if comb_med[l] < 10000), None)
        ls = json.loads((C.REPO / f"gemma_grid/geometry_out/{cell}/lens_summary.json").read_text())
        xcheck[cell] = {"fair_n": len(fair), "combined137_onset": comb_onset,
                        "lens_summary_onset": ls.get("onset_layer_median_lt_10k")}
        print(f"    pos {pstat} | neg {nstat} | xcheck comb137_onset={comb_onset} "
              f"vs lens_summary={ls.get('onset_layer_median_lt_10k')} (fair={len(fair)})")

    OUT.write_text(json.dumps(result, indent=2))
    sha = sha256_and_log(OUT)
    print(f"\nwrote {OUT}\n  sha256 {sha}  (appended to {SHA_LEDGER.name})")
    bad = [c for c, x in xcheck.items() if x["combined137_onset"] != x["lens_summary_onset"]]
    print("  CROSS-CHECK:", "ALL PASS" if not bad else f"MISMATCH in {bad}")
    for c, x in xcheck.items():
        print(f"    {c:18s} fair_n={x['fair_n']} comb137_onset={x['combined137_onset']} "
              f"lens_summary_onset={x['lens_summary_onset']}")


if __name__ == "__main__":
    main()
