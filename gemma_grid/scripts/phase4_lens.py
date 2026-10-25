#!/usr/bin/env python3
"""Phase 4 — the registered C5 logit-lens package, all six cells.

Registered at GATE 1 (PREREGISTRATION (c) C5: INCLUDE): the E1 §7.1 unembed
table (P4.1 Table-1 analog), the §7.2 raw-weights (1+w) lens (P4.3), and the
§7.3 lexicalization-onset depth (P4.5) — captured at extraction, analysis
never run until now (omission logged in the ledger).

Conventions copied from E1 (the anchor must reproduce E1's locked results):
  * Table-1 analog: RAW operation v @ W_U (no norm, no bias) on CLEANED
    vectors at the cell's analysis layer; report top-5 / bottom-5 tokens for
    Anthropic's 12 Table-1 emotions (P4.1).
  * Onset sweep: norm lens per P4.3 — logits = ((v/rms(v)) * (1+ln_w)) @ W_U,
    rms with eps=1e-6 (Gemma2 documented form), CLEANED vectors, ALL layers.
    Own-word self-rank per emotion (140/171 have single-token " word" forms —
    matches E1's fair-self-rank count exactly); onset = first layer where the
    MEDIAN self-rank < 10,000 (P4.5).
  * Anchor gates: onset == L23; median-rank collapse across L21–24 ≈
    23k→4.8k→3k; Table-1 anchor rows contain P4.1's listed tokens.

Engineering: one streaming pass over each cell's W_U (vocab-chunked; anchor's
4.7GB f32 W_U is pre-converted to an f16 memmap — storage layout is
token-major (256000, d), verified from the pickle). All 46(+1 raw) layer
readouts are ranked in the same pass: rank_e = 1 + #(logits_row > own_logit).
Resumable within the 45 s window; state under mnt/outputs/lens_scratch/.

Usage: python gemma_grid/scripts/phase4_lens.py --cell <cell>   (repeat until DONE)
       python gemma_grid/scripts/phase4_lens.py --convert-anchor (first, for 27b-base)
Output: geometry_out/<cell>/lens_summary.json
"""
import argparse, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf
from phase4_stageA import CELLS

OUT = REPO / "gemma_grid" / "geometry_out"
SCRATCH = Path(os.environ.get("BOOT_SCRATCH", ".scratch")) / "lens_scratch"
LENS_ASSET_DIR = Path(os.environ.get("LENS_ASSET_DIR", os.environ.get("BOOT_SCRATCH", ".scratch")))
VOCAB = json.load(open(LENS_ASSET_DIR / "anchor_vocab.json"))
EMO_TID = json.load(open(LENS_ASSET_DIR / "emotion_token_ids.json"))
V_SIZE = 256000
EPS = 1e-6
CHUNK = 8000
TABLE1_EMOS = ["happy", "sad", "angry", "afraid", "surprised", "guilty",
               "nervous", "desperate", "proud", "loving", "inspired", "calm"]
ANCHOR_LENS_ZIP = REPO / "data/vectors/lens_ingredients.pt"
ANCHOR_F16 = SCRATCH / "anchor_WU_f16.npy"
LENS = {c: REPO / f"gemma_grid/data/{c}/vectors/{c}__lens_ingredients.pt" for c in CELLS}


def convert_anchor(budget=33.0):
    """Stream the anchor W_U storage (token-major (256000,4608) f32, ZIP_STORED)
    into an f16 memmap. Resumable."""
    t0 = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    st_p = SCRATCH / "convert_state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {"tok": 0}
    mm = np.lib.format.open_memmap(ANCHOR_F16, mode=("r+" if ANCHOR_F16.exists() else "w+"),
                                   dtype=np.float16, shape=(V_SIZE, 4608))
    zf = zipfile.ZipFile(ANCHOR_LENS_ZIP)
    name = [n for n in zf.namelist() if n.endswith("data/0")][0]
    row_b = 4608 * 4
    with zf.open(name) as f:
        f.seek(st["tok"] * row_b)
        while st["tok"] < V_SIZE and time.time() - t0 < budget:
            blk = min(20000, V_SIZE - st["tok"])
            rows = np.frombuffer(f.read(blk * row_b), np.float32).reshape(blk, 4608)
            mm[st["tok"]:st["tok"] + blk] = rows.astype(np.float16)
            st["tok"] += blk
    st_p.write_text(json.dumps(st))
    print(("DONE convert" if st["tok"] >= V_SIZE else f"PROGRESS convert {st['tok']}/{V_SIZE}")
          + f" ({time.time()-t0:.0f}s)")
    return st["tok"] >= V_SIZE


def get_ln(cell):
    if cell == "gemma-2-27b-base":
        return np.load(LENS_ASSET_DIR / "anchor_ln_final.npy").astype(np.float64)
    l = torch.load(LENS[cell], map_location="cpu", weights_only=False, mmap=True)
    return l["ln_final_w"].float().numpy().astype(np.float64)


def wu_chunk(cell, t0_, t1_):
    """(chunk, d) float32 block of W_U in token-major orientation."""
    if cell == "gemma-2-27b-base":
        mm = np.load(ANCHOR_F16, mmap_mode="r")
        return mm[t0_:t1_].astype(np.float32)
    l = torch.load(LENS[cell], map_location="cpu", weights_only=False, mmap=True)
    return l["W_U"][:, t0_:t1_].float().numpy().T.copy()   # W_U is (d, vocab) view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=CELLS)
    ap.add_argument("--convert-anchor", action="store_true")
    ap.add_argument("--budget", type=float, default=33.0)
    a = ap.parse_args()
    t0 = time.time()
    if a.convert_anchor:
        convert_anchor(a.budget); return
    cell = a.cell
    sdir = SCRATCH / cell; sdir.mkdir(parents=True, exist_ok=True)
    st_p = sdir / "state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {"chunk": 0}

    em, V = nf.load_vectors(REPO / CELLS[cell]["vec"])       # cleaned, all layers
    n, L, d = V.shape
    ln = get_ln(cell); assert ln.shape[0] == d
    layer_A = CELLS[cell]["layer"]
    fair = [e for e in em if e in EMO_TID]
    tids = np.array([EMO_TID[e] for e in fair])
    # build the stacked readout matrix: L norm-lens layers + 1 raw analysis layer
    Vf = V.astype(np.float64)
    rms = np.sqrt((Vf ** 2).mean(axis=2, keepdims=True) + EPS)
    Vn = (Vf / rms) * (1.0 + ln)                             # (n, L, d) norm-lens
    stack = np.concatenate([Vn.transpose(1, 0, 2).reshape(L * n, d),
                            Vf[:, layer_A, :]], axis=0).astype(np.float32)  # (+171 raw rows)
    R = stack.shape[0]

    counts_p, own_p, top_p = sdir / "counts.npy", sdir / "own.npy", sdir / "tops.npz"
    if own_p.exists():
        own = np.load(own_p)
    else:
        own = None
    counts = np.load(counts_p) if counts_p.exists() else np.zeros(R, np.int64)  # per row: #logits > own (own col per row's emotion)
    # per-row own token id: rows are layer-major then emotion; raw rows same emotion order
    row_tid = np.full(R, -1, np.int64)
    fair_idx = {em.index(e): EMO_TID[e] for e in fair}
    for l_ in range(L + 1):
        for ei in range(n):
            row = l_ * n + ei
            if ei in fair_idx: row_tid[row] = fair_idx[ei]
    # running top/bottom-5 for Table-1 (raw rows only) stored across chunks
    if top_p.exists():
        z = np.load(top_p, allow_pickle=True)
        top_val, top_id = z["tv"].copy(), z["ti"].copy()
        bot_val, bot_id = z["bv"].copy(), z["bi"].copy()
    else:
        k = 8
        top_val = np.full((n, k), -np.inf, np.float32); top_id = np.zeros((n, k), np.int64)
        bot_val = np.full((n, k), np.inf, np.float32);  bot_id = np.zeros((n, k), np.int64)

    if own is None:
        # pass A: own-token logits (gather 140 W_U rows once)
        rows_needed = np.unique(tids)
        W_own = np.stack([wu_chunk(cell, t, t + 1)[0] for t in rows_needed])  # (u, d)
        idx_of = {t: i for i, t in enumerate(rows_needed)}
        own = np.full(R, np.nan, np.float32)
        for row in range(R):
            if row_tid[row] >= 0:
                own[row] = float(stack[row] @ W_own[idx_of[row_tid[row]]])
        np.save(own_p, own)
        print(f"pass A done ({time.time()-t0:.0f}s)")
        if time.time() - t0 > a.budget:
            st_p.write_text(json.dumps(st)); return

    torch.set_num_threads(4)
    S_t = torch.from_numpy(stack)
    nchunks = (V_SIZE + CHUNK - 1) // CHUNK
    while st["chunk"] < nchunks and time.time() - t0 < a.budget:
        c0 = st["chunk"] * CHUNK; c1 = min(c0 + CHUNK, V_SIZE)
        Wb = torch.from_numpy(wu_chunk(cell, c0, c1))        # (chunk, d)
        logits = (S_t @ Wb.T).numpy()                        # (R, chunk)
        m = row_tid >= 0
        counts[m] += (logits[m] > own[m, None]).sum(axis=1)
        # Table-1 running top/bottom on raw rows (last n rows)
        raw = logits[L * n:]
        for arrs, vals, ids, better in ((0, top_val, top_id, 1), (1, bot_val, bot_id, -1)):
            cat_v = np.concatenate([vals, raw * (1 if better == 1 else 1)], axis=1) if False else None
        cat_v = np.concatenate([top_val, raw], axis=1)
        cat_i = np.concatenate([top_id, np.tile(np.arange(c0, c1), (n, 1))], axis=1)
        sel = np.argsort(-cat_v, axis=1)[:, :top_val.shape[1]]
        top_val = np.take_along_axis(cat_v, sel, 1); top_id = np.take_along_axis(cat_i, sel, 1)
        cat_v = np.concatenate([bot_val, raw], axis=1)
        cat_i = np.concatenate([bot_id, np.tile(np.arange(c0, c1), (n, 1))], axis=1)
        sel = np.argsort(cat_v, axis=1)[:, :bot_val.shape[1]]
        bot_val = np.take_along_axis(cat_v, sel, 1); bot_id = np.take_along_axis(cat_i, sel, 1)
        st["chunk"] += 1
        if st["chunk"] % 4 == 0 or st["chunk"] == nchunks:
            np.save(counts_p, counts)
            np.savez(top_p, tv=top_val, ti=top_id, bv=bot_val, bi=bot_id)
            st_p.write_text(json.dumps(st))
    if st["chunk"] < nchunks:
        print(f"PROGRESS {cell} chunk {st['chunk']}/{nchunks} ({time.time()-t0:.0f}s)"); return
    np.save(counts_p, counts); np.savez(top_p, tv=top_val, ti=top_id, bv=bot_val, bi=bot_id)

    # ---- assemble
    ranks = counts + 1
    per_layer = []
    for l_ in range(L):
        rr = [int(ranks[l_ * n + em.index(e)]) for e in fair]
        per_layer.append({"layer": l_, "median_own_rank": float(np.median(rr))})
    onset = next((r["layer"] for r in per_layer if r["median_own_rank"] < 10000), None)
    raw_ranks = {e: int(ranks[L * n + em.index(e)]) for e in fair}
    table1 = {}
    for e in TABLE1_EMOS:
        if e not in em: continue
        i = em.index(e)
        table1[e] = {"top": [VOCAB[t] for t in top_id[i][:5]],
                     "bottom": [VOCAB[t] for t in bot_id[i][:5]]}
    S = {"cell": cell, "analysis_layer": layer_A, "n_layers": L,
         "fair_selfrank_n": len(fair), "onset_layer_median_lt_10k": onset,
         "onset_fractional_depth": (onset / L if onset is not None else None),
         "per_layer_median_own_rank": per_layer,
         "analysis_layer_raw_selfranks_median": float(np.median(list(raw_ranks.values()))),
         "table1_raw_analysis_layer": table1,
         "conventions": "raw v@W_U for Table-1; (1+w) norm lens (eps 1e-6) for sweep; cleaned vectors; own token = ' word' single-token (140/171)"}
    (OUT / cell / "lens_summary.json").write_text(json.dumps(S, indent=2))
    med = {r["layer"]: r["median_own_rank"] for r in per_layer}
    print(f"DONE [{cell}] onset L{onset} (frac {S['onset_fractional_depth']:.3f}) | "
          f"medians around onset: " + ", ".join(f"L{l_}:{med[l_]:.0f}" for l_ in sorted(med) if abs(l_ - (onset or 0)) <= 2))


if __name__ == "__main__":
    main()
