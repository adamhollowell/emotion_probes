#!/usr/bin/env python3
"""(C) P-lens-axis -- logit-lens promoted/suppressed tokens for the valence axis
and polar emotions, base<->IT.

Paper arXiv 2605.30232 sec 3.2 / Table 2 / App B. Extends G6.1 (which reads only
*promoted* tokens per *emotion*): here we read top-k PROMOTED and SUPPRESSED
tokens for (i) the emotion valence axis (PC1, sign-fixed to +valence) and (ii) a
few polar emotion vectors, per cell. Tests reported as numbers (not graded):
whether tuning sharpens a coherent token theme (base<->IT promoted-set overlap on
the valence axis) and whether promoted/suppressed lists mirror across the
antiparallel poles (positive-pole promoted vs negative-pole suppressed).

Machinery reuses the registered C5 lens package (phase4_lens.py):
  * per-cell unembedding W_U from <cell>__lens_ingredients.pt (anchor 27b-base =
    the repo-root f32 lens_ingredients.pt, token-major (256000, d)); vocab is
    shared across the Gemma-2 family (same tokenizer) so the anchor vocab maps
    tokens for all cells.
  * two conventions, both as in the package (NO bias, matching P4.1/P4.3):
      raw      : logits = d @ W_U                       (Table-1 / P4.1 readout)
      normlens : logits = ((d/rms(d)) * (1+ln_w)) @ W_U (P4.3, eps=1e-6)
  * W_U is a model weight -> basis-immune; the emotion vectors / PC1 axis are
    100T (guarded).

COVERAGE: per-cell W_U is persisted for ALL SIX cells, so this is a real
base<->IT contrast at every scale (no anchor-only fallback needed).

Producer reports numbers only; does not grade. AX firewall: 'loving' may appear
in the JSON but is not interpreted.
Output: baseit_welfarepaper_scan/p_lens_axis_baseit.json
"""
import json
import sys
import zipfile

import numpy as np
import torch

import common as C
from common import nf

sys.path.insert(0, str(C.REPO / "gemma_grid" / "scripts"))
import phase4_lens_preflight as pf  # noqa: E402  (_plain vocab loader, torch-free)

EPS = 1e-6
CHUNK = 16000
V_SIZE = 256000
K_REPORT = 15
K_BUF = 25


def load_vocab():
    vdict = pf._plain(C.REPO / "data/vectors/lens_ingredients.pt", "vocab")
    assert len(vdict) == V_SIZE, f"vocab {len(vdict)} != {V_SIZE}"
    return [vdict[i] for i in range(V_SIZE)]


def wu_chunks(cell, d):
    """Yield (c0, c1, Wb(chunk,d) float32) sequentially over the whole vocab."""
    if cell == "gemma-2-27b-base":
        zf = zipfile.ZipFile(C.REPO / C.CELLS[cell]["lens"])
        name = [n for n in zf.namelist() if n.endswith("data/0")][0]
        row_b = d * 4  # f32 token-major (V_SIZE, d)
        with zf.open(name) as f:
            c0 = 0
            while c0 < V_SIZE:
                blk = min(CHUNK, V_SIZE - c0)
                Wb = np.frombuffer(f.read(blk * row_b), np.float32).reshape(blk, d)
                yield c0, c0 + blk, Wb
                c0 += blk
    else:
        l = torch.load(C.REPO / C.CELLS[cell]["lens"], map_location="cpu",
                       weights_only=False, mmap=True)
        WU = l["W_U"]  # (d, vocab)
        assert WU.shape[0] == d, f"{cell} W_U d {WU.shape[0]} != {d}"
        c0 = 0
        while c0 < V_SIZE:
            c1 = min(c0 + CHUNK, V_SIZE)
            yield c0, c1, WU[:, c0:c1].float().numpy().T.copy()
            c0 = c1


def get_ln(cell, d):
    if cell == "gemma-2-27b-base":
        # anchor ln_final = data/1 (4608 f32) in the root bundle
        zf = zipfile.ZipFile(C.REPO / C.CELLS[cell]["lens"])
        ln_raw = zf.read([n for n in zf.namelist() if n.endswith("data/1")][0])
        ln = np.frombuffer(ln_raw, np.float32)
    else:
        l = torch.load(C.REPO / C.CELLS[cell]["lens"], map_location="cpu",
                       weights_only=False, mmap=True)
        ln = l["ln_final_w"].float().numpy()
    assert ln.shape[0] == d, f"{cell} ln_final {ln.shape[0]} != {d}"
    return ln.astype(np.float64)


def build_directions(cell):
    """Return (dir_matrix (D,d) float32, labels list, meta) for one cell.
    Each underlying direction contributes two rows: raw and normlens forms."""
    em, V, layer = C.load_cell(cell)
    Vl = V[:, layer, :].astype(np.float64)
    idx, pl, ar = nf.load_ratings(str(C.RATINGS), em)
    idx = np.asarray(idx)

    # signed valence PC1 direction in residual space (unit vector, +valence)
    Vc = Vl - Vl.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    pc1 = Vt[0]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = Vc @ pc1  # spurious numpy-2.0 matmul FP-flag warnings; result verified finite
    if np.corrcoef(scores[idx], pl)[0, 1] < 0:
        pc1 = -pc1

    # polar emotion vectors: 2 highest + 2 lowest RM valence (raw cleaned vectors)
    order = np.argsort(pl)
    pos_pole = [em[idx[order[-1]]], em[idx[order[-2]]]]
    neg_pole = [em[idx[order[0]]], em[idx[order[1]]]]
    underlying = [("valence_axis_pc1", pc1)]
    for e in pos_pole + neg_pole:
        underlying.append((f"emotion:{e}", Vl[em.index(e)]))

    ln = get_ln(cell, Vl.shape[1])
    dirs, labels = [], []
    for name, vec in underlying:
        vec = np.asarray(vec, np.float64)
        dirs.append(vec)                      # raw form
        labels.append((name, "raw"))
        rms = np.sqrt((vec ** 2).mean() + EPS)
        dirs.append((vec / rms) * (1.0 + ln))  # normlens form
        labels.append((name, "normlens"))
    meta = dict(layer=layer, pos_pole=pos_pole, neg_pole=neg_pole,
                d_model=Vl.shape[1], n_layers=V.shape[1])
    return np.asarray(dirs, np.float32), labels, meta


def topk_stream(cell, dir_mat):
    D = dir_mat.shape[0]
    d = dir_mat.shape[1]
    tv = np.full((D, K_BUF), -np.inf, np.float32); ti = np.zeros((D, K_BUF), np.int64)
    bv = np.full((D, K_BUF), np.inf, np.float32);  bi = np.zeros((D, K_BUF), np.int64)
    Dt = torch.from_numpy(dir_mat)
    torch.set_num_threads(4)
    for c0, c1, Wb in wu_chunks(cell, d):
        logits = (Dt @ torch.from_numpy(np.ascontiguousarray(Wb)).T).numpy()  # (D, chunk)
        ids = np.tile(np.arange(c0, c1), (D, 1))
        cat_v = np.concatenate([tv, logits], 1); cat_i = np.concatenate([ti, ids], 1)
        sel = np.argsort(-cat_v, 1)[:, :K_BUF]
        tv = np.take_along_axis(cat_v, sel, 1); ti = np.take_along_axis(cat_i, sel, 1)
        cat_v = np.concatenate([bv, logits], 1); cat_i = np.concatenate([bi, ids], 1)
        sel = np.argsort(cat_v, 1)[:, :K_BUF]
        bv = np.take_along_axis(cat_v, sel, 1); bi = np.take_along_axis(cat_i, sel, 1)
    return tv, ti, bv, bi


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return float(len(sa & sb) / len(sa | sb)) if (sa or sb) else 0.0


def run_cell(cell, vocab):
    dir_mat, labels, meta = build_directions(cell)
    tv, ti, bv, bi = topk_stream(cell, dir_mat)
    readouts = {}
    for r, (name, conv) in enumerate(labels):
        readouts.setdefault(name, {})[conv] = {
            "promoted": [vocab[t] for t in ti[r][:K_REPORT]],
            "suppressed": [vocab[t] for t in bi[r][:K_REPORT]],
            "promoted_ids": [int(t) for t in ti[r][:K_REPORT]],
        }
    return {
        "cell": cell, "scale": C.CELLS[cell]["scale"],
        "tuning": C.CELLS[cell]["tuning"], "analysis_layer": meta["layer"],
        "d_model": meta["d_model"], "n_layers": meta["n_layers"],
        "positive_pole_emotions": meta["pos_pole"],
        "negative_pole_emotions": meta["neg_pole"],
        "readouts": readouts,
    }


def main():
    vocab = load_vocab()
    cells = list(C.CELLS)
    rows = {}
    for c in cells:
        print(f"[lens] {c} ...", flush=True)
        rows[c] = run_cell(c, vocab)
        va = rows[c]["readouts"]["valence_axis_pc1"]["raw"]
        print(f"    valence-axis promoted(raw): {va['promoted'][:6]}")

    # --- reported tests (numbers only) ---
    # theme sharpening: base<->IT promoted-set overlap on the valence axis
    theme = {}
    for scale in ("2b", "9b", "27b"):
        b = rows[f"gemma-2-{scale}-base"]["readouts"]["valence_axis_pc1"]
        it = rows[f"gemma-2-{scale}-it"]["readouts"]["valence_axis_pc1"]
        theme[scale] = {conv: {
            "jaccard_promoted_base_vs_it": jaccard(b[conv]["promoted"], it[conv]["promoted"]),
            "jaccard_suppressed_base_vs_it": jaccard(b[conv]["suppressed"], it[conv]["suppressed"]),
        } for conv in ("raw", "normlens")}

    # pole mirroring: positive-pole promoted vs negative-pole suppressed (per cell)
    mirror = {}
    for c in cells:
        ro = rows[c]["readouts"]
        pos = f"emotion:{rows[c]['positive_pole_emotions'][0]}"
        neg = f"emotion:{rows[c]['negative_pole_emotions'][0]}"
        mirror[c] = {conv: {
            "pos_pole": pos, "neg_pole": neg,
            "jaccard_pospromoted_vs_negsuppressed":
                jaccard(ro[pos][conv]["promoted"], ro[neg][conv]["suppressed"]),
            "jaccard_possuppressed_vs_negpromoted":
                jaccard(ro[pos][conv]["suppressed"], ro[neg][conv]["promoted"]),
            "jaccard_pospromoted_vs_negpromoted":
                jaccard(ro[pos][conv]["promoted"], ro[neg][conv]["promoted"]),
        } for conv in ("raw", "normlens")}

    out = {
        "producer": "P-lens-axis (C)",
        "paper": "arXiv 2605.30232 sec 3.2 / Table 2 / App B",
        "basis": "emotion vectors / PC1 = 100T (guarded); W_U = model weight (basis-immune)",
        "coverage": {
            "cells_with_real_unembedding": cells,
            "note": ("per-cell W_U persisted for all 6 cells (<cell>__lens_ingredients.pt; "
                     "anchor 27b-base = repo-root lens_ingredients.pt). Full base<->IT "
                     "contrast at every scale; no anchor-only fallback needed."),
        },
        "conventions": ("raw = d @ W_U (Table-1/P4.1); normlens = ((d/rms(d))*(1+ln_w)) @ W_U "
                        f"(P4.3, eps={EPS}); NO bias (matches the C5 package); k={K_REPORT}. "
                        "For the valence axis under raw, suppressed(+PC1) == promoted(-PC1) by "
                        "linearity, so only +PC1 is reported."),
        "vocab_note": "Gemma-2 family shares one tokenizer; anchor vocab maps tokens for all cells",
        "per_cell": rows,
        "tests": {
            "theme_sharpening_valence_axis_base_vs_it": theme,
            "pole_mirroring_per_cell": mirror,
        },
    }
    outp = C.OUT_DIR / "p_lens_axis_baseit.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"wrote {outp}")
    print(f"  sha256 {C.sha256_file(outp)}")


if __name__ == "__main__":
    main()
