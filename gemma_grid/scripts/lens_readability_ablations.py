#!/usr/bin/env python3
"""Layer-0 readability checks for the vocabulary readout (§6.2, Appendix E).

Written for the release on 2026-10-26. It produces the figures the paper prints for these checks.

The readout is phase4_lens.py's, imported unchanged: the same W_U reader, final-norm weight, eps and
own-token map, the own-word rank 1 + #(logits > own logit) over the 256,000-token vocabulary, and the
median over the 140 emotions whose " word" form is a single token. Every arm reads layer 0.

  reference   cleaned vectors through the (1 + w) norm lens. Gate: each cell's median must equal
              L0_median_rank in lens_cross_cell.json, which phase4_lens.py wrote. The run stops
              before writing if any cell differs.
  no_norm     cleaned vectors, raw v @ W_U with no final normalization.
  no_cleanup  uncleaned vectors (the vectors_raw key) through the (1 + w) norm lens.
  random      N_SETS sets of 140 standard Gaussian directions in the cell's width through the
              (1 + w) norm lens. Direction j of a set is scored against fair emotion j's own token.
              The generator is reseeded with SEED for each cell, so the two cells of one width
              read the same directions.

Logits are computed as phase4_lens.py computes them (float32 rows, the own logit by a NumPy dot,
the vocabulary in chunks of 8,000 through torch with 4 threads), so the reference arm reproduces
the stored medians exactly.

Run from the repository root with LENS_ASSET_DIR set as for phase4_lens.py (it holds
anchor_vocab.json, emotion_token_ids.json, anchor_ln_final.npy and the anchor's f16 W_U copy,
anchor_WU_f16.npy, written by phase4_lens.py --convert-anchor):
    LENS_ASSET_DIR=<dir> python gemma_grid/scripts/lens_readability_ablations.py
    LENS_ASSET_DIR=<dir> python gemma_grid/scripts/lens_readability_ablations.py --cell gemma-2-2b-base
A run with --cell writes that cell's partial result to the scratch folder. A run without --cell
does every cell not yet done and then writes the output and its manifest.
"""
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import phase4_lens as L                                     # noqa: E402  (the readout, imported unchanged)

CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base", "gemma-2-9b-it",
         "gemma-2-27b-base", "gemma-2-27b-it"]
VEC = {c: f"gemma_grid/data/{c}/vectors/canonical.pt" for c in CELLS}
VEC["gemma-2-27b-base"] = "data/vectors/emotion_vectors_171.pt"
WIDTH = {"gemma-2-2b-base": 2304, "gemma-2-2b-it": 2304, "gemma-2-9b-base": 3584,
         "gemma-2-9b-it": 3584, "gemma-2-27b-base": 4608, "gemma-2-27b-it": 4608}
REFERENCE = REPO / "gemma_grid" / "geometry_out" / "lens_cross_cell.json"
OUT = REPO / "gemma_grid" / "geometry_out" / "lens_readability_ablations.json"
SCRATCH = Path(os.environ.get("BOOT_SCRATCH", ".scratch")) / "lens_ablation_scratch"
LAYER = 0
SEED = 20261026
N_SETS = 10
ARMS = ["reference", "no_norm", "no_cleanup"]


def anchor_f16():
    d = Path(os.environ["LENS_ASSET_DIR"])
    for p in (d / "anchor_WU_f16.npy", d / "lens_scratch" / "anchor_WU_f16.npy"):
        if p.exists():
            return p
    sys.exit("STOP: anchor_WU_f16.npy not found under LENS_ASSET_DIR; run phase4_lens.py --convert-anchor first")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def summary(r):
    r = np.asarray(r)
    return {"median": float(np.median(r)), "q10": float(np.percentile(r, 10)), "q25": float(np.percentile(r, 25)),
            "q75": float(np.percentile(r, 75)), "frac_below_10000": float(np.mean(r < 10000)), "n": int(r.size)}


def run_cell(cell):
    t0 = time.time()
    p = torch.load(REPO / VEC[cell], map_location="cpu", weights_only=False)
    em = list(p["emotions"])
    Vc = p["vectors"].float().numpy().astype(np.float64)[:, LAYER, :]
    Vr = p["vectors_raw"].float().numpy().astype(np.float64)[:, LAYER, :]
    n, d = Vc.shape
    assert d == WIDTH[cell], (cell, d)
    ln = L.get_ln(cell); assert ln.shape[0] == d
    fair = [e for e in em if e in L.EMO_TID]
    fi = [em.index(e) for e in fair]
    tid = np.array([L.EMO_TID[e] for e in fair], np.int64)
    k = len(fair)

    def norm(X):
        return (X / np.sqrt((X ** 2).mean(axis=1, keepdims=True) + L.EPS)) * (1.0 + ln)

    G = np.random.default_rng(SEED).standard_normal((N_SETS * k, d))
    stack = np.concatenate([norm(Vc[fi]), Vc[fi], norm(Vr[fi]), norm(G)], axis=0).astype(np.float32)
    row_tid = np.concatenate([tid, tid, tid, np.tile(tid, N_SETS)])
    R = stack.shape[0]

    rows_needed = np.unique(row_tid)
    W_own = np.stack([L.wu_chunk(cell, t, t + 1)[0] for t in rows_needed])
    idx_of = {t: i for i, t in enumerate(rows_needed)}
    own = np.array([float(stack[r] @ W_own[idx_of[row_tid[r]]]) for r in range(R)], np.float32)

    torch.set_num_threads(4)
    S_t = torch.from_numpy(stack)
    counts = np.zeros(R, np.int64)
    for c0 in range(0, L.V_SIZE, L.CHUNK):
        c1 = min(c0 + L.CHUNK, L.V_SIZE)
        logits = (S_t @ torch.from_numpy(L.wu_chunk(cell, c0, c1)).T).numpy()
        counts += (logits > own[:, None]).sum(axis=1)
    ranks = counts + 1

    res = {"width": d, "fair_n": k, "seconds": round(time.time() - t0, 1)}
    for i, arm in enumerate(ARMS):
        rr = ranks[i * k:(i + 1) * k]
        res[arm] = summary(rr)
        res[arm]["per_emotion_rank"] = {e: int(v) for e, v in zip(fair, rr)}
    rnd = ranks[3 * k:].reshape(N_SETS, k)
    res["random"] = summary(rnd.ravel())
    res["random"]["per_set_median"] = [float(np.median(s)) for s in rnd]
    res["random"]["ranks"] = [int(v) for v in rnd.ravel()]
    print(f"{cell}: reference {res['reference']['median']}, no_norm {res['no_norm']['median']}, "
          f"no_cleanup {res['no_cleanup']['median']}, random {res['random']['median']} ({res['seconds']}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=CELLS)
    a = ap.parse_args()
    L.ANCHOR_F16 = anchor_f16()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    todo = [a.cell] if a.cell else CELLS
    for c in todo:
        part = SCRATCH / f"{c}.json"
        if not part.exists() or a.cell:
            part.write_text(json.dumps(run_cell(c), indent=1))
    if a.cell:
        return
    cells = {c: json.loads((SCRATCH / f"{c}.json").read_text()) for c in CELLS}

    ref = json.loads(REFERENCE.read_text())
    gate = {c: {"stored_L0_median_rank": ref[c]["L0_median_rank"], "recomputed": cells[c]["reference"]["median"],
                "match": ref[c]["L0_median_rank"] == cells[c]["reference"]["median"]} for c in CELLS}
    if not all(g["match"] for g in gate.values()):
        print(json.dumps(gate, indent=1))
        sys.exit("STOP: the reference arm does not reproduce lens_cross_cell.json; nothing written")

    by_width = {}
    for w in sorted(set(WIDTH.values())):
        cs = [c for c in CELLS if WIDTH[c] == w]
        pooled = sum((cells[c]["random"]["ranks"] for c in cs), [])
        by_width[str(w)] = {"cells": cs, "pooled": summary(pooled),
                            "median_per_cell": {c: cells[c]["random"]["median"] for c in cs}}
    out = {"analysis": "Layer-0 readability checks for the vocabulary readout (§6.2, Appendix E)",
           "basis": "100T", "layer": LAYER, "readout": "phase4_lens.py, imported unchanged",
           "own_word_rank": "1 + #(logits > own logit) over 256,000 tokens; median over the fair emotions (single-token ' word' form)",
           "arms": {"reference": "cleaned vectors, (1 + w) norm lens, eps 1e-6",
                    "no_norm": "cleaned vectors, raw v @ W_U",
                    "no_cleanup": "uncleaned vectors (vectors_raw), (1 + w) norm lens",
                    "random": f"{N_SETS} sets x fair_n standard Gaussian directions, (1 + w) norm lens, numpy default_rng({SEED}) reseeded per cell"},
           "chance_median": L.V_SIZE / 2,
           "reference_gate": gate,
           "cells": cells,
           "random_by_width": by_width}
    OUT.write_text(json.dumps(out, indent=1))
    lens_src = {c: (REPO / "data/vectors/lens_ingredients.pt") if c == "gemma-2-27b-base" else
                L.LENS[c] for c in CELLS}
    man = {"output": str(OUT.relative_to(REPO)), "output_sha256": sha(OUT),
           "script": str(Path(__file__).resolve().relative_to(REPO)), "script_sha256": sha(Path(__file__).resolve()),
           "readout_module_sha256": sha(Path(L.__file__)),
           "inputs": {VEC[c]: sha(REPO / VEC[c]) for c in CELLS},
           "lens_ingredients": {str(Path(lens_src[c]).relative_to(REPO)): sha(lens_src[c]) for c in CELLS},
           "emotion_token_ids_sha256": sha(Path(os.environ["LENS_ASSET_DIR"]) / "emotion_token_ids.json"),
           "reference_file": str(REFERENCE.relative_to(REPO)), "reference_sha256": sha(REFERENCE),
           "numpy": np.__version__, "torch": torch.__version__, "python": sys.version.split()[0],
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    OUT.with_name(OUT.stem + "_manifest.json").write_text(json.dumps(man, indent=1))
    print("wrote", OUT, man["output_sha256"])
    for c in CELLS:
        r = cells[c]
        print(f"{c:18s} reference {r['reference']['median']:>9}  no_norm {r['no_norm']['median']:>9}  "
              f"no_cleanup {r['no_cleanup']['median']:>9}  random {r['random']['median']:>9}")
    for w, v in by_width.items():
        print(f"width {w}: random median {v['pooled']['median']} (pooled over {', '.join(v['cells'])})")


if __name__ == "__main__":
    main()
