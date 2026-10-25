#!/usr/bin/env python3
"""Embedding-table transport baseline for §4.3 (unregistered; review response).

Handoff of record: gemma_grid/drafting/HANDOFF_CLAUDE_CODE_TWO_REVIEW_ANALYSES_100T_2026-10-14.md
(Analysis A). Same procedure as the frozen P-map transport with only the vectors
changed: the static embedding-table vectors (geometry_out_100T/<cell>/c3_vectors.pt,
cleaned against the 100T neutral-embedding basis, k=3) replace the contextual
analysis-layer vectors.

The transport machinery (fit_procrustes, center, cosine_rows, eval_transport) and
the frozen constants (FOLD_SEED, N_FOLDS, PRIMARY_*, ANALYSIS_LAYERS) are IMPORTED
from nogpu_additions_100T/pmap_transport_100T.py, whose sha256 is asserted below.
Nothing in that module is modified.

Order of work:
  1. GATE: primary pair (2b-base -> 27b-base) on the contextual vectors through
     this wrapper must reproduce pmap_transport_100T.json. Failure -> exit, no output.
  2. Contextual vectors, all 30 ordered pairs (the stored artifact holds 16 of them;
     the recomputed 16 are diffed against it, the other 14 are new and labeled so).
  3. Static-table vectors, all 30 ordered pairs, per-emotion rank and cosine.
  4. Primary pair, static table: correspondence-shuffled null, N_NULL=500, seed 0,
     statistic top-1. SHAPE PROBE ONLY (Adam's amendment A, 2026-10-14): no p value,
     no comparison to the §4.3 bar, no pass/fail.

Producer reports numbers only; it does not grade.
Run from the project root with VECLIB_MAXIMUM_THREADS=1 (workers are processes).
"""
import argparse, hashlib, json, os, platform, sys, time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
GG = REPO / "gemma_grid"
sys.path.insert(0, str(GG / "nogpu_additions_100T"))
sys.path.insert(0, str(GG / "scripts"))

FROZEN_MODULE = GG / "nogpu_additions_100T" / "pmap_transport_100T.py"
FROZEN_MODULE_SHA = "ee736f5d36c9a04c8d6e4eaf043d932cbb979a5e4fba22a0ea0e8613b91e5526"
ANCHOR_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
CTX_ARTIFACT = GG / "geometry_out_100T" / "pmap_transport_100T.json"
STAGEA_100T = GG / "scripts" / "phase4_stageA_100T.py"
N_NULL = 500
NULL_SEED = 0
GATE_COS_TOL = 1e-9
# Adam's amendment A, 2026-10-14: the null is a shape probe only. No p value, no
# comparison to the §4.3 bar, no pass/fail.
NULL_LABEL = ("SHAPE PROBE ONLY (amendment A, 2026-10-14): describes the shape of the "
              "correspondence-shuffled null at 500 draws. Not the frozen 10,000-draw floor; "
              "not compared to the §4.3 bar; no p value is computed or reported; no pass/fail.")

import pmap_transport_100T as P  # noqa: E402  (frozen; imported, not edited)

CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base",
         "gemma-2-9b-it", "gemma-2-27b-base", "gemma-2-27b-it"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def size_of(cell):
    return cell.split("-")[2]


def load_pt(path):
    import torch
    b = torch.load(path, map_location="cpu", weights_only=False)
    return [str(e) for e in b["emotions"]], b["vectors"]


def load_contextual(path, layer):
    em, V = load_pt(path)
    V = np.asarray(V[:, layer, :], dtype=np.float64)
    assert V.shape[0] == 171, V.shape
    return em, P.center(V)


def load_static(path):
    em, V = load_pt(path)
    assert tuple(V.shape[:2]) == (171, 1), f"unexpected c3 shape {tuple(V.shape)} at {path}"
    V = np.asarray(V, dtype=np.float64)[:, 0, :]
    return em, P.center(V)


def summarize(ranks, coss):
    return {"median_rank": float(np.median(ranks)),
            "top1_acc": float(np.mean(ranks == 1)), "top5_acc": float(np.mean(ranks <= 5)),
            "median_cosine": float(np.median(coss)),
            "top1_count": int(np.sum(ranks == 1)), "top5_count": int(np.sum(ranks <= 5))}


# ---- worker plumbing (spawn context; each worker holds the two matrices) ----
_W = {}

def _init(X, Y, fold_ids):
    _W["X"], _W["Y"], _W["f"] = X, Y, fold_ids

def _pair_job(args):
    key, X, Y, fold_ids = args
    r, c = P.eval_transport(X, Y, fold_ids, "procrustes")
    return key, r, c

def _null_job(perm):
    r, _ = P.eval_transport(_W["X"], _W["Y"][perm], _W["f"], "procrustes")
    return int(np.sum(r == 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(GG / "geometry_out_100T" / "pmap_embedding_table_transport_100T.json"))
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()

    assert sha256(FROZEN_MODULE) == FROZEN_MODULE_SHA, "frozen pmap_transport_100T.py hash mismatch"
    import phase4_stageA_100T as SA  # HA_VEC: the 100T contextual paths of record
    ctx_paths = {c: REPO / SA.HA_VEC[c] for c in CELLS}
    st_paths = {c: GG / "geometry_out_100T" / c / "c3_vectors.pt" for c in CELLS}
    for c in CELLS:
        assert "100T" in str(ctx_paths[c]) and "100T" in str(st_paths[c]), c
    assert sha256(ctx_paths["gemma-2-27b-base"]) == ANCHOR_SHA, "anchor hash-gate (6de2e293) failed"

    ctx, st, em_ref = {}, {}, None
    for c in CELLS:
        layer = P.ANALYSIS_LAYERS[size_of(c)]
        e1, ctx[c] = load_contextual(ctx_paths[c], layer)
        e2, st[c] = load_static(st_paths[c])
        if em_ref is None:
            em_ref = e1
        assert e1 == em_ref and e2 == em_ref, f"emotion list mismatch at {c}"

    fold_ids = np.random.RandomState(P.FOLD_SEED).permutation(np.arange(171) % P.N_FOLDS)
    stored = json.loads(CTX_ARTIFACT.read_text())["pairs"]
    prim = f"{P.PRIMARY_SOURCE}->{P.PRIMARY_TARGET}"
    pairs = [(P.PRIMARY_SOURCE, P.PRIMARY_TARGET)] + [
        (s, t) for s in CELLS for t in CELLS if s != t and (s, t) != (P.PRIMARY_SOURCE, P.PRIMARY_TARGET)]
    assert len(pairs) == 30

    ctxpool = get_context("spawn")
    with ctxpool.Pool(a.workers) as pool:
        # ---- 1. GATE -------------------------------------------------------
        _, r, c = _pair_job((prim, ctx[P.PRIMARY_SOURCE], ctx[P.PRIMARY_TARGET], fold_ids))
        g = summarize(r, c); s = stored[prim]
        gate = {"pair": prim, "reproduced": g,
                "stored": {k: s[k] for k in ("median_rank", "top1_acc", "top5_acc", "median_cosine")},
                "abs_diff_median_cosine": abs(g["median_cosine"] - s["median_cosine"]),
                "tolerance_median_cosine": GATE_COS_TOL}
        gate["pass"] = bool(g["median_rank"] == s["median_rank"] == 1.0 and g["top1_acc"] == s["top1_acc"] == 1.0
                            and g["top5_acc"] == s["top5_acc"] and gate["abs_diff_median_cosine"] <= GATE_COS_TOL)
        print("GATE:", json.dumps(gate, indent=1), flush=True)
        if not gate["pass"]:
            sys.exit("GATE FAILED — stopping before any new computation (no output written)")

        # ---- 2 + 3. all 30 ordered pairs, contextual and static -------------
        jobs = [(f"ctx|{s}->{t}", ctx[s], ctx[t], fold_ids) for s, t in pairs] + \
               [(f"st|{s}->{t}", st[s], st[t], fold_ids) for s, t in pairs]
        res = {k: (r, c) for k, r, c in pool.imap_unordered(_pair_job, jobs)}
        print(f"pairs done {time.time()-t0:.0f}s", flush=True)

    out_pairs, ctx_check = {}, {}
    for s_, t_ in pairs:
        key = f"{s_}->{t_}"
        rs, cs = res[f"st|{key}"]; rc, cc = res[f"ctx|{key}"]
        ctx_sum = summarize(rc, cc)
        if key in stored:
            src = "stored artifact pmap_transport_100T.json"
            ctx_rep = {k: stored[key][k] for k in ("median_rank", "top1_acc", "top5_acc", "median_cosine")}
            ctx_check[key] = {k: abs(ctx_sum[k] - stored[key][k]) for k in ctx_rep}
        else:
            src = "recomputed here (pair not in stored artifact; frozen script runs a<b pairs only)"
            ctx_rep = {k: ctx_sum[k] for k in ("median_rank", "top1_acc", "top5_acc", "median_cosine")}
        out_pairs[key] = {
            "static_table": summarize(rs, cs),
            "contextual": ctx_rep, "contextual_source": src,
            "contextual_recomputed": ctx_sum,
            "static_per_emotion": {"rank": rs.tolist(), "cosine": [float(x) for x in cs]},
        }

    # ---- 4. null on the primary pair, static table ------------------------
    prs = np.random.RandomState(NULL_SEED)
    perms = [prs.permutation(171) for _ in range(N_NULL)]
    Xs, Yt = st[P.PRIMARY_SOURCE], st[P.PRIMARY_TARGET]
    with ctxpool.Pool(a.workers, initializer=_init, initargs=(Xs, Yt, fold_ids)) as pool:
        counts = np.array(pool.map(_null_job, perms, chunksize=5))   # map keeps perm order
    print(f"null done {time.time()-t0:.0f}s", flush=True)
    obs_c = out_pairs[prim]["static_table"]["top1_count"]
    q = lambda p_: float(np.percentile(counts, p_))
    null = {"label": NULL_LABEL,
            "n_null": N_NULL, "seed": NULL_SEED, "statistic": "top-1 count of 171 (rate = count/171)",
            "construction": "RandomState(0).permutation(171) per draw, applied to target rows (as frozen run_pair)",
            "observed_top1_count": obs_c, "observed_top1_rate": obs_c / 171,
            "counts": {"max": int(counts.max()), "p99": q(99), "p90": q(90),
                       "median": float(np.median(counts)), "mean": float(counts.mean())},
            "rates": {"max": counts.max() / 171, "p99": q(99) / 171, "p90": q(90) / 171,
                      "median": float(np.median(counts)) / 171, "mean": float(counts.mean()) / 171},
            "n_at_or_above_observed": int(np.sum(counts >= obs_c)),
            "percentile_method": "numpy default (linear)",
            "null_counts": counts.tolist()}

    out = {"analysis": "A: embedding-table transport baseline (unregistered, review response)",
           "handoff": "gemma_grid/drafting/HANDOFF_CLAUDE_CODE_TWO_REVIEW_ANALYSES_100T_2026-10-14.md",
           "frozen_imported": {"fold_seed": P.FOLD_SEED, "n_folds": P.N_FOLDS,
                               "primary": prim, "analysis_layers": P.ANALYSIS_LAYERS,
                               "fold_construction": "RandomState(FOLD_SEED).permutation(arange(171) % N_FOLDS)"},
           "emotions": em_ref, "gate": gate,
           "contextual_recompute_vs_stored_max_abs_diff": {
               k: max(v[k] for v in ctx_check.values()) for k in ("median_rank", "top1_acc", "top5_acc", "median_cosine")},
           "contextual_pairs_in_stored_artifact": len(ctx_check),
           "pairs": out_pairs, "primary_static_null": null}
    op = Path(a.out); op.write_text(json.dumps(out, indent=1))

    import torch
    man = {"output": str(op.relative_to(REPO)), "output_sha256": sha256(op),
           "script": str(Path(__file__).resolve().relative_to(REPO)), "script_sha256": sha256(__file__),
           "command": " ".join([sys.executable] + sys.argv),
           "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "wall_s": round(time.time() - t0, 1),
           "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                             [*ctx_paths.values(), *st_paths.values(), CTX_ARTIFACT, FROZEN_MODULE,
                              STAGEA_100T, GG / "scripts" / "phase4_stageA.py", GG / "scripts" / "noise_floor.py"]},
           "seeds": {"fold_seed": P.FOLD_SEED, "null_seed": NULL_SEED, "n_null": N_NULL},
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform(), "blas": "Apple Accelerate (numpy default on macOS arm64)",
                        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"), "workers": a.workers},
           "precision": "float64 (vectors stored float32, cast as the frozen loader does)"}
    mp = op.with_name(op.stem + "_manifest.json"); mp.write_text(json.dumps(man, indent=1))
    print("wrote", op, man["output_sha256"]); print("wrote", mp)


if __name__ == "__main__":
    main()
