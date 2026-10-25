#!/usr/bin/env python3
"""T.3 9B centered near-null diagnostic — port of diag_T3.py (27B) to 9B.
RAW / basis-immune; matches fixed_probe_tuning_shifts inputs (neutral_100T acts,
IT-cell raw probes). Pre-stated: mirror the 27B rule exactly — top-20 variance-dim union
drop restores centered r_structure iff the near-null is massive-activation-driven."""
import sys, json
from pathlib import Path
import numpy as np
REPO = Path(".").resolve()
sys.path.insert(0, str(REPO))
import ptload

def normr(M): return M / np.clip(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12, None)
def prof(P, A, centered=True, drop=None):
    P2, A2 = P, A
    if drop is not None:
        keep = np.ones(A.shape[1], bool); keep[drop] = False
        A2 = A2[:, keep]; P2 = P[:, keep]
    if centered: A2 = A2 - A2.mean(0, keepdims=True)
    return (normr(P2) @ normr(A2).T).mean(axis=1)
def corr(a, b): return float(np.corrcoef(a, b)[0, 1])

res = json.load(open("gemma_grid/geometry_out/fixed_probe_tuning_shifts_results.json"))
L = 28
pv = ptload.load("gemma_grid/data/gemma-2-9b-it/vectors/canonical.pt")
em = list(pv["emotions"]); P = np.asarray(pv["vectors_raw"], dtype=np.float32)[:, L, :]; del pv
Ab = np.asarray(ptload.load("gemma_grid/data/gemma-2-9b-base/vectors/gemma-2-9b-base__neutral.pt")["acts"], dtype=np.float32)[:, L, :]
Ai = np.asarray(ptload.load("gemma_grid/data/gemma-2-9b-it/vectors/gemma-2-9b-it__neutral.pt")["acts"], dtype=np.float32)[:, L, :]

out = {"job": "diag_T3_9b (T.3 9B centered near-null)", "basis": "RAW/basis-immune; neutral_100T acts, 9b-it raw probes, L28",
       "target_centered_r_structure_from_artifact": res["9B"]["ITprobes_centered"]["r_structure_base_vs_it"]}

# validate reproduction of the centered near-null
r_cent_none = corr(prof(P, Ab), prof(P, Ai))
r_uncent = corr(prof(P, Ab, centered=False), prof(P, Ai, centered=False))
out["reproduce"] = {"centered_none": round(r_cent_none, 4), "uncentered": round(r_uncent, 4),
                    "artifact_centered": res["9B"]["ITprobes_centered"]["r_structure_base_vs_it"],
                    "artifact_uncentered": res["9B"]["ITprobes_uncentered"]["r_structure_base_vs_it"]}

# massive-activation dims
vb = (Ab - Ab.mean(0)).var(0); vi = (Ai - Ai.mean(0)).var(0)
topb = np.argsort(-vb)[:20]; topi = np.argsort(-vi)[:20]; union20 = np.union1d(topb, topi)
out["massive_dims"] = {"n_union20": int(len(union20)),
                       "it_top5_dims": topi[:5].tolist(), "base_top5_dims": topb[:5].tolist(),
                       "it_top5_var_share_pct": round(float(vi[topi[:5]].sum()/vi.sum()*100), 2),
                       "dim4140_in_union": bool(4140 in union20),
                       "dim4140_var_rank_it": int(np.where(np.argsort(-vi) == 4140)[0][0]) if 4140 < len(vi) else None}

# DROP test — the mechanism gate
rows = {}
for tag, drop in (("none", None), ("it-top20", topi), ("union", union20)):
    pb, pi = prof(P, Ab, drop=drop), prof(P, Ai, drop=drop)
    rows[tag] = {"r_structure": round(corr(pb, pi), 4), "SD_base": round(float(pb.std()), 5), "SD_it": round(float(pi.std()), 5)}
out["drop_test_centered_r_structure"] = rows

# odd/even reliability after union drop (rule out noise-restore)
keep = np.setdiff1d(np.arange(Ab.shape[1]), union20)
def prof_sl(P, A, sl): return prof(P[:, keep], A[sl][:, keep])
out["reliability_after_union_drop_oddeven"] = {"base": round(corr(prof_sl(P, Ab, slice(0, None, 2)), prof_sl(P, Ab, slice(1, None, 2))), 4),
                                               "it": round(corr(prof_sl(P, Ai, slice(0, None, 2)), prof_sl(P, Ai, slice(1, None, 2))), 4)}

# cross-size uncentered-shift consistency, full vs union-dropped (does NOT explain collapse — reported only)
d9_full = prof(P, Ai, centered=False) - prof(P, Ab, centered=False)
d9_drop = prof(P, Ai, centered=False, drop=union20) - prof(P, Ab, centered=False, drop=union20)
d2 = np.array([res["2B"]["diff_full"][e] for e in em]); d27 = np.array([res["27B"]["diff_full"][e] for e in em])
out["cross_size_uncentered_consistency"] = {"2B_vs_9B_full": round(corr(d2, d9_full), 4), "2B_vs_9B_dropped": round(corr(d2, d9_drop), 4),
                                            "9B_vs_27B_full": round(corr(d9_full, d27), 4), "9B_vs_27B_dropped": round(corr(d9_drop, d27), 4),
                                            "note": "cross-size collapse NOT claimed explained (mirrors 27B note)"}

Path("gemma_grid/geometry_out").mkdir(exist_ok=True)
open("gemma_grid/geometry_out/diag_T3_9b_result.json", "w").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
