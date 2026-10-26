#!/usr/bin/env python3
"""Layer stability under cosine instead of Pearson (Appendix H; the measure of §8.3).

Written for the release on 2026-10-26. The registered layer-stability measure compares the pairwise
cosine structure of the 171 emotions across layers. At each of 14 layers sampled evenly over depth,
the upper triangle of the 171 x 171 cosine matrix is one row. The measure is the mean off-diagonal
Pearson correlation between those rows, over all 14 sampled layers (all-layer) or over the sampled
layers at or past half depth (mid-late). This script computes it both ways, Pearson as registered
and uncentered cosine between the same rows, with nothing else changed: the same vectors, the same
sampled layers and the same paired case bootstrap as phase4_layerstab_ci.py (2,000 draws, seed 0,
self-duplicate pairs excluded, both measures on the same draws).

It reports, under both measures and with the absolute difference:
  per cell             the mid-late and all-layer values;
  contrasts            tuning mid-late pooled, tuning all-layer pooled, scale mid-late base and scale
                       mid-late tuned (2B to 27B), each with its per-size tuning values where pooled,
                       its 95 percent bootstrap interval, and its status against the 0.01 floor;
  layer 45 excluded    the same mid-late contrasts with absolute layer 45 dropped from the 27B cells,
                       as phase4_layerstab_ci_exclL45.py does;
  maximum difference   over every reported value, and over the point values alone.

Before writing it checks the Pearson side against the stored results: each cell's values against
stageA_summary.json to within 1e-8 (the stored values come from
another machine's float32 arithmetic, and the deviation found is recorded), and every Pearson contrast and interval against
layerstab_ci.json and layerstab_ci_exclL45.json at their stored five decimals.

    python gemma_grid/scripts/layerstab_cosine_check.py
"""
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf                                    # noqa: E402  (load_vectors and cos, as the producer)

GEO = REPO / "gemma_grid" / "geometry_out"
OUT = GEO / "layerstab_cosine_check.json"
CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base", "gemma-2-9b-it",
         "gemma-2-27b-base", "gemma-2-27b-it"]
VEC = {c: f"gemma_grid/data/{c}/vectors/canonical.pt" for c in CELLS}
VEC["gemma-2-27b-base"] = "data/vectors/emotion_vectors_171.pt"
REF_CI = GEO / "layerstab_ci.json"
REF_EXCL = GEO / "layerstab_ci_exclL45.json"
TUNING = [("gemma-2-2b-base", "gemma-2-2b-it"), ("gemma-2-9b-base", "gemma-2-9b-it"),
          ("gemma-2-27b-base", "gemma-2-27b-it")]
CHAINS = {"base": ("gemma-2-2b-base", "gemma-2-27b-base"), "it": ("gemma-2-2b-it", "gemma-2-27b-it")}
B, SEED, FLOOR, EXCL_LAYER = 2000, 0, 0.01, 45
GATE_TOL = 1e-8
KINDS = ["pearson", "cosine"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cell_mats(V, excl=False):
    n, L, d = V.shape
    samp = sorted(set(np.linspace(0, L - 1, min(14, L)).round().astype(int).tolist()))
    if excl:
        samp = [l for l in samp if l != EXCL_LAYER]
    mats = np.stack([nf.cos(V[:, l, :]) for l in samp])
    ml = np.array([k for k, l in enumerate(samp) if l >= L // 2])
    return {"mats": mats, "ml": ml, "all": np.arange(len(samp)), "samp": samp}


def stats(c, ii, jj):
    """{(kind, which): value} for one cell on one pair set."""
    tris = c["mats"][:, ii, jj]
    T = tris / np.linalg.norm(tris, axis=1, keepdims=True)
    Ms = {"pearson": np.corrcoef(tris), "cosine": T @ T.T}
    out = {}
    for kind, M in Ms.items():
        for which in ("ml", "all"):
            sel = c[which]
            sub = M[np.ix_(sel, sel)]
            out[(kind, which)] = float((sub.sum() - len(sel)) / (len(sel) ** 2 - len(sel)))
    return out


def contrasts(vals, which):
    out = {"tuning_pooled": float(np.mean([vals[(b, which)] - vals[(a, which)] for a, b in TUNING]))}
    for a, b in TUNING:
        out[f"tuning_{a.split('-')[2]}"] = vals[(b, which)] - vals[(a, which)]
    if which == "ml":
        for ch, (a, b) in CHAINS.items():
            out[f"scale_2b27b_{ch}"] = vals[(b, "ml")] - vals[(a, "ml")]
    return out


def run(data, label):
    n = 171
    pi, pj = np.triu_indices(n, 1)
    s0 = {c: stats(data[c], pi, pj) for c in CELLS}
    obs = {k: {(c, w): s0[c][(k, w)] for c in CELLS for w in ("ml", "all")} for k in KINDS}
    obs_con = {k: {**{"midlate_" + n_: v for n_, v in contrasts(obs[k], "ml").items()},
                   **{"alllayer_" + n_: v for n_, v in contrasts(obs[k], "all").items()}} for k in KINDS}
    graded = GRADED
    boots = {k: np.full((B, len(graded)), np.nan) for k in KINDS}
    rng = np.random.default_rng(SEED)
    for b in range(B):
        idx = rng.integers(0, n, n)
        ii, jj = idx[pi], idx[pj]
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
        sb = {c: stats(data[c], ii, jj) for c in CELLS}
        for k in KINDS:
            v = {(c, w): sb[c][(k, w)] for c in CELLS for w in ("ml", "all")}
            con = {**{"midlate_" + n_: x for n_, x in contrasts(v, "ml").items()},
                   **{"alllayer_" + n_: x for n_, x in contrasts(v, "all").items()}}
            boots[k][b] = [con[g] for g in graded]
        if (b + 1) % 500 == 0:
            print(f"{label}: {b + 1}/{B}", flush=True)
    res = {"per_cell": {}, "contrasts": {}}
    for c in CELLS:
        res["per_cell"][c] = {}
        for w, name in (("ml", "mid_late"), ("all", "all_layer")):
            p_, c_ = obs["pearson"][(c, w)], obs["cosine"][(c, w)]
            res["per_cell"][c][name] = {"pearson": p_, "cosine": c_, "abs_diff": abs(c_ - p_)}
    for name in obs_con["pearson"]:
        e = {}
        for k in KINDS:
            d_ = obs_con[k][name]
            e[k] = {"delta": d_, "floor_status": "at_or_above_floor" if abs(d_) >= FLOOR else "below_floor"}
            if name in graded:
                lo, hi = np.percentile(boots[k][:, graded.index(name)], [2.5, 97.5])
                real = (lo > 0 or hi < 0) and abs(d_) >= FLOOR
                e[k]["ci95"] = [float(lo), float(hi)]
                e[k]["verdict"] = "REAL" if real else ("NULL_subfloor" if abs(d_) < FLOOR else "NULL_CI_includes_0")
        e["abs_diff_delta"] = abs(e["cosine"]["delta"] - e["pearson"]["delta"])
        if name in graded:
            e["abs_diff_ci95"] = [abs(x - y) for x, y in zip(e["cosine"]["ci95"], e["pearson"]["ci95"])]
            e["verdict_changes"] = e["cosine"]["verdict"] != e["pearson"]["verdict"]
        e["floor_status_changes"] = e["cosine"]["floor_status"] != e["pearson"]["floor_status"]
        res["contrasts"][name] = e
    return res


def check(res, ref, label):
    """Pearson side against the stored producer output, at its five decimals."""
    bad = []
    for c, v in ref["per_cell_mid_late_obs"].items():
        if round(res["per_cell"][c]["mid_late"]["pearson"], 5) != v:
            bad.append(f"{label} {c} mid_late")
    for name, v in ref["contrasts"].items():
        e = res["contrasts"]["midlate_" + name]["pearson"]
        if round(e["delta"], 5) != v["delta_obs"] or [round(x, 5) for x in e["ci95"]] != v["ci95"] or e["verdict"] != v["verdict"]:
            bad.append(f"{label} {name}: {round(e['delta'], 5)} {[round(x, 5) for x in e['ci95']]} {e['verdict']} vs stored {v}")
    return bad


GRADED = ["midlate_tuning_pooled", "alllayer_tuning_pooled", "midlate_scale_2b27b_base", "midlate_scale_2b27b_it"]


def by_group(full, excl):
    """Maximum absolute difference per group of reported values, with where it falls."""
    groups = {}
    def put(g, v, where):
        if g not in groups or v > groups[g]["max_abs_diff"]:
            groups[g] = {"max_abs_diff": v, "at": where}
    for part, r in (("all_layers", full), ("layer45_excluded", excl)):
        for c, x in r["per_cell"].items():
            for w, v in x.items():
                put(f"per_cell_{w}", v["abs_diff"], f"{part}/{c}")
        for n_, e in r["contrasts"].items():
            g = "registered_contrast_points" if n_ in GRADED else "per_size_tuning_points"
            put(g, e["abs_diff_delta"], f"{part}/{n_}")
            for v in e.get("abs_diff_ci95", []):
                put("interval_bounds", v, f"{part}/{n_}")
    for g in groups.values():
        g["within_1e-5"] = g["max_abs_diff"] <= 1e-5
    return groups


def maxdiff(res):
    pts = [v["abs_diff"] for x in res["per_cell"].values() for v in x.values()]
    pts += [e["abs_diff_delta"] for e in res["contrasts"].values()]
    allv = pts + [d for e in res["contrasts"].values() for d in e.get("abs_diff_ci95", [])]
    return max(pts), max(allv)


def main():
    V = {c: nf.load_vectors(REPO / VEC[c])[1] for c in CELLS}
    full = run({c: cell_mats(V[c]) for c in CELLS}, "all layers")
    excl = run({c: cell_mats(V[c], excl=c.startswith("gemma-2-27b")) for c in CELLS}, "layer 45 excluded")
    bad, dev = [], 0.0
    for c in CELLS:
        sa = json.loads((GEO / c / "stageA_summary.json").read_text())["deterministic"]
        for key, name in (("layerstab_mid_late", "mid_late"), ("layerstab_all", "all_layer")):
            dev = max(dev, abs(full["per_cell"][c][name]["pearson"] - sa[key]))
            if abs(full["per_cell"][c][name]["pearson"] - sa[key]) > GATE_TOL:
                bad.append(f"{c} {name}: {full['per_cell'][c][name]['pearson']} vs stageA {sa[key]}")
    bad += check(full, json.loads(REF_CI.read_text()), "all layers")
    bad += check(excl, json.loads(REF_EXCL.read_text()), "layer 45 excluded")
    if bad:
        print("\n".join(bad))
        sys.exit("STOP: the Pearson side does not reproduce the stored results; nothing written")
    excl["per_cell"] = {c: {"mid_late": excl["per_cell"][c]["mid_late"]} for c in CELLS}
    excl["contrasts"] = {k: v for k, v in excl["contrasts"].items() if k.startswith("midlate_")}
    m_pts = max(maxdiff(full)[0], maxdiff(excl)[0])
    m_all = max(maxdiff(full)[1], maxdiff(excl)[1])
    out = {"analysis": "Layer stability under uncentered cosine in place of Pearson, nothing else changed",
           "measure": "mean off-diagonal similarity between the per-layer upper-triangle rows of the 171 x 171 "
                      "cosine matrices, over the 14 sampled layers (all-layer) or those at or past half depth (mid-late)",
           "bootstrap": {"B": B, "seed": SEED, "design": "paired case bootstrap over emotions, self-duplicate pairs "
                         "excluded, Pearson and cosine on the same draws, as phase4_layerstab_ci.py"},
           "floor": FLOOR, "verdict_rule": "REAL iff the 95 percent interval excludes 0 and |delta| >= floor",
           "pearson_gate": f"per-cell values equal stageA_summary.json to {GATE_TOL}; contrasts, intervals and verdicts "
                           "equal layerstab_ci.json and layerstab_ci_exclL45.json at five decimals",
           "pearson_max_abs_dev_from_stageA": dev,
           "all_layers": full, "layer45_excluded": excl,
           "max_abs_diff_point_values": m_pts, "max_abs_diff_all_reported": m_all,
           "claim_threshold": 1e-5, "claim_holds": m_all <= 1e-5,
           "max_abs_diff_by_group": by_group(full, excl),
           "any_verdict_changes": any(e.get("verdict_changes") for r in (full, excl) for e in r["contrasts"].values()),
           "any_floor_status_changes": any(e["floor_status_changes"] for r in (full, excl) for e in r["contrasts"].values())}
    OUT.write_text(json.dumps(out, indent=1))
    man = {"output": str(OUT.relative_to(REPO)), "output_sha256": sha(OUT),
           "script": str(Path(__file__).resolve().relative_to(REPO)), "script_sha256": sha(Path(__file__).resolve()),
           "noise_floor_sha256": sha(Path(nf.__file__)),
           "inputs": {VEC[c]: sha(REPO / VEC[c]) for c in CELLS},
           "references": {str(p.relative_to(REPO)): sha(p) for p in
                          [REF_CI, REF_EXCL] + [GEO / c / "stageA_summary.json" for c in CELLS]},
           "python": sys.version.split()[0], "numpy": np.__version__,
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    OUT.with_name(OUT.stem + "_manifest.json").write_text(json.dumps(man, indent=1))
    print("wrote", OUT, man["output_sha256"])
    print(f"max abs difference: point values {m_pts:.3e}, all reported {m_all:.3e}; claim holds: {out['claim_holds']}; "
          f"verdict changes: {out['any_verdict_changes']}; floor-status changes: {out['any_floor_status_changes']}")


if __name__ == "__main__":
    main()
