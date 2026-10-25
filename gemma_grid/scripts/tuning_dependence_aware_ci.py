#!/usr/bin/env python3
"""Dependence-aware intervals for the registered tuning contrasts (unregistered; review response).

Handoff of record: gemma_grid/drafting/HANDOFF_CLAUDE_CODE_TWO_REVIEW_ANALYSES_100T_2026-10-14.md
(Analysis B).

Signal definitions are IMPORTED from scripts/noise_floor.py (locked GATE 1 code):
load_vectors (float32), pca, amean, POS/NEG/HI/LO, load_ratings, and its sign rules.
Cell paths come from scripts/run_stageA.py VEC; layers from phase4_stageA.CELLS.

Order of work:
  1. GATE (a): per cell, valence_r / arousal_r / offdiag_pc1_arousal / offdiag_pc2_pleasure
     on the full 45 must match geometry_out/<cell>/stageA_summary.json
     "deterministic" to 1e-6.
     GATE (b): pooled tuning deltas, per-size deltas, pooled mean dz and the
     interaction must match geometry_out/stageC_grading.json to 1e-6.
     Any failure -> exit, no output.
  2. PCA fit ONCE per cell on all 171 (registered POS/NEG, HI/LO signs). Not resampled.
  3. Bootstrap B=2000, seed 20261014 (numpy default_rng). Each draw samples 45 of the
     45 rated emotions with replacement; the SAME indices are used for all six cells.
     In each draw the per-cell rule from noise_floor.compute flips each axis to agree
     with its own rating on the resampled emotions, then the four correlations are taken.
  4. Derived per draw: per-size tuning deltas (IT - base, raw r), pooled delta (mean of
     the three), interaction (IT-base)@27B - (IT-base)@2B on arousal_r in raw r; and the
     Fisher-z forms of each (for the SE ratio against the registered Fisher SEs).

Per Adam's amendment B (2026-10-14) no interval is compared to the .15 floor or any
other floor. Producer reports numbers only; it does not grade.
"""
import argparse, hashlib, json, os, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
GG = REPO / "gemma_grid"
sys.path.insert(0, str(GG / "scripts"))
import noise_floor as nf          # noqa: E402  locked signal definitions
import phase4_stageA as SA0       # noqa: E402  CELLS (layers)
import run_stageA as SA   # noqa: E402  VEC (100T paths)

B = 2000
SEED = 20261014
# Adam's amendment B, 2026-10-14: no floor flags; intervals are not compared to any floor.
GATE_TOL = 1e-6
N45 = 45
ANCHOR_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
RATINGS = REPO / "data" / "ratings" / "russell_mehrabian_1977.csv"
STAGEC = GG / "geometry_out" / "stageC_grading.json"
SIGNALS = ["valence_r", "arousal_r", "offdiag_pc1_arousal", "offdiag_pc2_pleasure"]
TUNING = [("2b", "gemma-2-2b-base", "gemma-2-2b-it"), ("9b", "gemma-2-9b-base", "gemma-2-9b-it"),
          ("27b", "gemma-2-27b-base", "gemma-2-27b-it")]
CELLS = [c for _, b_, t_ in TUNING for c in (b_, t_)]
# Registered Fisher SEs, as defined in scripts/phase4_stageC_grade.py
SE_PAIR = np.sqrt(2 / (N45 - 3))
SE_POOL_STAGEC = np.sqrt(2 / (3 * (N45 - 3)))
SE_DID = np.sqrt(4 / (N45 - 3))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def four(c1, c2, pl, ar):
    """noise_floor.compute's per-cell flip rule + the four correlations, verbatim logic."""
    c1 = c1.copy(); c2 = c2.copy()
    if np.corrcoef(c1, pl)[0, 1] < 0: c1 = -c1
    if np.corrcoef(c2, ar)[0, 1] < 0: c2 = -c2
    return {"valence_r": float(np.corrcoef(c1, pl)[0, 1]),
            "arousal_r": float(np.corrcoef(c2, ar)[0, 1]),
            "offdiag_pc1_arousal": float(np.corrcoef(c1, ar)[0, 1]),
            "offdiag_pc2_pleasure": float(np.corrcoef(c2, pl)[0, 1])}


def fit_cell(path, layer):
    em, V = nf.load_vectors(path)
    Vl = V[:, layer, :]
    coords, _ = nf.pca(Vl, 5)
    if nf.amean(coords[:, 0], em, nf.POS) < nf.amean(coords[:, 0], em, nf.NEG): coords[:, 0] *= -1
    if nf.amean(coords[:, 1], em, nf.HI) < nf.amean(coords[:, 1], em, nf.LO): coords[:, 1] *= -1
    idx, pl, ar = nf.load_ratings(RATINGS, em)
    return em, coords[idx, 0].copy(), coords[idx, 1].copy(), pl, ar


def derive(stats):
    """stats[cell][signal] -> dict of derived quantities, raw r and Fisher z."""
    d = {}
    for sig in SIGNALS:
        drs, dzs = [], []
        for sz, b_, t_ in TUNING:
            dr = stats[t_][sig] - stats[b_][sig]
            dz = np.arctanh(stats[t_][sig]) - np.arctanh(stats[b_][sig])
            d[f"{sig}|{sz}"] = (dr, dz); drs.append(dr); dzs.append(dz)
        d[f"{sig}|pooled"] = (float(np.mean(drs)), float(np.mean(dzs)))
    dr27, dz27 = d["arousal_r|27b"]; dr2, dz2 = d["arousal_r|2b"]
    d["interaction_arousal_r_raw"] = (dr27 - dr2, dz27 - dz2)
    # registered stageC form of the interaction's reported delta: tanh(dz27) - tanh(dz2)
    d["interaction_arousal_r_stageC_form"] = (float(np.tanh(dz27) - np.tanh(dz2)), dz27 - dz2)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(GG / "geometry_out" / "tuning_dependence_aware_ci.json"))
    a = ap.parse_args()
    t0 = time.time()

    paths = {c: REPO / SA.VEC[c] for c in CELLS}
    assert sha256(paths["gemma-2-27b-base"]) == ANCHOR_SHA, "anchor hash-gate (6de2e293) failed"
    fits, em_ref, pl_ref, ar_ref = {}, None, None, None
    for c in CELLS:
        layer = SA0.CELLS[c]["layer"]
        em, c1, c2, pl, ar = fit_cell(paths[c], layer)
        if em_ref is None: em_ref, pl_ref, ar_ref = em, pl, ar
        assert em == em_ref and np.array_equal(pl, pl_ref) and np.array_equal(ar, ar_ref), c
        assert len(c1) == N45, len(c1)
        fits[c] = (c1, c2, layer)
    pl, ar = pl_ref, ar_ref

    # ---- GATE (a): stage A deterministic block ------------------------------
    full = {c: four(fits[c][0], fits[c][1], pl, ar) for c in CELLS}
    gate_a = {}
    for c in CELLS:
        det = json.loads((GG / "geometry_out" / c / "stageA_summary.json").read_text())["deterministic"]
        gate_a[c] = {s: {"reproduced": full[c][s], "stored": det[s], "abs_diff": abs(full[c][s] - det[s])} for s in SIGNALS}
    gate_a_max = max(v["abs_diff"] for c in gate_a for v in gate_a[c].values())

    # ---- GATE (b): stage C tuning + interaction ------------------------------
    pt = derive(full)
    tests = json.loads(STAGEC.read_text())["tests"]
    want = {}
    for t in tests:
        if t["family"] == "tuning" and t["signal"] in SIGNALS:
            k = f"{t['signal']}|{'pooled' if t['contrast'].startswith('POOLED') else t['contrast']}"
            want[k] = t["delta"]
        if t["family"] == "interaction":
            want["interaction_arousal_r_stageC_form"] = t["delta"]
    gate_b = {k: {"reproduced": pt[k][0], "stored": v, "abs_diff": abs(pt[k][0] - v)} for k, v in want.items()}
    # pooled mean dz echoed in stageC notes (4 d.p.) for valence/arousal
    gate_b_dz_note = {s: round(pt[f"{s}|pooled"][1], 4) for s in ("valence_r", "arousal_r")}
    gate_b_max = max(v["abs_diff"] for v in gate_b.values())
    table2 = {"valence_r": 0.0558, "arousal_r": -0.0829, "offdiag_pc1_arousal": 0.0409, "offdiag_pc2_pleasure": -0.0881}
    gate_table2 = {s: {"reproduced_4dp": round(pt[f"{s}|pooled"][0], 4), "draft_table2": v,
                       "match": round(pt[f"{s}|pooled"][0], 4) == v} for s, v in table2.items()}
    gate = {"a_stageA_deterministic": gate_a, "a_max_abs_diff": gate_a_max,
            "b_stageC": gate_b, "b_max_abs_diff": gate_b_max, "b_pooled_mean_dz_4dp": gate_b_dz_note,
            "b_draft_table2": gate_table2, "tolerance": GATE_TOL}
    gate["pass"] = bool(gate_a_max <= GATE_TOL and gate_b_max <= GATE_TOL
                        and all(v["match"] for v in gate_table2.values()))
    print("GATE a max", gate_a_max, "b max", gate_b_max, "table2", {k: v["match"] for k, v in gate_table2.items()},
          "dz", gate_b_dz_note, "PASS" if gate["pass"] else "FAIL", flush=True)
    if not gate["pass"]:
        print(json.dumps(gate, indent=1))
        sys.exit("GATE FAILED — stopping before any new computation (no output written)")

    # ---- bootstrap ------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, N45, size=(B, N45))          # shared across all six cells
    keys = list(pt.keys())
    R = np.empty((B, len(keys))); Z = np.empty((B, len(keys)))
    n_degenerate = 0
    for b in range(B):
        ix = draws[b]
        st = {c: four(fits[c][0][ix], fits[c][1][ix], pl[ix], ar[ix]) for c in CELLS}
        dv = derive(st)
        R[b] = [dv[k][0] for k in keys]; Z[b] = [dv[k][1] for k in keys]
        if not (np.all(np.isfinite(R[b])) and np.all(np.isfinite(Z[b]))): n_degenerate += 1
    assert n_degenerate == 0, f"{n_degenerate} non-finite draws"

    def reg_se(k):
        if k.startswith("interaction"): return SE_DID, SE_DID
        if k.endswith("|pooled"): return SE_POOL_STAGEC, SE_PAIR   # (stageC actual, handoff-stated)
        return SE_PAIR, SE_PAIR

    out_q = {}
    for j, k in enumerate(keys):
        r = R[:, j]; z = Z[:, j]
        lo95, hi95 = np.percentile(r, [2.5, 97.5]); lo90, hi90 = np.percentile(r, [5, 95])
        se_r = float(np.std(r, ddof=1)); se_z = float(np.std(z, ddof=1))
        s_stagec, s_handoff = reg_se(k)
        out_q[k] = {"point_r": pt[k][0], "boot_se_r": se_r,
                    "ci95_pct": [float(lo95), float(hi95)], "ci90_pct": [float(lo90), float(hi90)],
                    "point_z": pt[k][1], "boot_se_z": se_z,
                    "registered_se_z_stageC": float(s_stagec),
                    "se_ratio_boot_over_registered_stageC": se_z / s_stagec,
                    "registered_se_z_as_stated_in_handoff": float(s_handoff),
                    "se_ratio_boot_over_handoff_stated": se_z / s_handoff}

    out = {"analysis": "B: dependence-aware bootstrap intervals for the tuning contrasts (unregistered, review response)",
           "handoff": "gemma_grid/drafting/HANDOFF_CLAUDE_CODE_TWO_REVIEW_ANALYSES_100T_2026-10-14.md",
           "B": B, "seed": SEED, "rng": "numpy.random.default_rng(seed).integers(0,45,size=(B,45)); same rows for all six cells",
           "amendment_B": "2026-10-14: floor flags removed; intervals are not compared to any floor",
           "n_rated": N45,
           "layers": {c: fits[c][2] for c in CELLS},
           "pca": "fit once per cell on all 171 (noise_floor.pca, float32 as loaded by noise_floor.load_vectors); POS/NEG, HI/LO signs; not resampled",
           "flip_rule": "per cell, per draw: each axis flipped to correlate positively with its own rating on the resampled emotions (noise_floor.compute)",
           "scales": {"r": "raw correlation units: deltas, intervals, boot_se_r",
                      "z": "Fisher z (arctanh) form of the same contrast: per-size dz, pooled mean dz, DiD dz27-dz2; boot_se_z and SE ratios"},
           "interaction_note": ("interaction_arousal_r_raw = (IT-base)@27B - (IT-base)@2B in raw r, as the handoff defines it; "
                                "interaction_arousal_r_stageC_form = tanh(dz27) - tanh(dz2), the form stageC_grading.json stores as delta"),
           "se_note": ("stageC uses sqrt(2/42) per-size, sqrt(2/(3*42)) for POOLED, sqrt(4/42) for the DiD; "
                       "the handoff states sqrt(2/42) for pooled. Both ratios are given."),
           "gate": gate, "full_sample_per_cell": full, "quantities": out_q,
           "draw_indices_sha256": hashlib.sha256(draws.astype(np.int64).tobytes()).hexdigest()}
    op = Path(a.out); op.write_text(json.dumps(out, indent=1))

    import torch
    man = {"output": str(op.relative_to(REPO)), "output_sha256": sha256(op),
           "script": str(Path(__file__).resolve().relative_to(REPO)), "script_sha256": sha256(__file__),
           "command": " ".join([sys.executable] + sys.argv),
           "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "wall_s": round(time.time() - t0, 1),
           "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                             [*paths.values(), RATINGS, STAGEC,
                              *[GG / "geometry_out" / c / "stageA_summary.json" for c in CELLS],
                              GG / "scripts" / "noise_floor.py", GG / "scripts" / "phase4_stageA.py",
                              GG / "scripts" / "run_stageA.py", GG / "scripts" / "phase4_stageC_grade.py"]},
           "seeds": {"bootstrap_seed": SEED, "B": B},
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform(), "blas": "Apple Accelerate (numpy default on macOS arm64)"},
           "precision": "PCA in float32 (noise_floor.load_vectors casts .float()); correlations via np.corrcoef (float64)"}
    mp = op.with_name(op.stem + "_manifest.json"); mp.write_text(json.dumps(man, indent=1))
    print("wrote", op, man["output_sha256"]); print("wrote", mp)


if __name__ == "__main__":
    main()
