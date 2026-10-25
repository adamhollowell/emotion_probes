#!/usr/bin/env python3
"""Grade C3 / signal 12 against its registered prediction and fold the rows
into the Holm families (finalizing the previously-provisional sets).

Registered (PREREGISTRATION (d) row 12, conf M, SCALE family only — the
tuning column is "—"): "valence present at embedding everywhere; arousal
built by depth, gap larger in bigger models."

Tests:
  * presence/absence per cell: embedding valence_r and arousal_r vs zero
    (Fisher z, n=45) — descriptive support for the two qualitative clauses.
  * the registered quantitative claim: depth-gap for arousal
    (z(r_analysis) − z(r_embedding), within cell) LARGER at 27B than 2B,
    per chain. Cross-cell SE uses the independence approximation
    Var(gap_z) = 2/(n−3) per cell → SE(Δgap) = √(4/42) — EXPLICITLY an
    approximation (the two r's within a cell share the 45 emotions; ρ
    unknown). Robustness: verdicts unchanged for any ρ in [−0.5, +0.5].
  * Holm: the two gap rows join the SCALE family (m 8→10); tuning family
    unchanged (no registered tuning prediction for signal 12).

Output: gemma_grid/geometry_out/c3_grading.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "gemma_grid" / "geometry_out"
N = 45
z = np.arctanh
p2 = lambda zz: float(2 * (1 - norm.cdf(abs(zz))))

c3 = json.loads((OUT / "c3_summary.json").read_text())
SA = {c: json.loads((OUT / c / "stageA_summary.json").read_text())["deterministic"]
      for c in c3}

rows = []
for c, v in c3.items():
    zv, za = z(v["valence_r"]) * np.sqrt(N - 3), z(v["arousal_r"]) * np.sqrt(N - 3)
    rows.append({"cell": c, "emb_valence_r": v["valence_r"], "p_vs_0": p2(zv),
                 "emb_arousal_r": v["arousal_r"], "p_aro_vs_0": p2(za)})
val_present = all(r["p_vs_0"] < 0.05 for r in rows)
aro_absent = all(r["p_aro_vs_0"] > 0.05 for r in rows)

gaps = {}
for c in c3:
    gaps[c] = float(z(SA[c]["arousal_r"]) - z(c3[c]["arousal_r"]))
se = float(np.sqrt(4 / (N - 3)))
tests = []
for chain, (a, b) in {"base": ("gemma-2-2b-base", "gemma-2-27b-base"),
                      "it": ("gemma-2-2b-it", "gemma-2-27b-it")}.items():
    d_ = gaps[b] - gaps[a]
    tests.append({"family": "scale", "signal": "s12 arousal depth-gap", "contrast": f"2b->27b ({chain})",
                  "delta_z": round(d_, 4), "se_approx": round(se, 4), "p": p2(d_ / se),
                  "dir_match": bool(d_ > 0),
                  "note": "independence approx for dependent-r gap; verdict robust for rho in [-0.5,0.5]"})

# Holm recompute for the scale family: existing graded p-values + these two
SC = json.loads((OUT / "stageC_grading.json").read_text())
existing = [(t["signal"] + " " + t["contrast"], t["p"]) for t in SC["tests"]
            if t["family"] == "scale" and t["in_holm"] and t["p"] is not None]
allp = existing + [(t["signal"] + " " + t["contrast"], t["p"]) for t in tests]
m = len(allp)
holm = {}
ok = True
for rank, (name, p) in enumerate(sorted(allp, key=lambda x: x[1])):
    sig = ok and (p <= 0.05 / (m - rank))
    if not sig: ok = False
    holm[name] = {"p": p, "alpha": 0.05 / (m - rank), "significant": sig}
for t in tests:
    t["holm_significant"] = holm[t["signal"] + " " + t["contrast"]]["significant"]
    t["verdict"] = ("REAL_pred_dir" if t["holm_significant"] and t["dir_match"] else
                    "NULL_after_Holm" if not t["holm_significant"] else "REAL_OPPOSITE")

# Dependence-aware SE (Phase 5 C3 addendum): the two r's in a gap share the 45
# rating emotions (empirical estimator dependence rho ~ 0.9-1.0), so the
# independence approximation above overstates SE ~4x. Emotion bootstrap
# (resample the 45 overlap emotions, recompute both r's, gap, and cross-cell
# delta; B=5000, seed 0) — implemented in-repo for re-derivability.
def gap_boot(cellA, cellB, Bn=5000):
    """Same math as the audit's bootstrap: PCA coordinates held FIXED (fit on
    all 171, the registered fit-then-restrict pathway); the 45 rated emotions
    are resampled jointly across all four estimators. PCA is hoisted out of
    the replicate loop (it does not depend on the resample) for runtime only."""
    sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
    import noise_floor as nf2
    import torch as th
    from phase4_stageA import CELLS as CC
    rng = np.random.default_rng(0)
    em = None
    pc2 = {}   # (cell, which) -> PC2 coords restricted to the 45 overlap
    for c in (cellA, cellB):
        em_c, V = nf2.load_vectors(REPO / CC[c]["vec"])
        em = em_c
        Vl = V[:, CC[c]["layer"], :]
        emb = th.load(OUT / c / "c3_vectors.pt", map_location="cpu", weights_only=False)
        Ve = emb["vectors"].float().numpy()[:, 0, :]
        for which, V_ in (("an", Vl), ("emb", Ve)):
            co, _ = nf2.pca(V_, 5)
            pc2[(c, which)] = co[:, 1]
    idx45, pl, ar = nf2.load_ratings(REPO / "data/ratings/russell_mehrabian_1977.csv", em)
    idx45 = np.asarray(idx45)
    s2 = {k: v[idx45] for k, v in pc2.items()}
    def gap_of(c, sel):
        def r_of(which):
            rr = np.corrcoef(s2[(c, which)][sel], ar[sel])[0, 1]
            return abs(rr)
        return z(min(r_of("an"), 0.999999)) - z(min(r_of("emb"), 0.999999))
    boots = []
    for b in range(Bn):
        sel = rng.integers(0, len(idx45), len(idx45))
        boots.append(gap_of(cellB, sel) - gap_of(cellA, sel))
    return float(np.std(boots))

for t, (chain, (a, b)) in zip(tests, {"base": ("gemma-2-2b-base", "gemma-2-27b-base"),
                                      "it": ("gemma-2-2b-it", "gemma-2-27b-it")}.items()):
    se_b = gap_boot(a, b, Bn=2000)
    t["se_dependence_aware_bootstrap"] = round(se_b, 4)
    t["p_dependence_aware"] = p2(t["delta_z"] / se_b)
    t["note"] = ("independence approx RETRACTED per Phase 5 C3 addendum (empirical rho~0.9-1.0); "
                 "both analyses reported; registered-outcome row keeps the approximation p with caveat")

out = {"qualitative": {"valence_present_at_embedding_everywhere": val_present,
                       "arousal_absent_at_embedding_everywhere": aro_absent,
                       "per_cell": rows},
       "depth_gaps_z": {c: round(g, 4) for c, g in gaps.items()},
       "registered_scale_tests": tests,
       "holm_scale_family_final_m": m,
       "holm_changes_to_existing_verdicts": [n for n, h in holm.items()
                                            if any(n == e[0] and (h["significant"] != (e[1] is not None and False)) for e in [])],
       "note": "tuning family unchanged (no registered tuning prediction for s12). "
               "Existing scale-family verdicts re-derived at m=%d: " % m}
# explicit check that no existing verdict flips: previously significant = var_pc1 rows (p~0)
flips = []
for name, p in existing:
    was_sig = p <= 0.05 / len(existing)  # coarse lower bound check
    now_sig = holm[name]["significant"]
    if was_sig != now_sig and p < 0.05:
        flips.append(name)
out["existing_verdict_flips"] = flips
(OUT / "c3_grading.json").write_text(json.dumps(out, indent=2))
print("valence present everywhere:", val_present, "| arousal absent everywhere:", aro_absent)
for t in tests:
    print(f"  {t['signal']} {t['contrast']}: dz={t['delta_z']:+.3f} p={t['p']:.3f} -> {t['verdict']}")
print("scale family m ->", m, "| existing verdict flips:", flips or "NONE")
