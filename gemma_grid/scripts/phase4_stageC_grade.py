#!/usr/bin/env python3
"""Phase 4 / Stage C (part 2) — tier-2 grading of every registered contrast.

Implements the LOCKED tier-2 package (PREREGISTRATION.md (e)):
  * Fisher r-to-z two-sample tests for correlations (n=45 R&M overlap; n=171
    norm-r), pairwise + POOLED tuning main effect (mean [IT-base] Δz over the
    three size pairs) + the registered interaction DiD (arousal).
  * Story-bootstrap SEs (phase4_stageC_bootstrap outputs, B=1000) for PCA
    variance % and neighbor cosines; cross-cell gap tested vs pooled SD.
  * Locked effect-size floors: variance 1.0pp; r(n=45) 0.15; r(n=171) 0.10;
    neighbor cos 0.03; layer stability 0.01; ARI real-reorg threshold 0.51;
    loving position graded POST-PROCRUSTES only (via C4 iv).
  * C4 i/ii/iv floors, measured per the pre-registered rule from the
    bootstrap: WITHIN-CELL noise ceiling = statistic computed between paired
    independent replicates of the SAME cell (b vs b+B/2); cross-cell value
    read against it. Neighbor-cosine cross-cell Δ uses the ANCHOR's official
    top-1 pairs indexed into every cell's bootstrap cosine triangle (same
    fixed pair in both cells — never each cell's own possibly-different pair).
  * Holm–Bonferroni WITHIN each contrast family (tuning / scale /
    interaction). Exploratory rows (signal 6 norm-r, the two interaction
    sub-claims) are graded but OUTSIDE the Holm sets, as registered.

Verdicts per test: REAL (+direction vs registered prediction) | NULL_subfloor
| NULL_after_Holm | NO_FLOOR_descriptive.  C3/signal-12 rows are PENDING (not
yet computed); Holm sets are marked provisional accordingly.

Output: gemma_grid/geometry_out/stageC_grading.json and stageC_grading.md
Usage: python gemma_grid/scripts/phase4_stageC_grade.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
from phase4_stageA import CELLS

OUT = REPO / "gemma_grid" / "geometry_out"
B = 1000
N45, N171 = 45, 171
SE_PAIR_45 = np.sqrt(2 / (N45 - 3))          # 0.218
SE_POOL_45 = np.sqrt(2 / (3 * (N45 - 3)))    # pooled mean of 3 independent Δz
SE_DID_45 = np.sqrt(4 / (N45 - 3))           # DiD of two Δz
SE_PAIR_171 = np.sqrt(2 / (N171 - 3))
FLOORS = {"var_pp": 1.0, "r45": 0.15, "r171": 0.10, "cos": 0.03, "stab": 0.01, "ari": 0.51}
TUNING = [("gemma-2-2b-base", "gemma-2-2b-it"), ("gemma-2-9b-base", "gemma-2-9b-it"),
          ("gemma-2-27b-base", "gemma-2-27b-it")]
CHAINS = {"base": ("gemma-2-2b-base", "gemma-2-27b-base"), "it": ("gemma-2-2b-it", "gemma-2-27b-it")}

z = np.arctanh
p2 = lambda zz: float(2 * (1 - norm.cdf(abs(zz))))


def load(cell):
    S = json.loads((OUT / cell / "stageA_summary.json").read_text())
    with np.load(OUT / cell / f"bootstrap_B{B}.npz", allow_pickle=True) as bz:
        # materialize once — indexing an NpzFile member re-decompresses per access
        b = {k: bz[k] for k in ("tri", "coords", "ratios", "pair_cos", "rs", "pairs", "emotions")}
    return S, b


def procrustes_disp(Xa, Xb):
    A = Xa - Xa.mean(0); A = A / np.linalg.norm(A)
    Bm = Xb - Xb.mean(0); Bm = Bm / np.linalg.norm(Bm)
    U, _, Vt = np.linalg.svd(A.T @ Bm)
    R = U @ Vt
    resid = A @ R - Bm
    return float((resid ** 2).sum()), np.linalg.norm(resid, axis=1)


def main():
    cells = {c: load(c) for c in CELLS}
    em = list(cells["gemma-2-27b-base"][1]["emotions"])
    li = em.index("loving")
    anchor_S = cells["gemma-2-27b-base"][0]
    anchor_pairs = [(p_, nb[0][0]) for p_, nb in anchor_S["neighbours_top5"].items()]
    iu = np.triu_indices(len(em), 1)
    tri_index = {}
    for (x, y) in anchor_pairs:
        i, j = sorted((em.index(x), em.index(y)))
        tri_index[f"{x}->{y}"] = int(np.where((iu[0] == i) & (iu[1] == j))[0][0])

    det = {c: cells[c][0]["deterministic"] for c in CELLS}
    boot = {c: cells[c][1] for c in CELLS}
    tests = []   # rows: family, signal, contrast, value, err, floor, p, verdict pieces

    def add(family, signal, contrast, delta, se, floor_key, pred_dir, in_holm=True, note=""):
        pv = p2(delta / se) if se and se > 0 else None
        floor = FLOORS.get(floor_key)
        tests.append(dict(family=family, signal=signal, contrast=contrast,
                          delta=float(delta), se=(float(se) if se else None), p=pv,
                          floor=floor, above_floor=(None if floor is None else bool(abs(delta) >= floor)),
                          pred_dir=pred_dir, dir_match=(None if pred_dir == 0 else bool(np.sign(delta) == pred_dir)),
                          in_holm=in_holm, note=note))

    # ---------------- TUNING family ----------------
    for key, fl, pred in [("valence_r", "r45", +1), ("arousal_r", "r45", +1)]:
        dzs = [z(det[b_][key]) - z(det[a_][key]) for a_, b_ in TUNING]
        drs = [det[b_][key] - det[a_][key] for a_, b_ in TUNING]
        for (a_, b_), dz_, dr_ in zip(TUNING, dzs, drs):
            add("tuning", key, f"{a_.split('-')[2]}", dr_, None, fl, pred, in_holm=False,
                note=f"pairwise Fisher z={dz_/SE_PAIR_45:.2f} p={p2(dz_/SE_PAIR_45):.4f} (family test is pooled)")
            tests[-1]["p"] = p2(dz_ / SE_PAIR_45)
        add("tuning", key, "POOLED[IT-base]", float(np.mean(drs)), None, fl, pred,
            note=f"pooled Fisher: mean dz={np.mean(dzs):.4f} z={np.mean(dzs)/SE_POOL_45:.2f}")
        tests[-1]["p"] = p2(np.mean(dzs) / SE_POOL_45)
    for key, pred in [("offdiag_pc1_arousal", 0), ("offdiag_pc2_pleasure", 0)]:
        drs = [det[b_][key] - det[a_][key] for a_, b_ in TUNING]
        dzs = [z(det[b_][key]) - z(det[a_][key]) for a_, b_ in TUNING]
        add("tuning", key, "POOLED[IT-base]", float(np.mean(drs)), None, "r45", -int(np.sign(np.mean([det[a_][key] for a_, _ in TUNING]))),
            note="prediction: shrink toward 0; pooled Fisher")
        tests[-1]["p"] = p2(np.mean(dzs) / SE_POOL_45)
    for pc, pred in [("var_pc1", +1), ("var_pc2", +1), ("var_pc3", 0)]:
        col = {"var_pc1": 0, "var_pc2": 1, "var_pc3": 2}[pc]
        deltas, ses = [], []
        for a_, b_ in TUNING:
            d_ = det[b_][pc] - det[a_][pc]
            s_ = 100 * np.sqrt(boot[a_]["ratios"][:, col].std() ** 2 + boot[b_]["ratios"][:, col].std() ** 2)
            deltas.append(d_); ses.append(s_)
        add("tuning", pc, "POOLED[IT-base]", float(np.mean(deltas)),
            float(np.sqrt(np.sum(np.square(ses))) / 3), "var_pp", pred, note="bootstrap SE")
    # neighbor cosines: anchor official pairs, from tri
    dcs = []
    for a_, b_ in TUNING:
        for name, ti in tri_index.items():
            dcs.append(boot[b_]["tri"][:, ti].mean() - boot[a_]["tri"][:, ti].mean())
    add("tuning", "neighbor_cos(9 anchor pairs)", "POOLED[IT-base]", float(np.mean(dcs)),
        float(np.std(dcs) / np.sqrt(len(dcs))), "cos", 0, note="mean over pairs&sizes; floor applies per Δ; max|Δ|=%.4f" % np.max(np.abs(dcs)))
    # self-ARI (seed spread)
    dsa = [cells[b_][0]["kmeans_seed_sweep"]["self_ari_mean"] - cells[a_][0]["kmeans_seed_sweep"]["self_ari_mean"] for a_, b_ in TUNING]
    ses = [np.sqrt(cells[a_][0]["kmeans_seed_sweep"]["self_ari_std"] ** 2 + cells[b_][0]["kmeans_seed_sweep"]["self_ari_std"] ** 2) for a_, b_ in TUNING]
    add("tuning", "self_ARI(k=10)", "POOLED[IT-base]", float(np.mean(dsa)), float(np.sqrt(np.sum(np.square(ses))) / 3),
        None, +1, note="seed-spread SE; no locked floor beyond band logic")
    # cohesion (no locked floor)
    dco = [cells[b_][0]["cohesion"]["mean_purity_top10"] - cells[a_][0]["cohesion"]["mean_purity_top10"] for a_, b_ in TUNING]
    add("tuning", "cohesion_purity_top10", "POOLED[IT-base]", float(np.mean(dco)), None, None, +1,
        in_holm=False, note="NO pre-registered floor -> descriptive only")
    # layer stability
    for key in ("layerstab_all", "layerstab_mid_late"):
        d_ = float(np.mean([det[b_][key] - det[a_][key] for a_, b_ in TUNING]))
        add("tuning", key, "POOLED[IT-base]", d_, None, "stab", 0, note="floor-only grading (locked)")

    # ---------------- SCALE family (chain endpoints within tuning) ----------------
    for chain, (a_, b_) in CHAINS.items():
        for key, fl, pred in [("valence_r", "r45", +1), ("arousal_r", "r45", +1)]:
            dz_ = z(det[b_][key]) - z(det[a_][key])
            add("scale", key, f"2b->27b ({chain})", det[b_][key] - det[a_][key], None, fl, pred,
                note=f"Fisher z={dz_/SE_PAIR_45:.2f}")
            tests[-1]["p"] = p2(dz_ / SE_PAIR_45)
        for pc, pred in [("var_pc1", +1), ("var_pc2", +1)]:
            col = {"var_pc1": 0, "var_pc2": 1}[pc]
            s_ = 100 * np.sqrt(boot[a_]["ratios"][:, col].std() ** 2 + boot[b_]["ratios"][:, col].std() ** 2)
            add("scale", pc, f"2b->27b ({chain})", det[b_][pc] - det[a_][pc], float(s_), "var_pp", pred, note="bootstrap SE")
        for key in ("layerstab_mid_late",):
            add("scale", key, f"2b->27b ({chain})", det[b_][key] - det[a_][key], None, "stab", 0, note="floor-only")

    # ---------------- INTERACTION family ----------------
    dz27 = z(det["gemma-2-27b-it"]["arousal_r"]) - z(det["gemma-2-27b-base"]["arousal_r"])
    dz2 = z(det["gemma-2-2b-it"]["arousal_r"]) - z(det["gemma-2-2b-base"]["arousal_r"])
    did = dz27 - dz2
    add("interaction", "arousal_r DiD (27B-2B)", "([IT-base]@27B)-([IT-base]@2B)",
        float(np.tanh(dz27) - np.tanh(dz2)), None, "r45", +1,
        note=f"registered primary interaction; Fisher DiD z={did/SE_DID_45:.2f}")
    tests[-1]["p"] = p2(did / SE_DID_45)

    # ---------------- EXPLORATORY (outside Holm) ----------------
    for key in ("norm_valence_r", "norm_arousal_abs_r"):
        dzs = [np.arctanh(det[b_][key]) - np.arctanh(det[a_][key]) for a_, b_ in TUNING]
        add("exploratory", key, "POOLED[IT-base]", float(np.mean([det[b_][key] - det[a_][key] for a_, b_ in TUNING])),
            None, "r171", 0, in_holm=False, note=f"n=171 Fisher pooled z={np.mean(dzs)/np.sqrt(2/(3*(N171-3))):.2f}")
        tests[-1]["p"] = p2(np.mean(dzs) / np.sqrt(2 / (3 * (N171 - 3))))

    # ---------------- C4 i/ii/iv with bootstrap floors ----------------
    stageB = json.loads((OUT / "stageB_contrasts.json").read_text())
    obs = {r["pair"]: r for fam in ("family_tuning", "family_scale") for r in stageB[fam]}
    h = B // 2
    c4 = {}
    within = {}
    for c in CELLS:
        t = boot[c]["tri"]
        a1, a2 = t[:h], t[h:]
        rsa_w = np.array([float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(a1, a2)])
        disp_w = np.array([procrustes_disp(boot[c]["coords"][b_, :, :2], boot[c]["coords"][b_ + h, :, :2])[0] for b_ in range(h)])
        within[c] = {"rsa_mean": float(rsa_w.mean()), "rsa_p05": float(np.quantile(rsa_w, 0.05)),
                     "disp_mean": float(disp_w.mean()), "disp_p95": float(np.quantile(disp_w, 0.95))}
    for pair, r in obs.items():
        a_, b_ = pair.split("<->")
        ta, tb = boot[a_]["tri"], boot[b_]["tri"]
        rsa_x = np.array([float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(ta, tb)])
        disp_stats = [procrustes_disp(boot[a_]["coords"][b__, :, :2], boot[b_]["coords"][b__, :, :2]) for b__ in range(0, B, 2)]
        dsp = np.array([s[0] for s in disp_stats])
        lov = np.array([s[1][li] for s in disp_stats])
        med = np.array([np.median(s[1]) for s in disp_stats])
        ceiling = min(within[a_]["rsa_p05"], within[b_]["rsa_p05"])
        floor_d = max(within[a_]["disp_p95"], within[b_]["disp_p95"])
        c4[pair] = {
            "rsa_obs": r["rsa"]["cosine"], "rsa_boot_mean": float(rsa_x.mean()), "rsa_boot_sd": float(rsa_x.std()),
            "rsa_within_ceiling_p05": ceiling,
            "rsa_verdict": "REAL_geometry_difference" if rsa_x.mean() < ceiling else "NULL_within_sampling_noise",
            "procrustes_obs": r["procrustes_k2"]["disparity"], "procrustes_boot_mean": float(dsp.mean()),
            "procrustes_within_floor_p95": floor_d,
            "procrustes_verdict": "REAL_plane_difference" if np.quantile(dsp, 0.05) > floor_d else "NULL_within_sampling_noise",
            "ari_obs": r["cross_ari"]["ari"],
            "ari_verdict": "REAL_reorganization" if r["cross_ari"]["ari"] < FLOORS["ari"] else "NULL(>=0.51)",
            "loving_disp_gt_median_frac": float(np.mean(lov > med)),
            "loving_verdict_vs_H_moves_most": ("DISCONFIRMED (moves LESS than median)" if np.mean(lov > med) < 0.05
                                               else "CONFIRMED" if np.mean(lov > med) > 0.95 else "NULL"),
        }

    # ---------------- Holm within families ----------------
    for fam in ("tuning", "scale", "interaction"):
        rows = [t for t in tests if t["family"] == fam and t["in_holm"] and t["p"] is not None]
        m = len(rows)
        for rank, t in enumerate(sorted(rows, key=lambda r_: r_["p"])):
            alpha = 0.05 / (m - rank)
            t["holm_alpha"] = alpha
            t["holm_significant"] = bool(t["p"] <= alpha)
            if rank > 0 and not sorted(rows, key=lambda r_: r_["p"])[rank - 1].get("holm_significant", True):
                t["holm_significant"] = False
        for t in rows:
            if t.get("above_floor") is False:
                t["verdict"] = "NULL_subfloor"
            elif not t.get("holm_significant", False):
                t["verdict"] = "NULL_after_Holm"
            else:
                t["verdict"] = ("REAL_" + ("pred_dir" if t.get("dir_match") else "OPPOSITE_dir")) if t.get("dir_match") is not None else "REAL"
    for t in tests:
        t.setdefault("verdict", "descriptive/pairwise (not a Holm unit)" if not t["in_holm"] else "n/a")

    out = {"note": "Holm sets PROVISIONAL: C3 signal-12 rows pending. Session grades statistics only; interpretations await Phase 5 adversarial review.",
           "within_cell_noise": within, "tests": tests, "c4": c4}
    (OUT / "stageC_grading.json").write_text(json.dumps(out, indent=2))

    # readable table
    lines = ["# Stage C grading (tier-2, Holm within families) — PROVISIONAL (C3 pending)\n",
             "| family | signal | contrast | Δ | floor | p | verdict |", "|---|---|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['family']} | {t['signal']} | {t['contrast']} | {t['delta']:+.4f} | "
                     f"{t['floor'] if t['floor'] is not None else '—'} | "
                     f"{('%.4g' % t['p']) if t['p'] is not None else '—'} | {t['verdict']} |")
    lines.append("\n## C4 contrasts vs bootstrap floors\n")
    lines.append("| pair | RSA obs (ceiling) | verdict | Procrustes obs (floor) | verdict | ARI | verdict | loving>median frac | loving-H verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for pair, r in c4.items():
        lines.append(f"| {pair} | {r['rsa_obs']:.4f} ({r['rsa_within_ceiling_p05']:.4f}) | {r['rsa_verdict']} | "
                     f"{r['procrustes_obs']:.4f} ({r['procrustes_within_floor_p95']:.4f}) | {r['procrustes_verdict']} | "
                     f"{r['ari_obs']:.3f} | {r['ari_verdict']} | {r['loving_disp_gt_median_frac']:.3f} | {r['loving_verdict_vs_H_moves_most']} |")
    (OUT / "stageC_grading.md").write_text("\n".join(lines))
    print("wrote stageC_grading.json / stageC_grading.md")
    for t in tests:
        if t["in_holm"] and t["p"] is not None:
            print(f"  {t['family']:11s} {t['signal']:24s} {t['contrast']:18s} Δ={t['delta']:+.4f} p={t['p']:.4g} -> {t['verdict']}")


if __name__ == "__main__":
    main()
