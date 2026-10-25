#!/usr/bin/env python3
"""W7 — GGE1-G4 IT-colon P-colon analysis (CPU, post-session). Numbers only.

Pairs 27b-it's PRIMARY bare-final residuals (colon_analysis_it.pt::arms.bare)
against E1's persisted base-side scenario_acts.pt, under E1's FROZEN fig2 convention
(coordination_100T/E1_baseside_protocol_for_GGE1.md §1): bare prompt, final token,
L31 resid_post f32, UNIT-NORMALIZED dot resid·(v/‖v‖) onto E1's 100T cleaned probes
(emotion_vectors_171.pt), one probe set applied to BOTH models — comparable by
construction. The producer left these projections off-pod (P7.1 lesson: save residuals).

Reports the P-colon grade-determining numbers (§40 registration): IT−base diff of the
moral/self-conscious probes {regretful, remorseful, guilty; secondarily self-critical,
reflective} at L31 and band means, with the across-scenario noise bound. This session
does NOT adjudicate the grade — Grid/coordination grades (IT_COLON_ANALYSIS_DESIGN §4).

RUN: ./.venv/bin/python gemma_grid/colon_it/grade_pcolon.py
OUT: gemma_grid/geometry_out/g4_pcolon_grade.{json,md}
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "data/vectors/scenario_acts.pt"               # 27b-base, bare-final, frozen convention
IT = REPO / "gemma_grid/colon_it/colon_analysis_it.pt"  # 27b-it, 3 arms
V100 = REPO / "data/vectors/emotion_vectors_171.pt"     # E1 100T cleaned probes (both sides)
IT_NEUTRAL_100 = REPO / "gemma_grid/data/gemma-2-27b-it/vectors/gemma-2-27b-it__neutral.pt"
IT_CANON = REPO / "gemma_grid/data/gemma-2-27b-it/vectors/canonical.pt"  # IT raw, for own-probe sensitivity
OUTDIR = REPO / "gemma_grid" / "geometry_out"
L31 = 31
PRIMARY = ["regretful", "remorseful", "guilty"]
SECONDARY = ["self-critical", "reflective"]
# Appendix T.1 27B down (pooled-neutral) top-10, from PHASE5_APPENDIX_T_AUDIT_REPORT.md
T1_DOWN = ["nostalgic", "regretful", "remorseful", "guilty", "sentimental",
           "reflective", "sympathetic", "self-critical", "sorry", "jealous"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def unitnorm_proj(resid, Vhat):
    """resid: (S,46,d); Vhat: (E,46,d) unit-normed per (E,layer). -> (S,E,46) proj resid·v̂."""
    return np.einsum("sld,eld->sel", resid, Vhat)


def thirds(n=46):
    a = np.array_split(np.arange(n), 3)
    return {"early": a[0], "mid": a[1], "late": a[2]}


def main():
    for p in (BASE, IT, V100):
        if not p.exists():
            sys.exit(f"missing {p}")
    b = torch.load(BASE, map_location="cpu", weights_only=False)
    it = torch.load(IT, map_location="cpu", weights_only=False)
    Vd = torch.load(V100, map_location="cpu", weights_only=False)
    assert list(b["prompts"]) == list(it["prompts"]), "prompt order mismatch base vs IT — STOP"
    emotions = list(Vd["emotions"]); eidx = {e: i for i, e in enumerate(emotions)}
    for pr in PRIMARY + SECONDARY:
        assert pr in eidx, f"probe {pr} missing"

    base_resid = b["acts"].float().numpy()                          # (12,46,4608)
    it_bare = it["arms"]["bare"]["residuals"].float().numpy()       # (12,46,4608) PRIMARY
    V = Vd["vectors"].float().numpy()                               # (171,46,4608)
    Vhat = V / np.clip(np.linalg.norm(V, axis=2, keepdims=True), 1e-12, None)

    proj_base = unitnorm_proj(base_resid, Vhat)                     # (12,171,46)
    proj_it = unitnorm_proj(it_bare, Vhat)
    diff = proj_it - proj_base                                      # IT - base ; (12,171,46)
    S = diff.shape[0]

    def probe_stats(e, L=L31):
        i = eidx[e]
        d = diff[:, i, L]                                           # per-scenario IT-base at L
        mean = float(d.mean()); sd = float(d.std(ddof=1)); sem = sd / np.sqrt(S)
        return {"base_mean": float(proj_base[:, i, L].mean()),
                "it_mean": float(proj_it[:, i, L].mean()),
                "diff_mean": mean, "diff_sd_across_scenarios": sd, "diff_sem": sem,
                "z_vs_sem": (mean / sem if sem > 0 else float("nan")),
                "sign": "IT_lower" if mean < 0 else ("IT_higher" if mean > 0 else "equal")}

    per_probe = {e: probe_stats(e) for e in PRIMARY + SECONDARY}

    # aggregate over the primary triad: mean per-scenario diff across the 3 probes, then across scenarios
    pi = [eidx[e] for e in PRIMARY]
    agg_scn = diff[:, pi, L31].mean(axis=1)                        # (12,) per-scenario triad-mean diff
    agg = {"probes": PRIMARY, "diff_mean": float(agg_scn.mean()),
           "diff_sem": float(agg_scn.std(ddof=1) / np.sqrt(S)),
           "z_vs_sem": float(agg_scn.mean() / (agg_scn.std(ddof=1) / np.sqrt(S))),
           "sign": "IT_lower" if agg_scn.mean() < 0 else "IT_higher",
           "prediction": "IT_lower (P-colon, conf M)",
           "points_toward": ("CONFIRMS direction" if agg_scn.mean() < 0 else "FALSIFIES direction")
           + " (formal grade = coordination's per design §4)"}

    # band means (early/mid/late thirds) of the triad-aggregate diff
    bands = thirds()
    band_diff = {name: float(diff[:, pi, :][:, :, ix].mean()) for name, ix in bands.items()}

    # rank overlap vs T.1: our top-suppressed (most negative IT-base) at L31, all emotions
    dL31 = diff[:, :, L31].mean(axis=0)                            # (171,) mean IT-base per emotion
    order = np.argsort(dL31)                                       # most negative first
    our_top10 = [emotions[i] for i in order[:10]]
    t1_present = [e for e in T1_DOWN if e in eidx]
    overlap = sorted(set(our_top10) & set(t1_present))
    # Spearman-ish: rank of each T.1 emotion in our suppression order
    our_rank = {e: int(np.where(order == eidx[e])[0][0]) for e in t1_present}

    # ---- IT-own-probe sensitivity (§2.1): clean IT canonical raw thru IT 100T neutral basis ----
    sens = None
    if IT_NEUTRAL_100.exists() and IT_CANON.exists():
        sys.path.insert(0, str(REPO))
        from emotion_probes.probes import project_out_top_pcs
        itcanon = torch.load(IT_CANON, map_location="cpu", weights_only=False)
        assert list(itcanon["emotions"]) == emotions, "IT canonical emotion order differs"
        it_neu = torch.load(IT_NEUTRAL_100, map_location="cpu", weights_only=False)["acts"].float()
        Vit = project_out_top_pcs(itcanon["vectors_raw"].float(), it_neu, 0.5).numpy()
        Vit_hat = Vit / np.clip(np.linalg.norm(Vit, axis=2, keepdims=True), 1e-12, None)
        proj_it_own = unitnorm_proj(it_bare, Vit_hat)              # IT residuals on IT-own probes
        # base on E1 probes vs IT on IT-own probes is NOT comparable-by-construction; report IT-own
        # triad projection level only, as the disclosed sensitivity.
        sens = {"note": "IT bare-final residuals projected onto IT's OWN 100T-cleaned probes "
                        "(canonical raw thru gemma-2-27b-it__neutral_100T). Sensitivity only; "
                        "not paired to base (different probe set breaks comparability).",
                "it_own_triad_L31_mean": {e: float(proj_it_own[:, eidx[e], L31].mean()) for e in PRIMARY}}

    # ---- §4.6 datum: the two IT format arms contrasted at L31 (tertiary; excluded from P-colon) ----
    arm_L31 = {}
    for arm in ("bare", "template", "colon"):
        r = it["arms"][arm]["residuals"].float().numpy()
        arm_L31[arm] = {e: float(unitnorm_proj(r, Vhat)[:, eidx[e], L31].mean()) for e in PRIMARY}

    report = {
        "job": "GGE1-G4 / P-colon analysis (W7)",
        "basis": "100T (probes = emotion_vectors_171.pt cleaned vectors, both sides; "
                 "sha stamped in inputs_sha256.vectors_100T)",
        "convention": "frozen fig2 §1: bare-final, "
        "L31 resid_post f32, unit-normalized dot resid·(v/‖v‖), E1 100T probes both sides",
        "n_scenarios": S, "layer": L31,
        "pcolon_primary_probes": per_probe,
        "pcolon_aggregate_triad": agg,
        "band_means_triad_diff_IT_minus_base": band_diff,
        "rank_overlap_vs_T1": {"our_top10_suppressed_L31": our_top10,
                               "T1_down_top10": T1_DOWN,
                               "overlap_set": overlap, "n_overlap": len(overlap),
                               "T1_emotions_rank_in_our_suppression": our_rank},
        "it_own_probe_sensitivity": sens,
        "format_arms_L31_triax_datum_sec46": arm_L31,
        "inputs_sha256": {"scenario_acts": sha(BASE), "colon_it": sha(IT), "vectors_100T": sha(V100)},
        "grading_note": "producer reports numbers; Grid/coordination adjudicates the P-colon grade.",
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "g4_pcolon_grade.json").write_text(json.dumps(report, indent=1, default=float))

    print("=== W7 P-colon — numbers (frozen convention; grade = coordination) ===")
    print(f"\nPrimary probes, IT−base unit-norm proj @L31 (predict IT_lower = negative):")
    print(f"  {'probe':13s} {'base':>9s} {'IT':>9s} {'diff':>9s} {'sem':>8s} {'z':>7s}  sign")
    for e in PRIMARY + SECONDARY:
        s = per_probe[e]
        tag = "(secondary)" if e in SECONDARY else ""
        print(f"  {e:13s} {s['base_mean']:+9.3f} {s['it_mean']:+9.3f} {s['diff_mean']:+9.3f} "
              f"{s['diff_sem']:8.3f} {s['z_vs_sem']:+7.2f}  {s['sign']} {tag}")
    print(f"\nAggregate triad {PRIMARY}: diff={agg['diff_mean']:+.4f} sem={agg['diff_sem']:.4f} "
          f"z={agg['z_vs_sem']:+.2f} -> {agg['sign']}  [{agg['points_toward']}]")
    print(f"Band means (triad IT−base): early={band_diff['early']:+.4f} mid={band_diff['mid']:+.4f} late={band_diff['late']:+.4f}")
    print(f"\nRank overlap vs T.1 down-list: {len(overlap)}/10 in our top-10 suppressed = {overlap}")
    print(f"  T.1 emotion ranks in our suppression order (0=most suppressed): {our_rank}")
    if sens:
        print(f"\nIT-own-probe sensitivity (triad, IT proj @L31): {sens['it_own_triad_L31_mean']}")
    print(f"\n§4.6 format-arm datum (triad proj @L31, tertiary): " +
          " | ".join(f"{a}: " + ",".join(f"{e[:4]}{arm_L31[a][e]:+.2f}" for e in PRIMARY)
                     for a in ("bare", "template", "colon")))
    print(f"\nwrote {(OUTDIR / 'g4_pcolon_grade.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
