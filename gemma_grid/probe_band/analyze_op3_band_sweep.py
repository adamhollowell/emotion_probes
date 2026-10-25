#!/usr/bin/env python3
"""W11 — OP.3 band-sweep analysis (GGE1-G3). CPU, post-session, Grid-graded.

Applies the pre-stated analysis of OP3_BAND_SWEEP_SPEC.md §"Pre-stated analysis"
to the two-seed band extraction {31,42,43,44,45}. Reuses the probe record's exact
instrument (diff_of_means, cos_structure, rsa from analyze_probe_l45) verbatim —
nothing about RSA is reimplemented. RAW structures only (basis-immune): valid
under every cleanup basis.

  1. Per-layer ceilings   R_ff(l)=RSA(seed0-l,seed1-l); P(l)=sqrt(R_ff). Internal-
     consistency gate: reproduce P(45)=0.9462, P(31)=0.9856 within +/-0.005.
  2. Band shape           RSA(fresh-l, locked-l) vs P(l)-0.010, three-branch grade
     (GENUINE / vintage-flavored / INDETERMINATE) per the probe record's rule;
     consecutive-layer corr sequence + band-unlikeness vs locked-L42 (cliff vs slope).
  3. Mechanism            dimension attribution of the L44->L45 divergence (recovery
     fraction after dropping top-k change-dims; k=1,10,100; dim 4140 individually).
     ** operationalization DOF — see NOTE_PART3 — no verdict, seeds the question. **
  4. IT comparison        the same within-band instrument on 27b-it's raw means.

Producer does NOT grade. Output -> gemma_grid/geometry_out/.
RUN: ./.venv/bin/python gemma_grid/probe_band/analyze_op3_band_sweep.py
"""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "probe_l45"))
from analyze_probe_l45 import diff_of_means, cos_structure, rsa   # the exact instrument

S0 = REPO / "gemma_grid/probe_band/probe_band_means_seed0.pt"
S1 = REPO / "gemma_grid/probe_band/probe_band_means_seed1.pt"
LOCKED = REPO / "data/vectors/emotion_vectors_171.pt"             # raw = basis-immune; raw identical to the as-run vectors_171.pt
IT = REPO / "gemma_grid/data/gemma-2-27b-it/vectors/canonical.pt"  # raw = basis-immune
OUTDIR = REPO / "gemma_grid" / "geometry_out"
BAND = [31, 42, 43, 44, 45]
LOCKED_CEIL = {45: 0.9462, 31: 0.9856}     # probe-record ceilings to reproduce
CEIL_TOL = 0.005
MASSIVE_DIM = 4140                          # M.5 massive-activation dim (PRECISION27B_REPORT)
TOPK = [1, 10, 100]

NOTE_PART3 = ("Part-3 attribution operationalization (DOF resolved by the diag_T2.py "
              "precedent, flagged to coordination): dims ranked by their share of the "
              "L44->L45 change in the per-emotion diff-of-means (the object RSA sees), "
              "i.e. per-dim sum over emotions of (D45-D44)^2 where D_l = diff_of_means "
              "of the seed-averaged fresh means at layer l. 'Recovery fraction' = "
              "(RSA_after_drop - RSA_base)/(1 - RSA_base), RSA_base = RSA(struct L44, "
              "struct L45) on the seed-averaged fresh diff-of-means. dim 4140 reported "
              "individually. No verdict attached — first mechanism measurement only.")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def three_branch(P, s0_lk, s1_lk, l31_err):
    """Probe-record three-branch rule at one layer."""
    m1, m2 = P - 0.010, P - 0.020
    consistent = (abs(s0_lk - s1_lk) <= 0.02) and (l31_err <= 0.005)
    if not consistent:
        return "INDETERMINATE", f"internal inconsistency (|s0-s1|={abs(s0_lk-s1_lk):.4f}, L31 self-check |d|={l31_err:.4f})"
    if s0_lk >= m1 and s1_lk >= m1:
        return "GENUINE", f"both seed-vs-locked >= P-0.010 ({m1:.4f})"
    if s0_lk <= m2 and s1_lk <= m2:
        return "vintage-flavored", f"both seed-vs-locked <= P-0.020 ({m2:.4f}) while fresh draws agree"
    return "INDETERMINATE", f"between margins ({m2:.4f} < value < {m1:.4f})"


def main():
    for p in (S0, S1, LOCKED, IT):
        if not p.exists():
            sys.exit(f"missing {p}")
    d0 = torch.load(S0, map_location="cpu", weights_only=False)
    d1 = torch.load(S1, map_location="cpu", weights_only=False)
    lk = torch.load(LOCKED, map_location="cpu", weights_only=False)
    it = torch.load(IT, map_location="cpu", weights_only=False)
    assert list(d0["emotions"]) == list(d1["emotions"]) == list(lk["emotions"]) == list(it["emotions"]), "emotion order mismatch"
    assert list(d0["layers"]) == list(d1["layers"]) == BAND, f"layers {d0['layers']} != {BAND}"
    assert d0["provenance"]["sampling"]["seed"] == 0 and d1["provenance"]["sampling"]["seed"] == 1
    m0 = d0["means"].float().numpy(); m1 = d1["means"].float().numpy()   # (171,5,4608)
    Vraw = lk["vectors_raw"].float().numpy()                            # (171,46,4608)
    ITraw = it["vectors_raw"].float().numpy()

    idx = {L: i for i, L in enumerate(BAND)}
    s0 = {L: cos_structure(diff_of_means(m0[:, idx[L], :])) for L in BAND}
    s1 = {L: cos_structure(diff_of_means(m1[:, idx[L], :])) for L in BAND}
    lkk = {L: cos_structure(Vraw[:, L, :]) for L in BAND}

    # ---- Part 1: per-layer ceilings ----
    R_ff = {L: rsa(s0[L], s1[L]) for L in BAND}
    P = {L: math.sqrt(max(R_ff[L], 0.0)) for L in BAND}
    ceil_checks = {L: {"P": P[L], "expected": LOCKED_CEIL[L],
                       "abs_dev": abs(P[L] - LOCKED_CEIL[L]),
                       "pass": abs(P[L] - LOCKED_CEIL[L]) <= CEIL_TOL} for L in LOCKED_CEIL}
    ceil_gate = all(c["pass"] for c in ceil_checks.values())

    # ---- Part 2: band shape ----
    s0_lk = {L: rsa(s0[L], lkk[L]) for L in BAND}
    s1_lk = {L: rsa(s1[L], lkk[L]) for L in BAND}
    l31_err = abs(P[31] - s0_lk[31])          # attenuation self-check (should be ~0)
    band_layers = [42, 43, 44, 45]
    grades = {}
    for L in band_layers:
        v, why = three_branch(P[L], s0_lk[L], s1_lk[L], l31_err)
        grades[L] = {"grade": v, "why": why, "P": P[L], "P_minus_010": P[L]-0.010,
                     "P_minus_020": P[L]-0.020, "seed0_vs_locked": s0_lk[L],
                     "seed1_vs_locked": s1_lk[L]}
    # cliff vs slope: consecutive-layer corr (fresh, seed-averaged) + band-unlikeness vs locked-L42
    mA = (m0 + m1) / 2.0
    sA = {L: cos_structure(diff_of_means(mA[:, idx[L], :])) for L in BAND}
    consec = {f"{a}->{b}": rsa(sA[a], sA[b]) for a, b in zip(BAND, BAND[1:])}
    band_unlike = {L: {"seed0_L45style_vs_lockedL42": rsa(s0[L], lkk[42]),
                       "seed1_vs_lockedL42": rsa(s1[L], lkk[42])} for L in band_layers}

    # ---- Part 3: mechanism, L44->L45 dim attribution (see NOTE_PART3) ----
    D44 = diff_of_means(mA[:, idx[44], :])    # (171,4608)
    D45 = diff_of_means(mA[:, idx[45], :])
    rsa_base = rsa(cos_structure(D44), cos_structure(D45))
    change = ((D45 - D44) ** 2).sum(axis=0)   # per-dim L44->L45 change magnitude
    order = np.argsort(-change)

    def recovery(drop_idx):
        keep = np.ones(D44.shape[1], bool); keep[list(drop_idx)] = False
        r = rsa(cos_structure(D44[:, keep]), cos_structure(D45[:, keep]))
        frac = (r - rsa_base) / (1 - rsa_base) if (1 - rsa_base) > 1e-9 else float("nan")
        return {"rsa_after_drop": r, "recovery_fraction": frac}

    part3 = {"note": NOTE_PART3, "rsa_L44_L45_base": rsa_base,
             "top1_dim": int(order[0]), "top1": recovery(order[:1]),
             "top10": recovery(order[:10]), "top100": recovery(order[:100]),
             "dim4140_individually": recovery([MASSIVE_DIM]),
             "dim4140_rank_in_change_order": int(np.where(order == MASSIVE_DIM)[0][0]),
             "dim4140_change_share_pct": float(change[MASSIVE_DIM] / change.sum() * 100)}

    # ---- Part 4: IT within-band instrument (single realization; no seed pair) ----
    itk = {L: cos_structure(ITraw[:, L, :]) for L in BAND}
    it_consec = {f"{a}->{b}": rsa(itk[a], itk[b]) for a, b in zip(BAND, BAND[1:])}
    it_vs_L42 = {L: rsa(itk[L], itk[42]) for L in band_layers}
    base_vs_L42 = {L: rsa(lkk[L], lkk[42]) for L in band_layers}     # base locked, same instrument
    part4 = {"IT_consecutive_corr": it_consec, "IT_band_vs_L42": it_vs_L42,
             "BASE_locked_band_vs_L42": base_vs_L42,
             "note": ("IT has one realization (canonical raw), so no R_ff/ceiling; the dip is "
                      "read as within-band RSA(l, L42). Compare IT_band_vs_L42 to "
                      "BASE_locked_band_vs_L42 for the 'milder dip' note.")}

    report = {
        "job": "GGE1-G3 / OP.3 band-sweep analysis (W11)", "band_layers": BAND,
        "basis": "100T session output. NOTE: this analysis operates on RAW diff-of-means "
                 "structures (basis-immune — byte-identical under PA/HA/100T); the locked "
                 "reference vectors_raw sha is stamped in inputs_sha256 so no value rides on "
                 "the filename.",
        "part1_ceilings": {"R_ff": R_ff, "P": P, "ceiling_reproduction": ceil_checks,
                           "ceiling_gate_pass": ceil_gate,
                           "l31_attenuation_selfcheck_absdev": l31_err},
        "part2_band_shape": {"seed0_vs_locked": s0_lk, "seed1_vs_locked": s1_lk,
                             "grades": grades, "consecutive_corr_seedavg": consec,
                             "band_unlikeness_vs_lockedL42": band_unlike},
        "part3_mechanism_L44_L45": part3,
        "part4_IT_comparison": part4,
        "inputs_sha256": {"seed0": sha(S0), "seed1": sha(S1), "locked": sha(LOCKED), "it": sha(IT)},
        "provenance": {"seed0": d0["provenance"], "seed1": d1["provenance"]},
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "op3_band_sweep_analysis.json").write_text(json.dumps(report, indent=1, default=float))

    # ---- console: numbers, not narration ----
    print("=== W11 OP.3 band-sweep — numbers ===")
    print("\n[1] per-layer ceilings  R_ff, P=sqrt(R_ff)")
    for L in BAND:
        tag = ""
        if L in LOCKED_CEIL:
            c = ceil_checks[L]; tag = f"  (expect {c['expected']:.4f}, dev {c['abs_dev']:.4f} {'PASS' if c['pass'] else 'FAIL'})"
        print(f"    L{L}: R_ff={R_ff[L]:.4f}  P={P[L]:.4f}{tag}")
    print(f"    ceiling-reproduction gate: {'PASS' if ceil_gate else 'FAIL — flag to coordination'}")
    print(f"    L31 attenuation self-check |d|={l31_err:.4f} ({'ok' if l31_err<=0.005 else 'MODEL DEVIATES'})")
    print("\n[2] band shape  RSA(fresh-l, locked-l) + grade")
    for L in band_layers:
        g = grades[L]
        print(f"    L{L}: s0={g['seed0_vs_locked']:.4f} s1={g['seed1_vs_locked']:.4f} "
              f"P-0.010={g['P_minus_010']:.4f}  -> {g['grade']}")
    print(f"    consecutive corr (seed-avg): " + "  ".join(f"{k}={v:.4f}" for k, v in consec.items()))
    print("    band-unlikeness vs locked-L42:")
    for L in band_layers:
        print(f"      L{L}: s0={band_unlike[L]['seed0_L45style_vs_lockedL42']:.4f} "
              f"s1={band_unlike[L]['seed1_vs_lockedL42']:.4f}")
    print("\n[3] mechanism L44->L45 (no verdict; DOF flagged)")
    print(f"    RSA(L44,L45) base={rsa_base:.4f}  top1 dim={part3['top1_dim']}")
    for k in ("top1", "top10", "top100", "dim4140_individually"):
        print(f"    {k:22s}: rsa_after={part3[k]['rsa_after_drop']:.4f}  recovery={part3[k]['recovery_fraction']:.4f}")
    print(f"    dim4140 rank in change-order={part3['dim4140_rank_in_change_order']}  "
          f"change-share={part3['dim4140_change_share_pct']:.3f}%")
    print("\n[4] IT comparison (milder-dip instrument)")
    print(f"    IT consecutive corr:   " + "  ".join(f"{k}={v:.4f}" for k, v in it_consec.items()))
    print(f"    IT band vs L42:        " + "  ".join(f"L{L}={it_vs_L42[L]:.4f}" for L in band_layers))
    print(f"    BASE locked band vs L42:" + "  ".join(f"L{L}={base_vs_L42[L]:.4f}" for L in band_layers))
    print(f"\nwrote {(OUTDIR / 'op3_band_sweep_analysis.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
