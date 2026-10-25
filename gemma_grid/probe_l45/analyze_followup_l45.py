#!/usr/bin/env python3
"""
L45 probe FOLLOW-UP analysis — resolves the INDETERMINATE via the second seeded
draw + the attenuation model. CPU-only.

    python gemma_grid/probe_l45/analyze_followup_l45.py

LOCKED interpretation rule (pre-registered by the grid/Cowork session BEFORE the
seed-1 number existed; this session applies it mechanically, does not grade):
  Attenuation model (validated on L31 calibration, 435 pairs, accurate to 0.0007,
  pair sd 0.0019): fresh-vs-locked at ceiling ≈ sqrt(fresh-vs-fresh).
  P = sqrt(R_ff)   where R_ff = RSA(seed0-L45, seed1-L45)
  GENUINE      if seed0-vs-locked (0.9486) AND seed1-vs-locked are BOTH >= P-0.010
  MIDDLE       if BOTH <= P-0.020 while the fresh draws agree with each other
               ("genuine divergence, but locked-L45's realization carries
                vintage-specific detail")
  INDETERMINATE otherwise (between the margins, or internal inconsistency)
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "probe_l45"))
from analyze_probe_l45 import diff_of_means, cos_structure, rsa   # noqa: E402

S0 = REPO / "gemma_grid/probe_l45/probe_l45_means.pt"
S1 = REPO / "gemma_grid/probe_l45/probe_l45_means_seed1.pt"
LOCKED = REPO / "data/vectors/emotion_vectors_171.pt"  # reads emotions and vectors_raw only, identical to the as-run vectors_171.pt
REPORT = REPO / "gemma_grid/probe_l45/PROBE_REPORT_FOLLOWUP.md"
LAYERS = [31, 42, 45]


def main() -> int:
    for p in (S0, S1, LOCKED):
        if not p.exists():
            print(f"missing {p}"); return 1
    d0 = torch.load(S0, map_location="cpu", weights_only=False)
    d1 = torch.load(S1, map_location="cpu", weights_only=False)
    lk = torch.load(LOCKED, map_location="cpu", weights_only=False)
    assert list(d0["emotions"]) == list(d1["emotions"]) == list(lk["emotions"]), "emotion order mismatch"
    assert d0["provenance"]["sampling"]["seed"] == 0 and d1["provenance"]["sampling"]["seed"] == 1
    m0 = d0["means"].float().numpy(); m1 = d1["means"].float().numpy()
    Vraw = lk["vectors_raw"].float().numpy()

    s0 = {L: cos_structure(diff_of_means(m0[:, i, :])) for i, L in enumerate(LAYERS)}
    s1 = {L: cos_structure(diff_of_means(m1[:, i, :])) for i, L in enumerate(LAYERS)}
    lkk = {L: cos_structure(Vraw[:, L, :]) for L in LAYERS}

    R_ff = {L: rsa(s0[L], s1[L]) for L in LAYERS}                    # fresh-vs-fresh
    s0_lk = {L: rsa(s0[L], lkk[L]) for L in LAYERS}                  # seed0 vs locked
    s1_lk = {L: rsa(s1[L], lkk[L]) for L in LAYERS}                  # seed1 vs locked
    P = {L: math.sqrt(max(R_ff[L], 0.0)) for L in LAYERS}
    band0 = rsa(s0[45], lkk[42]); band1 = rsa(s1[45], lkk[42])      # band-unlikeness (ARTIFACT cond)

    # attenuation-model self-check at L31 (should predict seed0-vs-locked ~0.9849)
    l31_pred = P[31]; l31_obs = s0_lk[31]; l31_err = abs(l31_pred - l31_obs)

    # LOCKED RULE at L45.
    s0_45, s1_45, PL = s0_lk[45], s1_lk[45], P[45]
    m1_thr, m2_thr = PL - 0.010, PL - 0.020
    # internal consistency: the two independent draws must agree vs-locked, AND
    # the attenuation model must hold on our own L31 data (else P is untrustworthy).
    consistent = (abs(s0_45 - s1_45) <= 0.02) and (l31_err <= 0.005)
    if not consistent:
        verdict = "INDETERMINATE"
        why = (f"internal inconsistency — |seed0-seed1|={abs(s0_45-s1_45):.4f} (>0.02?) or "
               f"L31 attenuation self-check |Δ|={l31_err:.4f} (>0.005?); P not trustworthy")
    elif s0_45 >= m1_thr and s1_45 >= m1_thr:
        verdict = "GENUINE"; why = f"both seed-vs-locked >= P-0.010 ({m1_thr:.4f}); at L45's own measured ceiling"
    elif s0_45 <= m2_thr and s1_45 <= m2_thr:
        verdict = "MIDDLE (genuine divergence + vintage-specific L45 realization)"
        why = f"both seed-vs-locked <= P-0.020 ({m2_thr:.4f}) while the two fresh draws agree (R_ff={R_ff[45]:.4f})"
    else:
        verdict = "INDETERMINATE"; why = f"between the margins ({m2_thr:.4f} < value < {m1_thr:.4f})"

    L = []
    L.append("# L45 Probe — FOLLOW-UP report (attenuation-model resolution)"); L.append("")
    L.append(f"**Verdict (per the pre-registered follow-up rule): {verdict}**"); L.append(f"- {why}"); L.append("")
    L.append("## The resolution at L45")
    L.append("")
    L.append(f"- R_ff = RSA(seed0-L45, seed1-L45) = **{R_ff[45]:.4f}**  (fresh-vs-fresh, no vintage confound)")
    L.append(f"- P = sqrt(R_ff) = **{PL:.4f}**  → at-ceiling prediction for fresh-vs-locked at L45")
    L.append(f"- seed0-vs-locked L45 = **{s0_45:.4f}**   (margin P-0.010 = {m1_thr:.4f}, P-0.020 = {m2_thr:.4f})")
    L.append(f"- seed1-vs-locked L45 = **{s1_45:.4f}**")
    L.append("")
    L.append("Note: L45's ceiling P is measured here to be **below** L31's 0.9856 iff R_ff is low — "
             "the missing-ceiling assumption that made this INDETERMINATE is now measured directly.")
    L.append("")
    L.append("## Attenuation-model self-check (L31, our own data)")
    L.append(f"- sqrt(R_ff[L31]) = {l31_pred:.4f}  vs  seed0-vs-locked[L31] = {l31_obs:.4f}  |Δ|={l31_err:.4f} "
             f"({'model holds' if l31_err<=0.005 else 'MODEL DEVIATES — flag'})")
    L.append("")
    L.append("## Band-unlikeness (ARTIFACT branch — should be LOW if L45 divergence is real)")
    L.append(f"- seed0-L45 vs locked-L42 = {band0:.4f} | seed1-L45 vs locked-L42 = {band1:.4f}  (≥0.95 would be band-like)")
    L.append("")
    L.append("## Per-layer table")
    L.append("")
    L.append("| L | R_ff (s0,s1) | P=√R_ff | s0-vs-locked | s1-vs-locked |")
    L.append("|---|---|---|---|---|")
    for ll in LAYERS:
        L.append(f"| {ll} | {R_ff[ll]:.4f} | {P[ll]:.4f} | {s0_lk[ll]:.4f} | {s1_lk[ll]:.4f} |")
    L.append("")
    L.append("## Provenance (both draws)")
    for tag, d in [("seed0", d0), ("seed1", d1)]:
        pr = d["provenance"]
        L.append(f"- {tag}: rev `{pr['hf_revision_resolved'][:12]}` | sampling `{pr['sampling']}` | "
                 f"fingerprint `{'PASS' if pr['hf_revision_resolved']==pr['hf_revision_expected'] else 'MISMATCH'}` | "
                 f"{pr['n_stories_per_emotion']} stories | {pr.get('pod_gpu')}")
    L.append("")
    L.append("## §6.3 — this session does NOT grade the ledger")
    L.append("Hand this + both artifacts back to Adam → grid/Cowork session for the ledger, "
             "who cross-check the arithmetic. The rule above was pre-registered by that session.")
    REPORT.write_text("\n".join(L))
    print("\n".join(L[:16]))
    print(f"\n... full follow-up report -> {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
