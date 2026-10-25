#!/usr/bin/env python3
"""
L45 probe analysis + LOCKED decision rule (handoff §5). CPU-only.

    python gemma_grid/probe_l45/analyze_probe_l45.py

Reads probe_l45_means.pt + the locked vectors_171.pt, computes the three RSA
values, applies the pre-stated rule, and writes PROBE_REPORT.md. Does NOT touch
the ledger (producing session doesn't grade its own result — §6.3).

Decision rule (LOCKED before the run — not adjusted after seeing results):
  pipeline control  RSA(probe-L31, locked-L31) >= 0.980   else VOID
  near-band control RSA(probe-L42, locked-L42) >= 0.95    else (if L31 ok) ESCALATE
  verdict           RSA(probe-L45, locked-L45) >= 0.95           -> GENUINE
                    RSA(probe-L45, locked-L45) <= 0.90 AND
                        RSA(probe-L45, locked-L42) >= 0.95        -> ARTIFACT
                    else                                          -> INDETERMINATE
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "gemma_grid" / "probe_l45" / "probe_l45_means.pt"
LOCKED = REPO / "data" / "vectors" / "emotion_vectors_171.pt"  # reads emotions and vectors_raw only, identical to the as-run vectors_171.pt
REPORT = REPO / "gemma_grid" / "probe_l45" / "PROBE_REPORT.md"


def diff_of_means(M):                       # M: (n, d) per-emotion means -> raw diff-of-means
    return M - M.mean(axis=0, keepdims=True)


def cos_structure(V):                       # V: (n, d) -> upper-triangle of the n x n cosine matrix
    n = V.shape[0]
    Vn = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
    C = Vn @ Vn.T
    iu = np.triu_indices(n, 1)
    return C[iu]


def rsa(a, b):                              # Pearson correlation of two structure vectors
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    if not PROBE.exists():
        print(f"missing {PROBE} — run the probe extraction first."); return 1
    p = torch.load(PROBE, map_location="cpu", weights_only=False)
    lk = torch.load(LOCKED, map_location="cpu", weights_only=False)
    pem = list(p["emotions"]); lem = list(lk["emotions"])
    assert pem == lem, "emotion order differs between probe and locked anchor"
    layers = list(p["layers"])
    means = p["means"].float().numpy()                       # (171, 3, 4608)
    Vraw = lk["vectors_raw"].float().numpy()                 # (171, 46, 4608)

    # structure vectors per layer (probe: diff-of-means of the fresh means; locked: vectors_raw already diff-of-means)
    probe_struct = {L: cos_structure(diff_of_means(means[:, i, :])) for i, L in enumerate(layers)}
    locked_struct = {L: cos_structure(Vraw[:, L, :]) for L in layers}

    R = {}  # RSA(probe-La, locked-Lb)
    for La in layers:
        for Lb in layers:
            R[(La, Lb)] = rsa(probe_struct[La], locked_struct[Lb])

    r_l31 = R[(31, 31)]; r_l42 = R[(42, 42)]; r_l45 = R[(45, 45)]; r_l45_vs_l42 = R[(45, 42)]

    # --- LOCKED rule ---
    void = r_l31 < 0.980
    if void:
        verdict = "VOID"; detail = "pipeline control failed — do not interpret L42/L45; stop and diagnose (§7)"
    else:
        near_ok = r_l42 >= 0.95
        if r_l45 >= 0.95:
            verdict = "GENUINE"; detail = "fresh extraction reproduces the divergent L45 structure"
        elif r_l45 <= 0.90 and r_l45_vs_l42 >= 0.95:
            verdict = "ARTIFACT"; detail = "fresh L45 is well-behaved / band-like — locked L45 is a single-extraction artifact"
        else:
            verdict = "INDETERMINATE"; detail = "neither branch triggered — report numbers, escalate to Adam (§7)"
        if not near_ok and verdict != "VOID":
            verdict += " + ESCALATE"; detail += " | near-band control L42 < 0.95 while L31 passed — late-layer vintage suspicion widens (E1-relevant)"

    e1_line = {
        "GENUINE": "E1 implication: E1's late-layer band numbers (mid-late 0.949, Table 2) gain an INDEPENDENT replication; G3.4 caveat upgrades to 'replicated, genuine property'.",
        "ARTIFACT": "E1 implication: E1 MUST review its late-layer band numbers before submission — locked L45 is a round-one artifact; grid rows updated.",
        "VOID": "E1 implication: none — probe void, re-run/diagnose before any claim.",
        "INDETERMINATE": "E1 implication: unresolved — escalate to Adam; no E1 change either direction.",
    }[verdict.split(" + ")[0]]

    prov = p.get("provenance", {})
    lines = []
    lines.append("# L45 Probe — Report"); lines.append("")
    lines.append(f"**Verdict: {verdict}** — {detail}"); lines.append("")
    lines.append(e1_line); lines.append("")
    lines.append("## Fingerprint / provenance")
    for k in ["hf_revision_resolved", "hf_revision_expected", "torch", "transformer_lens",
              "tl_compatibility_mode", "dtype", "skip_first_n_tokens", "n_stories_per_emotion",
              "pod_gpu", "date_utc", "pilot"]:
        if k in prov: lines.append(f"- {k}: `{prov[k]}`")
    fp_ok = prov.get("hf_revision_resolved") == prov.get("hf_revision_expected")
    lines.append(f"- **fingerprint gate: {'PASS' if fp_ok else 'MISMATCH'}**")
    lines.append("")
    lines.append("## RSA (Pearson, 171x171 cosine-structure upper triangle; raw diff-of-means)")
    lines.append("")
    lines.append("| check | value | threshold | result |")
    lines.append("|---|---|---|---|")
    lines.append(f"| pipeline control RSA(probe-L31, locked-L31) | {r_l31:.4f} | ≥ 0.980 | {'PASS' if r_l31>=0.980 else 'FAIL → VOID'} |")
    lines.append(f"| near-band control RSA(probe-L42, locked-L42) | {r_l42:.4f} | ≥ 0.95 | {'PASS' if r_l42>=0.95 else 'FAIL → ESCALATE'} |")
    lines.append(f"| **verdict RSA(probe-L45, locked-L45)** | **{r_l45:.4f}** | ≥0.95 GENUINE / ≤0.90 ARTIFACT | {verdict} |")
    lines.append(f"| ARTIFACT cond RSA(probe-L45, locked-L42) | {r_l45_vs_l42:.4f} | ≥ 0.95 | {'met' if r_l45_vs_l42>=0.95 else 'not met'} |")
    lines.append("")
    lines.append("Full probe×locked RSA matrix:"); lines.append("")
    lines.append("| probe\\locked | " + " | ".join(f"L{L}" for L in layers) + " |")
    lines.append("|" + "---|"*(len(layers)+1))
    for La in layers:
        lines.append(f"| L{La} | " + " | ".join(f"{R[(La,Lb)]:.4f}" for Lb in layers) + " |")
    lines.append("")
    lines.append("Calibration ceiling (120-of-1200 subsample structure-RSA, measured): ~0.9856 mean, "
                 "p05 0.9841. A faithful layer sits near this; the artifact signature sits ~0.15 below.")
    lines.append("")
    lines.append("## Handoff §6.3 — this session does NOT grade the ledger")
    lines.append("Hand this report + `probe_l45_means.pt` back to Adam → grid session (Cowork) applies "
                 "the pre-stated rule to the ledger and cross-checks the RSA arithmetic from the artifact.")

    REPORT.write_text("\n".join(lines))
    print("\n".join(lines[:12]))
    print(f"\n... full report written to {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
