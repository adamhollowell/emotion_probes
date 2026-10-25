#!/usr/bin/env python3
"""Reproduce the registered tuning-family Fisher p values from source (SR.4).

Why this exists. G2.1's Method line states the test (Fisher r-to-z, n=45) and
the pooled definition (mean of three per-size deltas), and until 2026-10-08 it
did not state the SE the POOLED p is computed against. A reader who applies the
single-pair SE to the pooled delta reproduces .566 for the valence row instead
of its .320. This script fixes both SEs by reproducing every printed p, so the
pooled SE is a re-derived number rather than prose.

What it asserts, from the six hash-pinned stageA_summary.json deterministic
blocks and stageC_grading.json:
  per-size pairwise  p = 2*Phi(-|dz| / sqrt(2/42)),  SE 0.21821789023599236
  pooled             p = 2*Phi(-|mean dz| / (sqrt(2/42)/sqrt(3))), SE 0.12598815766974242
dz = atanh(r_it) - atanh(r_base) per size pair. Both correlation families
(valence_r, arousal_r) are checked, eight p values in all.

Consequence the assertions make visible: the registered pooled test assumes
independence TWICE, across the 45 rated emotions within a size pair and across
the three size pairs. The dependence-aware bootstraps at G3.2/G3.3 address only
the first. Nothing rests on the net, because every floored tuning row is
NULL_subfloor, a subtype in which the SE is not an input.

Stdlib only. Writes an SR.5 input manifest beside its output.
Usage: python3 gemma_grid/scripts/verify_tuning_fisher_pooled.py
"""
import hashlib, json, sys
from math import atanh, erfc, sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "gemma_grid" / "geometry_out"
SIZES = ("2b", "9b", "27b")
SE1 = sqrt(2.0 / 42.0)
SE_POOLED = SE1 / sqrt(3.0)
TOL = 1e-9


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def two_sided(z):
    return erfc(abs(z) / sqrt(2.0))


def main():
    inputs, r = {}, {}
    for size in SIZES:
        for tune in ("base", "it"):
            cell = "gemma-2-%s-%s" % (size, tune)
            p = OUT / cell / "stageA_summary.json"
            inputs[str(p.relative_to(REPO))] = sha(p)
            d = json.load(open(p))["deterministic"]
            r[(size, tune)] = (d["valence_r"], d["arousal_r"])
    gp = OUT / "stageC_grading.json"
    inputs[str(gp.relative_to(REPO))] = sha(gp)
    tests = json.load(open(gp))["tests"]

    def artifact_p(signal, contrast):
        hits = [t for t in tests if t.get("family") == "tuning"
                and t.get("signal") == signal and t.get("contrast") == contrast]
        if len(hits) != 1:
            sys.exit("FATAL: %d rows for %s / %s" % (len(hits), signal, contrast))
        return hits[0]["p"]

    rows, ok = [], True
    for signal, idx in (("valence_r", 0), ("arousal_r", 1)):
        dzs = []
        for size in SIZES:
            dz = atanh(r[(size, "it")][idx]) - atanh(r[(size, "base")][idx])
            dzs.append(dz)
            got, want = two_sided(dz / SE1), artifact_p(signal, size)
            good = abs(got - want) <= TOL
            ok = ok and good
            rows.append({"signal": signal, "contrast": size, "delta_z": dz,
                         "se_used": SE1, "p_recomputed": got, "p_artifact": want,
                         "match": good})
        mz = sum(dzs) / 3.0
        got = two_sided(mz / SE_POOLED)
        want = artifact_p(signal, "POOLED[IT-base]")
        good = abs(got - want) <= TOL
        ok = ok and good
        rows.append({"signal": signal, "contrast": "POOLED[IT-base]", "delta_z": mz,
                     "se_used": SE_POOLED, "p_recomputed": got, "p_artifact": want,
                     "match": good,
                     "p_if_single_pair_se_used": two_sided(mz / SE1)})

    # --- Holm ladder over the tuning family (m = in_holm rows carrying a p) ---
    inholm = [t for t in tests if t.get("family") == "tuning" and t.get("in_holm")]
    withp = sorted([t for t in inholm if t.get("p") is not None], key=lambda t: t["p"])
    m = len(withp)
    ladder, stopped = [], False
    for i, t in enumerate(withp, start=1):
        thr = 0.05 / (m - i + 1)
        rejected = (not stopped) and t["p"] <= thr
        if not rejected:
            stopped = True
        agree_alpha = abs(thr - t.get("holm_alpha", float("nan"))) <= 1e-15
        agree_sig = (rejected == bool(t.get("holm_significant")))
        ok = ok and agree_alpha and agree_sig
        ladder.append({"rank": i, "signal": t.get("signal"), "p": t["p"], "holm_threshold": thr,
                       "rejected": rejected, "artifact_holm_alpha": t.get("holm_alpha"),
                       "artifact_holm_significant": t.get("holm_significant"),
                       "threshold_agrees": agree_alpha, "flag_agrees": agree_sig,
                       "verdict": t.get("verdict")})
    survivors = [x["signal"] for x in ladder if x["rejected"]]
    noP = [{"signal": t.get("signal"), "floor": t.get("floor"), "verdict": t.get("verdict")}
           for t in inholm if t.get("p") is None]

    rec = {
        "record": "registered tuning Fisher p reproduction (SR.4 serialization of the pooled SE) + Holm ladder",
        "se_single_pair": SE1, "se_pooled": SE_POOLED,
        "se_pooled_construction": "sqrt(2/42)/sqrt(3) -- three size pairs treated as independent",
        "independence_assumptions": ["across the 45 rated emotions within a size pair",
                                     "across the three size pairs"],
        "tolerance": TOL, "all_match": ok, "rows": rows,
        "holm": {"m": m, "m_rule": "in_holm tuning rows CARRYING a p; floor-only rows with no p are excluded",
                 "in_holm_rows_total": len(inholm), "in_holm_rows_without_p": noP,
                 "alpha": 0.05, "ladder": ladder, "survivors": survivors,
                 "note": "the ladder stops at the first failure, so rows below it are unrejected too"},
        "script": str(Path(__file__).resolve().relative_to(REPO)),
        "script_sha256": sha(Path(__file__).resolve()),
    }
    (OUT / "tuning_fisher_pooled_verification.json").write_text(json.dumps(rec, indent=1))
    (OUT / "tuning_fisher_pooled_verification_manifest.json").write_text(json.dumps(
        {"m9_note": "SR.5 input manifest -- sha256 of every input, written at run time.",
         "inputs": inputs,
         "output": "gemma_grid/geometry_out/tuning_fisher_pooled_verification.json"},
        indent=1))
    for x in rows:
        print("%-10s %-16s dz=%+.9f se=%.17f p=%.9f artifact=%.9f %s"
              % (x["signal"], x["contrast"], x["delta_z"], x["se_used"],
                 x["p_recomputed"], x["p_artifact"], "OK" if x["match"] else "MISMATCH"))
    print()
    print("Holm ladder, m=%d (alpha 0.05):" % m)
    for x in ladder:
        print("  %d %-28s p=%-14.8g thr=%-10.7f %-8s  artifact agrees: alpha %s sig %s"
              % (x["rank"], x["signal"], x["p"], x["holm_threshold"],
                 "REJECT" if x["rejected"] else "no", x["threshold_agrees"], x["flag_agrees"]))
    print("  survivors: %s" % (", ".join(survivors) or "none"))
    print()
    print("ALL MATCH" if ok else "MISMATCH -- see output")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
