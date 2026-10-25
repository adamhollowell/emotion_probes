#!/usr/bin/env python3
"""Enumerate the four candidate constructions of the registered interaction DiD,
and identify which one each shipped number is (SR.4).

Why this exists. G4.1 reports the registered arousal interaction as +0.021 and
calls it "registered r-unit". PREREGISTRATION.md (e) registers the quantity as
`([IT-base] arousal r)@27B - ([IT-base] arousal r)@2B`, which in raw r units is
+0.047 and is what a reader gets by subtracting the paper's Table 2 arousal row.
Four constructions are in play and three distinct numbers have appeared in
drafting. This script pins each one to its arithmetic.

RULED (Adam, 2026-10-08): the paper reports A, the preregistration's own r-unit
arithmetic, at the headline interaction, and the stored artifact delta stays C.
Both are on the record at G4.1, and this script is what G4.1 cites to pin them.
The ruling changes no stored value: stageC_grading.json still holds C, and the p
comes from B either way, so the NULL_subfloor verdict is untouched.

  A  raw r-unit DiD      (r_it - r_base)@27B - (r_it - r_base)@2B
  B  Fisher-z DiD        (z_it - z_base)@27B - (z_it - z_base)@2B
  C  tanh-per-size DiD   tanh(dz@27B) - tanh(dz@2B)
  D  tanh of the z DiD   tanh(B)

Findings the assertions lock down:
  * stageC_grading.json's interaction `delta` is C, to 1e-15.
  * the registered p and the "Fisher DiD z=0.07" note come from B / sqrt(4/42).
  * all five Table E.4 delta rows are A, as of the 2026-10-08 table update that
    followed the ruling. Before that update they were C; the pre-update values
    are kept below as PRE_UPDATE_E4 so the change is visible, not silent.
  * norm_robustness_full.py, the named R.3 producer, prints A as its DiD
    line, so with E.4 on A the table is now reproducible from its own cited
    script, delta column included. That closes the SR.4 producer gap.
  * every construction is sub-floor against 0.15 and positive, so the verdict
    NULL_subfloor and the registered direction match hold under all four.

Stdlib only. Writes an SR.5 input manifest beside its output.
Usage: python3 gemma_grid/scripts/verify_interaction_did_constructions.py
"""
import hashlib, json, sys
from math import atanh, erfc, sqrt, tanh
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "gemma_grid" / "geometry_out"
FLOOR = 0.15
TOL = 1e-15


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def two_sided(z):
    return erfc(abs(z) / sqrt(2.0))


def constructions(r2b_base, r2b_it, r27_base, r27_it):
    dr2, dr27 = r2b_it - r2b_base, r27_it - r27_base
    dz2 = atanh(r2b_it) - atanh(r2b_base)
    dz27 = atanh(r27_it) - atanh(r27_base)
    return {"A_raw_r_did": dr27 - dr2,
            "B_fisher_z_did": dz27 - dz2,
            "C_tanh_per_size_did": tanh(dz27) - tanh(dz2),
            "D_tanh_of_z_did": tanh(dz27 - dz2)}


def main():
    inputs, ok = {}, True

    # --- registered arm, from the hash-pinned stageA blocks ---
    cells = {}
    for size in ("2b", "27b"):
        for tune in ("base", "it"):
            cell = "gemma-2-%s-%s" % (size, tune)
            p = OUT / cell / "stageA_summary.json"
            inputs[str(p.relative_to(REPO))] = sha(p)
            cells[(size, tune)] = json.load(open(p))["deterministic"]["arousal_r"]
    reg = constructions(cells[("2b", "base")], cells[("2b", "it")],
                        cells[("27b", "base")], cells[("27b", "it")])
    se4 = sqrt(4.0 / 42.0)
    zstat = reg["B_fisher_z_did"] / se4
    pval = two_sided(zstat)

    gp = OUT / "stageC_grading.json"
    inputs[str(gp.relative_to(REPO))] = sha(gp)
    rows = [t for t in json.load(open(gp))["tests"] if t.get("family") == "interaction"]
    if len(rows) != 1:
        sys.exit("FATAL: %d interaction rows" % len(rows))
    art = rows[0]

    which = [k for k, v in reg.items() if abs(v - art["delta"]) <= TOL]
    ok = ok and which == ["C_tanh_per_size_did"]
    ok = ok and abs(pval - art["p"]) <= 1e-12
    ok = ok and all(abs(v) < FLOOR and v > 0 for v in reg.values())

    # --- the five Appendix R.3 / Table E.4 arms ---
    np_ = REPO / "gemma_grid" / "geometry_out" / "norm_robustness_full_results.json"
    inputs[str(np_.relative_to(REPO))] = sha(np_)
    sp = REPO / "gemma_grid" / "scripts" / "norm_robustness_full.py"
    inputs[str(sp.relative_to(REPO))] = sha(sp)
    # Table E.4 delta column after the 2026-10-08 update to the registered form.
    E4_PRINTED = {"Russell-Mehrabian [reg]": 0.047, "NRC VAD [natural]": 0.072,
                  "NRC VAD [matched45]": 0.059, "Warriner [natural]": 0.052,
                  "Warriner [matched45]": 0.047}
    # What the column held before that update, retained so the switch stays visible.
    PRE_UPDATE_E4 = {"Russell-Mehrabian [reg]": 0.021, "NRC VAD [natural]": 0.095,
                     "NRC VAD [matched45]": 0.048, "Warriner [natural]": 0.054,
                     "Warriner [matched45]": 0.033}
    arms = []
    for label, v in json.load(open(np_)).items():
        per, n = v["per"], v["n"]
        c = constructions(per["gemma-2-2b-base"]["a_aro"], per["gemma-2-2b-it"]["a_aro"],
                          per["gemma-2-27b-base"]["a_aro"], per["gemma-2-27b-it"]["a_aro"])
        pn = two_sided(c["B_fisher_z_did"] / sqrt(4.0 / (n - 3)))
        printed = E4_PRINTED[label]
        matchA = abs(round(c["A_raw_r_did"], 3) - printed) < 1e-12
        matchC = abs(round(c["C_tanh_per_size_did"], 3) - printed) < 1e-12
        pre_is_C = abs(round(c["C_tanh_per_size_did"], 3) - PRE_UPDATE_E4[label]) < 1e-12
        ok = ok and matchA and not matchC and pre_is_C
        arms.append({"arm": label, "n": n, "p": pn, "E4_printed_delta": printed,
                     "E4_printed_is_A": matchA, "E4_printed_is_C": matchC,
                     "pre_update_delta": PRE_UPDATE_E4[label], "pre_update_was_C": pre_is_C, **c})

    rec = {
        "record": "four constructions of the registered arousal interaction DiD (SR.4 object-identity pin)",
        "ruling_2026_10_08": {
            "by": "Adam",
            "paper_reports": "A_raw_r_did (the preregistration's r-unit arithmetic)",
            "stored_artifact_delta_stays": "C_tanh_per_size_did (the r45 convention named at Appendix R.3)",
            "recomputation_implied": False,
            "verdict_effect": "none; p derives from B_fisher_z_did and NULL_subfloor holds under all four",
            "open_item": "whether Table E.4's five-arm delta column follows the headline to A or stays r45",
        },
        "prereg_formula": "([IT-base] arousal r)@27B - ([IT-base] arousal r)@2B  (PREREGISTRATION.md (e), r units)",
        "floor": FLOOR, "se_fisher_did": se4,
        "registered_arm": reg,
        "registered_arm_test": {"z_statistic": zstat, "p_two_sided": pval,
                                "se": se4, "statistic_built_from": "B_fisher_z_did"},
        "artifact_delta": art["delta"], "artifact_p": art["p"],
        "artifact_delta_is": which,
        "floor_ratios": dict((k, abs(v) / FLOOR) for k, v in reg.items()),
        "verdict_robustness": "every construction is positive and sub-floor; p is a function of B alone, "
                              "so NULL_subfloor and the registered direction hold under all four",
        "norm_arms": arms,
        "producer_note": "norm_robustness_full.py prints A as its DiD line; with Table E.4 switched to A "
                         "on 2026-10-08 the table is reproducible from its own cited script, delta column "
                         "included, which closes the SR.4 producer gap",
        "all_checks_pass": ok,
        "script": str(Path(__file__).resolve().relative_to(REPO)),
        "script_sha256": sha(Path(__file__).resolve()),
    }
    (OUT / "interaction_did_constructions.json").write_text(json.dumps(rec, indent=1))
    (OUT / "interaction_did_constructions_manifest.json").write_text(json.dumps(
        {"m9_note": "SR.5 input manifest -- sha256 of every input, written at run time.",
         "inputs": inputs,
         "output": "gemma_grid/geometry_out/interaction_did_constructions.json"}, indent=1))

    print("registered arm (Russell-Mehrabian, n=45):")
    for k in ("A_raw_r_did", "B_fisher_z_did", "C_tanh_per_size_did", "D_tanh_of_z_did"):
        print("  %-22s %+.17f   |x|/floor = %.4f  (1/%.2f)"
              % (k, reg[k], abs(reg[k]) / FLOOR, FLOOR / abs(reg[k])))
    print("  artifact delta         %+.17f  -> %s" % (art["delta"], which))
    print("  z = B/sqrt(4/42)       %.17f   p = %.17f  (artifact %.17f)"
          % (zstat, pval, art["p"]))
    print()
    print("Table E.4 arms: printed value matches A (post-2026-10-08); pre-update column was C")
    for a in arms:
        print("  %-28s n=%-4d printed %+.3f (was %+.3f) | A %+.4f  C %+.4f  p %.3f   A? %s  wasC? %s"
              % (a["arm"], a["n"], a["E4_printed_delta"], a["pre_update_delta"], a["A_raw_r_did"],
                 a["C_tanh_per_size_did"], a["p"], a["E4_printed_is_A"], a["pre_update_was_C"]))
    print()
    print("ALL CHECKS PASS" if ok else "CHECK FAILED -- see output")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
