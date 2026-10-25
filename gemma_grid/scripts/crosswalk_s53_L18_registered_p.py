#!/usr/bin/env python3
"""SR.4 crosswalk for the two derived p values at GGE1 v7 section 5.3.

The draft states, of the L18 sensitivity read: "the registered test is not
significant in either chain (p = .92 and .45)".  Those two p values are the
draft's own arithmetic.  They appear in no artifact.  This script derives them
from serialized inputs so that the claim is re-derivable, per SR.4, and hashes
every input at run time, per SR.5.

Inputs
  geometry_out/job1_L18_emotion_bootstrap_sensitivity.json
    results.<chain>.arousal_rise.delta_z  the Fisher-z 2B-to-27B arousal rise
    with the 2B cells read at the pre-registered sensitivity layer L18.

Constant
  The registered single-pair Fisher standard error for a difference of two
  independent z values on the 45-emotion Russell and Mehrabian overlap:
      SE = sqrt(2 / (45 - 3)) = sqrt(2/42) = 0.21821789023599236
  This is the same SE the registered tests use throughout the study.

Statistic
  z_stat = delta_z / SE;  two-sided normal p = erfc(|z_stat| / sqrt(2)).
  Identical in form to the artifact's own p_dep_aware, which substitutes the
  bootstrap SE for the registered one.

This script derives a p from a registered test.  It re-grades nothing (SR.7).
"""
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "geometry_out",
                   "job1_L18_emotion_bootstrap_sensitivity.json")
OUT = os.path.join(ROOT, "geometry_out",
                   "crosswalk_s53_L18_registered_p.json")

N_RATED = 45
SE = math.sqrt(2.0 / (N_RATED - 3))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def two_sided_p(delta_z, se):
    z = delta_z / se
    return z, math.erfc(abs(z) / math.sqrt(2.0))


def main():
    with open(ART) as fh:
        art = json.load(fh)

    out = {
        "record": ("SR.4 crosswalk for the two derived p values at v7 section 5.3, "
                   "the L18 registered-test p values quoted as .92 and .45."),
        "LABEL": ("DERIVATION of a REGISTERED test statistic from a serialized "
                  "delta_z. No new measurement. Re-grades nothing (SR.7)."),
        "procedure": ("z_stat = delta_z / SE_registered, two-sided normal "
                      "p = erfc(|z_stat|/sqrt(2)). SE_registered = "
                      "sqrt(2/(45-3)), the single-pair Fisher SE on the "
                      "45-emotion Russell and Mehrabian overlap."),
        "se_registered": SE,
        "se_registered_formula": "sqrt(2/(45-3)) on the 45-emotion rated overlap",
        "chains": {},
        "basis": "100T. Cleaned analysis-layer vectors, 2B read at L18, 27B at L31.",
        "sr5_input_manifest": {
            os.path.relpath(ART, ROOT): sha256(ART),
        },
        "draft_claim_checked": ("v7 section 5.3: \"Neither rise clears the "
                                "registered floor of .15, and the registered test "
                                "is not significant in either chain (p = .92 and .45).\""),
    }

    for chain in ("base", "tuned"):
        delta_z = art["results"][chain]["arousal_rise"]["delta_z"]
        delta_r = art["results"][chain]["arousal_rise"]["delta_r"]
        z_stat, p = two_sided_p(delta_z, SE)
        out["chains"][chain] = {
            "delta_z_from_artifact": delta_z,
            "delta_r_from_artifact": delta_r,
            "z_stat": z_stat,
            "p_two_sided_registered_SE": p,
            "p_rounded_2dp": round(p, 2),
            "clears_registered_floor_0.15_in_r_units": abs(delta_r) >= 0.15,
        }

    # Independent corroboration of the base chain's delta_z from the component
    # correlations, which the artifact serializes only for 2b-base inside its
    # sign-reversal flag. The tuned chain's components are not serialized, so
    # the same check cannot be run there. Recorded as a serialization gap.
    r_2b_base_L18 = 0.6557977
    r_27b_base = r_2b_base_L18 + art["results"]["base"]["arousal_rise"]["delta_r"]
    dz_check = math.atanh(r_27b_base) - math.atanh(r_2b_base_L18)
    out["base_chain_delta_z_corroboration"] = {
        "r_2b_base_L18_from_artifact_flag": r_2b_base_L18,
        "r_27b_base_implied": r_27b_base,
        "delta_z_recomputed": dz_check,
        "delta_z_from_artifact": art["results"]["base"]["arousal_rise"]["delta_z"],
        "abs_difference": abs(dz_check - art["results"]["base"]["arousal_rise"]["delta_z"]),
        "note": ("Agreement is bounded by the 7-figure precision at which the "
                 "artifact's flag states r_2b_base. The tuned chain's component "
                 "correlations are not serialized anywhere, so its delta_z rests "
                 "on the artifact's own reproduction gate alone."),
    }

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
        fh.write("\n")

    for chain in ("base", "tuned"):
        c = out["chains"][chain]
        print("%-6s delta_z %+.17f  z %+.8f  p %.8f  -> %.2f" % (
            chain, c["delta_z_from_artifact"], c["z_stat"],
            c["p_two_sided_registered_SE"], c["p_rounded_2dp"]))
    cb = out["base_chain_delta_z_corroboration"]
    print("base delta_z corroboration: recomputed %.10f vs artifact %.10f, |diff| %.2e"
          % (cb["delta_z_recomputed"], cb["delta_z_from_artifact"], cb["abs_difference"]))
    print("SE_registered %.17f" % SE)
    print("wrote", os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
