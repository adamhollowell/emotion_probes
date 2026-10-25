#!/usr/bin/env python3
"""
Released extract of the dependence-aware bootstrap of the tuning contrasts (paper section 7.1, last
paragraph).

Source: geometry_out/tuning_dependence_aware_ci.json, written by
scripts/tuning_dependence_aware_ci.py (both audited 2026-10-14). The analysis is
unregistered. It corroborates and grades nothing.

The extract carries 17 rows, each as the point estimate and the 95 percent percentile interval only:
the twelve per-size tuning deltas (four signals at 2B, 9B, 27B), the four pooled deltas, and the
raw-r arousal interaction. It leaves out the bootstrap standard errors, the 90 percent intervals,
the comparison of standard errors with the registered ones, and the second form of the interaction.
On three rows (2B PC2 with pleasure, pooled PC2 with pleasure, and the interaction) the per-draw
re-signing of each axis, which flips a sign in 10 of 2,000 draws, inflates the bootstrap standard
error, while the percentile interval still describes the row faithfully. The re-signing rule was
stated before the run in the specification for this analysis and is not part of the registration.

The extract commits by sha256 to the full output, its manifest, and the producer script.

Usage:
  python tuning_dependence_aware_ci_extract.py            # build, check against the paper, compare with the stored extract
  python tuning_dependence_aware_ci_extract.py --write    # build, check against the paper, and write the extract
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
G = REPO / "gemma_grid"
FULL = G / "geometry_out" / "tuning_dependence_aware_ci.json"
MANIFEST = G / "geometry_out" / "tuning_dependence_aware_ci_manifest.json"
PRODUCER = G / "scripts" / "tuning_dependence_aware_ci.py"
OUT = G / "geometry_out" / "tuning_dependence_aware_ci_extract.json"

SIGNALS = ["valence_r", "arousal_r", "offdiag_pc1_arousal", "offdiag_pc2_pleasure"]
ROWS = [f"{s}|{z}" for s in SIGNALS for z in ("2b", "9b", "27b", "pooled")] + ["interaction_arousal_r_raw"]

# Section 7.1 as installed in v7 (sha1 03b92c66, unchanged through 6e6bb8e1), four decimals: point, interval low, interval high.
PAPER = {
    "valence_r|pooled": (+0.0558, +0.0303, +0.0910),
    "arousal_r|pooled": (-0.0829, -0.1290, -0.0446),
    "offdiag_pc1_arousal|pooled": (+0.0409, +0.0139, +0.0690),
    "offdiag_pc2_pleasure|pooled": (-0.0881, -0.1390, -0.0351),
    "interaction_arousal_r_raw": (+0.0470, -0.0051, +0.0988),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    full = json.loads(FULL.read_text())
    q = full["quantities"]
    assert len(ROWS) == 17 and all(r in q for r in ROWS), [r for r in ROWS if r not in q]
    rows = {r: {"point_r": q[r]["point_r"], "ci95_percentile": q[r]["ci95_pct"]} for r in ROWS}
    return {
        "record": "Dependence-aware bootstrap of the tuning contrasts, released extract. Unregistered. Corroborates, grades nothing.",
        "quantities": "tuning delta = tuned minus base, in raw correlation units; pooled = mean of the three per-size deltas; interaction_arousal_r_raw = (tuned minus base) at 27B minus (tuned minus base) at 2B on arousal_r",
        "method": {
            "B": full["B"],
            "seed": full["seed"],
            "resampling": "in each draw, 45 indices drawn with replacement from the 45 rated emotions, the same indices for all six models",
            "pca": "each model's principal components fit once on all 171 emotions and held fixed across draws",
            "re_signing": "in each draw, each axis is re-signed to agree with its own human rating on the resampled emotions; stated in the specification before the run, not part of the registration",
            "interval": "percentile, 95 percent",
            "floors": "no interval is compared to any floor",
            "ratings": "Russell and Mehrabian (1977), 45-emotion overlap",
        },
        "omitted": "bootstrap standard errors, 90 percent intervals, comparisons with the registered standard errors, and the second form of the interaction; see the README",
        "rows": rows,
        "commitment_sha256": {
            "tuning_dependence_aware_ci.json": sha(FULL),
            "tuning_dependence_aware_ci_manifest.json": sha(MANIFEST),
            "tuning_dependence_aware_ci.py": sha(PRODUCER),
        },
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def check_paper(ex: dict) -> list:
    bad = []
    for r, (p, lo, hi) in PAPER.items():
        got = ex["rows"][r]
        vals = (round(got["point_r"], 4), round(got["ci95_percentile"][0], 4), round(got["ci95_percentile"][1], 4))
        if vals != (p, lo, hi):
            bad.append((r, vals, (p, lo, hi)))
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    ex = build()
    bad = check_paper(ex)
    if bad:
        sys.exit(f"paper check FAILED: {bad}")
    print("paper check: all five section 7.1 rows reproduced at four decimals")
    if a.write:
        OUT.write_text(json.dumps(ex, indent=1))
        print("wrote", OUT)
    elif OUT.exists():
        old = json.loads(OUT.read_text())
        same = all(old[k] == ex[k] for k in ex if k != "generated_utc")
        print("stored extract:", "identical apart from generated_utc" if same else "DIFFERS")
        sys.exit(0 if same else 1)
