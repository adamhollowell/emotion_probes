#!/usr/bin/env python3
"""
G6.1 vocabulary-readout grading: the released extract (paper section 6.1).

Builds geometry_out/g6_1_regrade/G6_1_verdicts.json from the blind re-grade of 2026-08-19.
The grading was run as one basis-blind pass over 144 interleaved rows under the rubric fixed before the
data existed (GGE1_100T_PRECOMMITMENTS.md item 4: a token is emotion-related per the rubric, a row is
on-theme if at least two of its five tokens are). This extract carries the 72 rows the paper reports,
each with its cell, emotion, five tokens, per-token verdicts, and on-theme verdict,
plus the per-emotion and per-cell counts the paper prints. It is a subset of a larger blind run whose
remainder is retained and not released.

The extract commits to the full run by hash: the sha256 of the complete sealed label key, the complete
grader verdicts file, the grading packet, the regrade manifest, the unblind result, and the transcription
inputs, so the commitment's scope is the whole retained set. Anyone later holding those files
can check that these 72 rows are the rows that were graded and that nothing was reselected. The six
lens_summary.json inputs the graded tokens came from are listed with their hashes.

Verify mode recomputes the extract and checks the counts against the paper's section 6.1 figures.

Usage:
  python g6_1_regrade_extract.py            # recompute, check against the paper's counts and the stored extract
  python g6_1_regrade_extract.py --write    # recompute and write the extract
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "gemma_grid" / "geometry_out" / "g6_1_regrade"
KEY, VERD, PACKET, MANIFEST = D / "G6_1_label_key_SEALED.json", D / "G6_1_grader_verdicts.json", D / "G6_1_grading_packet.json", D / "G6_1_regrade_manifest.json"
UNBLIND, TRANSCRIPTION = D / "G6_1_unblind_result.json", D / "G6_1_transcription_inputs.json"
OUT = D / "G6_1_verdicts.json"
CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base", "gemma-2-9b-it", "gemma-2-27b-base", "gemma-2-27b-it"]

# Section 6.1 of the paper: cells on-theme out of six, per emotion.
PAPER_COUNTS = {"surprised": 6, "happy": 6, "desperate": 6, "proud": 6,
                "afraid": 5, "loving": 5, "nervous": 5, "angry": 5, "inspired": 5,
                "sad": 4, "guilty": 2, "calm": 1}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    key = json.loads(KEY.read_text())
    verd = json.loads(VERD.read_text())
    manifest = json.loads(MANIFEST.read_text())
    basis = {r["row_id"]: r for r in key["key"]}
    rows = []
    for r in verd["rows"]:
        k = basis[r["row_id"]]
        assert (k["cell"], k["emotion"]) == (r["cell"], r["emotion"]), r["row_id"]
        if k["basis"] != "100T":
            continue
        rows.append({"cell": r["cell"], "emotion": r["emotion"], "tokens": r["tokens"],
                     "related_count": r["related_count"], "on_theme": r["on_theme"]})
    rows.sort(key=lambda r: (CELLS.index(r["cell"]), r["emotion"]))
    assert len(rows) == 72, len(rows)
    per_emotion = Counter(r["emotion"] for r in rows if r["on_theme"])
    per_cell = Counter(r["cell"] for r in rows if r["on_theme"])
    emotions = sorted({r["emotion"] for r in rows})
    inputs = {k: v for k, v in manifest["inputs_sha256"].items() if "/geometry_out" + "_100T/" in k}  # the run's manifest keys carry its working folder name
    assert len(inputs) == 6, inputs
    return {
        "record": "G6.1 vocabulary-readout grading, released extract: the 72 rows the paper reports, with verdicts and the counts the paper prints.",
        "grading": {
            "protocol": "GGE1_100T_PRECOMMITMENTS.md item 4. Rubric fixed before the data existed.",
            "rubric": verd["rubric"],
            "grade_unit": "per row: each of the 5 tokens marked emotion-related or not under the rubric; row on-theme iff at least 2 of 5 related",
            "run": "one basis-blind pass over 144 interleaved rows in randomized order, graded by a fresh-context session with the basis withheld. This extract is a subset of that run. The remainder is retained and not released.",
            "rubric_validation": "the rubric was validated against an earlier count of the same anchor cell, made under a different rubric, and reproduced it exactly",
        },
        "n_rows": len(rows),
        "emotions": emotions,
        "cells": CELLS,
        "rows": rows,
        "cells_on_theme_out_of_6_per_emotion": {e: per_emotion.get(e, 0) for e in emotions},
        "emotions_on_theme_out_of_12_per_cell": {c: per_cell.get(c, 0) for c in CELLS},
        "inputs_sha256": inputs,
        "full_run_commitment_sha256": {
            "G6_1_label_key_SEALED.json": sha(KEY),
            "G6_1_grader_verdicts.json": sha(VERD),
            "G6_1_grading_packet.json": sha(PACKET),
            "G6_1_regrade_manifest.json": sha(MANIFEST),
            "G6_1_unblind_result.json": sha(UNBLIND),
            "G6_1_transcription_inputs.json": sha(TRANSCRIPTION),
        },
        "blinding_limit": "The shuffle seed sits in the regrade manifest as well as in the sealed key, and the presentation order is reproducible from it, so access to the manifest would permit recovering the order. Whether the grader could see the manifest cannot be tested from the artifacts. This is a disclosed limit of the blinding, not a defect in the counts.",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    # release: the grading inputs are withheld by design, so stop with a message, not a missing-file error
    missing = [p.name for p in (KEY, VERD, PACKET, MANIFEST, UNBLIND, TRANSCRIPTION) if not p.exists()]
    if missing:
        sys.exit("The G6.1 grading inputs are withheld by design (" + ", ".join(missing) + "). "
                 "The sealed label key and its companion files cover the full blind run of 144 rows, of which "
                 "the 72 the paper reports are released, so this script cannot rebuild the extract from the release. "
                 "geometry_out/g6_1_regrade/G6_1_verdicts.json ships as the record of the grading and "
                 "commits to those files by their sha256.")
    new = build()
    got = new["cells_on_theme_out_of_6_per_emotion"]
    bad = {e: (got.get(e), n) for e, n in PAPER_COUNTS.items() if got.get(e) != n}
    extra = sorted(set(got) - set(PAPER_COUNTS))
    print("per-emotion counts:", got)
    print("per-cell counts:", new["emotions_on_theme_out_of_12_per_cell"])
    if bad or extra:
        sys.exit(f"paper check FAILED: mismatches {bad}, emotions not in paper list {extra}")
    print("paper check: all twelve section 6.1 counts reproduced")
    if a.write:
        OUT.write_text(json.dumps(new, indent=1, ensure_ascii=False))
        print("wrote", OUT)
    elif OUT.exists():
        old = json.loads(OUT.read_text())
        same = all(old[k] == new[k] for k in new if k != "generated_utc")
        print("stored extract:", "identical apart from generated_utc" if same else "DIFFERS")
        sys.exit(0 if same else 1)
