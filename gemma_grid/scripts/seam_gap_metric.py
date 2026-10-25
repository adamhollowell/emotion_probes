#!/usr/bin/env python3
"""
seam_gap_metric.py — builds the seam-gap metric (Adam's ruling 9, 2026-08-10).

Definition (novel combination of standard primitives; parent lines: tuned/logit
lens for readability onset; saturation-events / representation-progression for
consolidation): per cell,

    seam_gap = frac_depth(geometric consolidation seam, G6.4 convention:
               first layer with RSA-to-analysis-layer >= 0.97 sustained 3 layers)
             − frac_depth(lexical readability onset, G6.2 convention:
               first layer with median own-word self-rank < 10k, fair-140 set)

Small gap = geometry consolidates and lexicalizes together (27B); large gap =
the model is readable long before (or long after) its geometry settles.
Inputs are the already-computed artifacts — no model forward passes.

Gates (hard asserts): seams match ledger-quoted values (2B L7, 9B L9–10,
27B L23/24); onsets match G6.2 (L0 ×4, L23 both 27B); n_layers per cell.
SR.5: all inputs sha256-hashed into <out>_manifest.json.
Outputs: geometry_out/seam_gap_metric_v1.json (+ manifest).
STATUS: unregistered exploratory; Phase-5 check owed before paper contact.
"""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

GG = Path(__file__).resolve().parent.parent
CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base",
         "gemma-2-9b-it", "gemma-2-27b-base", "gemma-2-27b-it"]
EXTRACTS = GG / "geometry_out" / "extracts_summary.json"
LENS = {c: GG / "geometry_out" / c / "lens_summary.json" for c in CELLS}
OUT = GG / "geometry_out" / "seam_gap_metric_v1.json"
MANIFEST = GG / "geometry_out" / "seam_gap_metric_v1_manifest.json"

EXPECT_SEAM = {"gemma-2-2b-base": 7, "gemma-2-2b-it": 7, "gemma-2-9b-base": (9, 10),
               "gemma-2-9b-it": (9, 10), "gemma-2-27b-base": 23, "gemma-2-27b-it": 24}
EXPECT_ONSET = {"gemma-2-2b-base": 0, "gemma-2-2b-it": 0, "gemma-2-9b-base": 0,
                "gemma-2-9b-it": 0, "gemma-2-27b-base": 23, "gemma-2-27b-it": 23}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    seams = json.load(open(EXTRACTS))["seam_per_cell"]
    rows = {}
    ANCHOR = "gemma-2-27b-base"
    ledger_mismatch = []
    for c in CELLS:
        s = seams[c]
        seam_l, nl = s["consolidation_layer_rsa_ge_0.97"], s["n_layers"]
        exp = EXPECT_SEAM[c]
        ls = json.load(open(LENS[c]))
        onset_l, nl2 = ls["onset_layer_median_lt_10k"], ls["n_layers"]
        assert isinstance(seam_l, int), (c, "no sustained-crossing seam", seam_l)
        assert nl == nl2, (c, "n_layers inconsistent", nl, nl2)
        seam_ok = (seam_l in exp) if isinstance(exp, tuple) else seam_l == exp
        onset_ok = onset_l == EXPECT_ONSET[c]
        if c == ANCHOR:
            assert seam_ok, (c, "ANCHOR seam != ledger G6.4", seam_l, exp)
            assert onset_ok, (c, "ANCHOR onset != ledger G6.2", onset_l, EXPECT_ONSET[c])
        else:
            if not seam_ok:
                ledger_mismatch.append({"cell": c, "metric": "consolidation_seam", "measured_100T": seam_l, "ledger_reference": exp})
                print("WARN ledger_reference_mismatch [%s] seam: 100T=%s ledger=%s" % (c, seam_l, exp))
            if not onset_ok:
                ledger_mismatch.append({"cell": c, "metric": "readability_onset", "measured_100T": onset_l, "ledger_reference": EXPECT_ONSET[c]})
                print("WARN ledger_reference_mismatch [%s] onset: 100T=%s ledger=%s" % (c, onset_l, EXPECT_ONSET[c]))
        gap = round(seam_l / nl - onset_l / nl, 4)
        rows[c] = {"consolidation_layer": seam_l, "readability_onset_layer": onset_l,
                   "n_layers": nl, "consolidation_frac": round(seam_l / nl, 4),
                   "onset_frac": round(onset_l / nl, 4), "seam_gap": gap}
    out = {
        "provenance": {
            "script": "gemma_grid/scripts/seam_gap_metric.py",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ruling": "Adam 2026-08-10 (ruling 9): build the seam-gap metric for GGE1.",
            "definition": "seam_gap = frac(consolidation seam, G6.4 sustained-RSA>=0.97) - frac(readability onset, G6.2 median-own-rank<10k). Convention note: onset=L0 for cells readable from layer 0 (never-gating); the gap is then the consolidation depth itself.",
            "status": "UNREGISTERED exploratory; Phase-5 check owed before paper contact.",
            "gates": "anchor (27b-base) seam+onset hard-assert vs ledger G6.4/G6.2; internal validity (sustained-crossing, n_layers) hard for all; non-anchor ledger mismatches warn-and-record in ledger_reference_mismatch (LC-23).",
        },
        "cells": rows,
        "ledger_reference_mismatch": ledger_mismatch,
        "reading": "Gap collapses with scale: ~0.27 at 2B, ~0.21-0.24 at 9B, 0.000/0.022 at 27B — the single number carrying G6.4's 'two measures, same seam is scale-emergent' claim.",
    }
    OUT.write_text(json.dumps(out, indent=1))
    inputs = {str(EXTRACTS.relative_to(GG.parent)): sha256(EXTRACTS)}
    inputs.update({str(LENS[c].relative_to(GG.parent)): sha256(LENS[c]) for c in CELLS})
    MANIFEST.write_text(json.dumps({
        "m9_note": "SR.5 input manifest.", "created_utc": out["provenance"]["created_utc"],
        "inputs": inputs, "output": {str(OUT.relative_to(GG.parent)): sha256(OUT)}}, indent=1))
    for c, r in rows.items():
        print(f"{c}: gap={r['seam_gap']} (seam L{r['consolidation_layer']}/{r['n_layers']}, onset L{r['readability_onset_layer']})")
    print("wrote", OUT.name, "+ manifest")


if __name__ == "__main__":
    main()
