#!/usr/bin/env python3
"""Assemble the six-cell Stage A cross-cell table from the per-cell
stageA_summary.json files (SR.4 serialization aggregator).

Basis-neutral: reads <tree>/<cell>/stageA_summary.json for the six cells and
writes <tree>/stageA_cross_cell_table.csv in the 13-column ledger schema
(the schema the tripwire and Request-B transcription consume). Pure stdlib.

Usage:
  python phase4_stageA_crosstable.py --tree gemma_grid/geometry_out
"""
import argparse, csv, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base",
         "gemma-2-9b-it", "gemma-2-27b-base", "gemma-2-27b-it"]

# CSV column -> (dotted key path into stageA_summary.json, precision)
#   precision: int decimals for floats; "int" -> integer; None -> string
COLS = [
    ("cell",          "cell",                             None),
    ("layer",         "layer",                            "int"),
    ("var_pc1",       "deterministic.var_pc1",            2),
    ("var_pc2",       "deterministic.var_pc2",            2),
    ("var_pc3",       "deterministic.var_pc3",            2),
    ("valence_r",     "deterministic.valence_r",          4),
    ("arousal_r",     "deterministic.arousal_r",          4),
    ("cohesion",      "cohesion.mean_purity_top10",       4),
    ("mutual_pairs",  "cohesion.mutual_top1_pairs",       "int"),
    ("self_ari_mean", "kmeans_seed_sweep.self_ari_mean",  4),
    ("self_ari_std",  "kmeans_seed_sweep.self_ari_std",   4),
    ("loving_pc1",    "deterministic.loving_pc1",         2),
    ("loving_pc2",    "deterministic.loving_pc2",         2),
]
HEADER = [c for c, _, _ in COLS]


def dig(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def fmt(v, prec):
    if prec is None:
        return str(v)
    if prec == "int":
        return str(int(round(float(v))))
    return str(round(float(v), prec))


def rows_for(tree):
    out = []
    for c in CELLS:
        p = REPO / tree / c / "stageA_summary.json"
        if not p.exists():
            sys.exit(f"FATAL: missing per-cell summary: {p}")
        d = json.load(open(p))
        out.append([fmt(dig(d, path), prec) for _, path, prec in COLS])
    return out


def write_table(tree, out_path=None):
    rows = rows_for(tree)
    out_path = Path(out_path) if out_path else (REPO / tree / "stageA_cross_cell_table.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return out_path, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", help="geometry_out tree to assemble (e.g. gemma_grid/geometry_out)")
    ap.add_argument("--out", help="explicit output path (default <tree>/stageA_cross_cell_table.csv)")
    a = ap.parse_args()
    if not a.tree:
        sys.exit("FATAL: --tree required")
    out, rows = write_table(a.tree, a.out)
    print(f"wrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
