#!/usr/bin/env python3
"""Assemble the six-cell lens cross-cell table from per-cell lens_summary.json
(SR.4 serialization aggregator; the lens sibling of phase4_stageA_crosstable.py).

Basis-neutral: reads <tree>/<cell>/lens_summary.json for the six cells and writes
<tree>/lens_cross_cell.json in the {onset, onset_frac, L0_median_rank,
final_median_rank} per-cell schema the ledger's C5 / P4.5 rows consume. Pure
stdlib; no basis contact (it only re-serializes numbers the lens step already
computed under the cell's own basis).

Usage:
  python phase4_lens_crosstable.py --tree gemma_grid/geometry_out
"""
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base",
         "gemma-2-9b-it", "gemma-2-27b-base", "gemma-2-27b-it"]
FIELDS = ("onset", "onset_frac", "L0_median_rank", "final_median_rank")


def row_for(summary):
    per = {int(e["layer"]): e["median_own_rank"] for e in summary["per_layer_median_own_rank"]}
    final_layer = max(per)
    return {
        "onset": summary["onset_layer_median_lt_10k"],
        "onset_frac": summary["onset_fractional_depth"],
        "L0_median_rank": per[0],
        "final_median_rank": per[final_layer],
    }


def table_for(tree):
    out = {}
    for c in CELLS:
        p = REPO / tree / c / "lens_summary.json"
        if not p.exists():
            sys.exit(f"FATAL: missing per-cell lens summary: {p}")
        out[c] = row_for(json.load(open(p)))
    return out


def write_table(tree, out_path=None):
    tbl = table_for(tree)
    out_path = Path(out_path) if out_path else (REPO / tree / "lens_cross_cell.json")
    out_path.write_text(json.dumps(tbl, indent=2))
    return out_path, tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", help="geometry_out tree to assemble (e.g. gemma_grid/geometry_out)")
    ap.add_argument("--out", help="explicit output path (default <tree>/lens_cross_cell.json)")
    a = ap.parse_args()
    if not a.tree:
        sys.exit("FATAL: --tree required")
    out, tbl = write_table(a.tree, a.out)
    print(f"wrote {out} ({len(tbl)} cells)")


if __name__ == "__main__":
    main()
