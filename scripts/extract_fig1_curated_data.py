#!/usr/bin/env python3
"""Build the single portable render file for the CURATED (redacted) Figure 1.

Reads fig1_selection.json (the resolved curation record) plus the per-token
arrays and histograms, and writes:

    data/heldout/maxact/fig1_render_data_v2_CURATED.json

Self-contained: panels, rows, tokens, per-token values, peak index, redaction
spans, thresholds, layout, and shading rules -- everything an external
renderer needs, with NO other files required.

SAFETY PROPERTY: the token text of every redacted span is REPLACED in this
file by full-block characters of matching length. The explicit text does not
leave the project; an external tool receives correct widths and values only.

Row policy: usable rows in rank order, one row per document; first 7 are
display rows, the 8th is a flagged reserve.

Run from project root: python3 scripts/extract_fig1_curated_data.py
"""
import hashlib
import json
import tarfile
import time
from pathlib import Path

import numpy as np

BASE = Path("data/heldout/maxact")
PANEL = ["afraid", "angry", "calm", "desperate", "guilty", "happy",
         "inspired", "loving", "nervous", "proud", "sad", "surprised"]
CTX = 30
N_DISPLAY, N_RESERVE = 7, 1


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantile_from_counts(counts, edges, q):
    c = np.cumsum(counts)
    idx = int(np.searchsorted(c, q * c[-1]))
    return float(edges[min(idx, len(counts) - 1) + 1])


def block_out(tok):
    """Replace a token's visible characters with full blocks, preserving
    leading whitespace so row spacing survives."""
    lead = len(tok) - len(tok.lstrip())
    return tok[:lead] + "\u2588" * max(1, len(tok) - lead)


def main():
    sel = json.loads((BASE / "fig1_selection.json").read_text())
    r = json.loads((BASE / "pass1_results.json").read_text())
    emotions = list(r["by_emotion"].keys())

    z = np.load(BASE / "histograms.npz", allow_pickle=True)
    all_counts = z["counts"]
    edges = np.linspace(float(z["lo"]), float(z["hi"]), int(z["bins"]) + 1)

    pt_dir = BASE / "pertoken"
    tar = tarfile.open(BASE / "pertoken.tar") if (BASE / "pertoken.tar").exists() else None

    def load_npz(doc_id):
        p = pt_dir / (doc_id + ".npz")
        if p.exists():
            return np.load(p, allow_pickle=True)
        return np.load(tar.extractfile("pertoken/%s.npz" % doc_id), allow_pickle=True)

    panels = {}
    for emo in PANEL:
        e_idx = emotions.index(emo)
        p90 = quantile_from_counts(all_counts[e_idx], edges, 0.90)
        wins = r["by_emotion"][emo]

        rows, seen = [], set()
        for sr in sorted(sel["panels"][emo], key=lambda x: x["rank"]):
            if sr["doc_id"] in seen:
                continue
            seen.add(sr["doc_id"])
            w = wins[sr["rank"] - 1]
            zz = load_npz(sr["doc_id"])
            toks = [str(t) for t in zz["tokens"]]
            proj = zz["proj"][:, e_idx].astype(np.float32)
            pos, seq = w["pos"], len(toks)
            lo, hi = max(1, pos - CTX), min(seq, pos + CTX + 1)
            win = list(toks[lo:hi])
            redacted_idx = set()
            for a, b in sr["redact_token_spans"]:
                redacted_idx.update(range(a, b + 1))
            display_tokens = [block_out(t) if k in redacted_idx else t
                              for k, t in enumerate(win)]
            rows.append({
                "rank": sr["rank"], "mark": sr["mark"], "doc_id": sr["doc_id"],
                "source": sr["source"], "peak_value": round(w["value"], 1),
                "peak_index_in_row": pos - lo,
                "clipped_left": lo > 1, "clipped_right": hi < seq,
                "tokens": display_tokens,
                "values": [round(float(v), 1) for v in proj[lo:hi]],
                "redacted_token_indices": sorted(redacted_idx),
            })
            if len(rows) == N_DISPLAY + N_RESERVE:
                break
        for i, row in enumerate(rows):
            row["display"] = i < N_DISPLAY
            row["reserve"] = i >= N_DISPLAY
        panels[emo] = {"p90_floor": p90, "top1_value": rows[0]["peak_value"],
                       "rows": rows}

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_kind": "CURATED REDACTED FIGURE -- manual safety selection; NOT the faithful replication",
        "inputs": {"fig1_selection_sha256": sha256_file(BASE / "fig1_selection.json"),
                   "pass1_results_sha256": sha256_file(BASE / "pass1_results.json"),
                   "histograms_sha256": sha256_file(BASE / "histograms.npz")},
        "safety_property": "redacted spans' token text replaced by U+2588 blocks of matching length; values and peak positions preserved",
        "layout": {"grid": "2 columns x 6 rows of panels, alphabetical", "rows_per_panel": N_DISPLAY,
                   "reserve_rows": N_RESERVE, "panel_order": PANEL},
        "render_rules": {
            "shading": "highlight token i iff values[i] > p90_floor; alpha = clamp((values[i]-p90_floor)/(top1_value-p90_floor),0,1); per-panel scale; bubble behind text",
            "redaction": "tokens listed in redacted_token_indices render as solid blocks (text already replaced) WITH their activation bubble drawn at normal alpha beneath/around the block",
            "text": "concatenate tokens verbatim, no added spaces; newlines inside tokens render as a pilcrow, never a line break; ellipsis at clipped edges; monospace or near-monospace",
            "no_edits": "never alter, translate, reflow, or re-rank rows; rows are rank-ordered top-activating windows, one document each",
        },
        "caption_note": "Rows are each emotion's strongest-activating held-out windows, one per document, with passages depicting sexual or violent content redacted; activation shading is unaltered by redaction.",
        "panels": panels,
    }
    path = BASE / "fig1_render_data_v2_CURATED.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote %s (%.2f MB)" % (path, path.stat().st_size / 1e6))
    for emo in PANEL:
        rows = panels[emo]["rows"]
        nred = sum(1 for x in rows if x["redacted_token_indices"] and x["display"])
        print("  %-11s rows %d (display %d + reserve %d) | display rows with redactions: %d"
              % (emo, len(rows), sum(r_["display"] for r_ in rows),
                 sum(r_["reserve"] for r_ in rows), nred))


if __name__ == "__main__":
    main()
