#!/usr/bin/env python3
"""
Build the Figure-1 render data from the canonical sweep and the window selection.

Run from the project root:
    python3 scripts/build_fig1_render_data.py

Inputs (all resolved relative to the current working directory = project root):
  data/heldout/maxact_v3/fig1_window_selection.json
        the displayed-window selection: per emotion panel, the chosen ranks with
        doc_id + peak_pos + source. This is the only curation input.
  data/heldout/maxact_v3/pass1_results.json
        top-20 (doc_id, pos, value) per emotion; run config; vectors_sha256.
        The by_emotion key order is the projection column order.
  data/heldout/maxact_v3/histograms.npz
        per-emotion activation counts over all swept tokens (171 x bins);
        the 90th-percentile shading floor is derived from these exactly.
  data/heldout/maxact_v3/pertoken/<doc_id>.npz
        per-token arrays for every winning document:
          tokens[N]   object array of token strings (leading spaces preserved)
          proj[N,171] L31 projection of each token onto the 171 vectors (float16)

Output:
  data/heldout/maxact_v3/fig1_render_data.json
        self-contained per-panel render data (tokens, per-token values, per-panel
        p90_floor + top1_value) in the schema render_parallel_fig1_maxact.py consumes.

Shading anchors (both derived from THIS sweep, no cross-sweep reuse):
  top1_value = the emotion's rank-1 peak value (pass1_results by_emotion[emo][0]).
  p90_floor  = the emotion's 90th-percentile activation over all swept tokens,
               read from histograms.npz. Shading rule: token shaded iff
               value > p90_floor, intensity = (value-p90_floor)/(top1_value-p90_floor).

Window convention (per row):
  tokens = doc_tokens[peak_pos - LEAD : peak_pos + SPAN]
  values = proj[:, emotion_col] over the same slice
  Uniform LEAD/SPAN for every row; no per-row overrides.

The emotion -> projection column mapping is verified against the stored rank-1
value (proj[peak_pos, col] must equal it); the script aborts on any mismatch,
and it aborts if any selected (rank, doc_id, peak_pos) disagrees with the sweep.
Input sha256s and the vectors hash are recorded in the output for provenance.
"""
from __future__ import annotations

# --- basis gate: refuse to run unless the vectors match the expected sha256.
# Identify the vectors by their hash, not the filename `emotion_vectors_171.pt`. ---
import hashlib as _hlg, pathlib as _plg
_VECG = _plg.Path("data/vectors/emotion_vectors_171.pt")
_EXPG = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
if _VECG.exists():
    _hg = _hlg.sha256(_VECG.read_bytes()).hexdigest()
    assert _hg == _EXPG, (
        "%s basis mismatch: %s != expected canonical %s -- refusing non-canonical basis"
        % (_VECG, _hg[:12], _EXPG[:12]))
# --- end basis gate ---

import hashlib
import json
import os
import sys
import time

import numpy as np

# ---- window geometry (uniform, no per-row overrides) ----------------------
LEAD = 13
SPAN = 60
# ---------------------------------------------------------------------------

ROOT = os.getcwd()
SWEEP_DIR = os.path.join(ROOT, "data", "heldout", "maxact_v3")
SEL_PATH = os.path.join(SWEEP_DIR, "fig1_window_selection.json")
PASS1_PATH = os.path.join(SWEEP_DIR, "pass1_results.json")
HIST_PATH = os.path.join(SWEEP_DIR, "histograms.npz")
PT_DIR = os.path.join(SWEEP_DIR, "pertoken")
VEC_PATH = os.path.join(ROOT, "data", "vectors", "emotion_vectors_171.pt")
OUT_PATH = os.path.join(SWEEP_DIR, "fig1_render_data.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantile_from_counts(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    """90th-percentile floor from histogram counts (same rule as
    extract_fig1_curated_data.py::quantile_from_counts)."""
    c = np.cumsum(counts)
    idx = int(np.searchsorted(c, q * c[-1]))
    return float(edges[min(idx, len(counts) - 1) + 1])


def main() -> None:
    for p in (SEL_PATH, PASS1_PATH, HIST_PATH, VEC_PATH):
        if not os.path.exists(p):
            sys.exit("FATAL: missing input %s" % p)

    sel = json.load(open(SEL_PATH))
    pass1 = json.load(open(PASS1_PATH))
    by_emotion = pass1["by_emotion"]
    emotions = list(by_emotion.keys())  # projection column order

    # Basis provenance must match the selection's declared basis.
    vectors_sha = sha256_file(VEC_PATH)
    if sel.get("basis_sha256") and sel["basis_sha256"] != vectors_sha:
        sys.exit("FATAL: emotion_vectors_171.pt sha256 %s != selection basis_sha256 %s"
                 % (vectors_sha, sel["basis_sha256"]))
    if pass1.get("vectors_sha256") and pass1["vectors_sha256"] != vectors_sha:
        sys.exit("FATAL: pass1_results vectors_sha256 %s != emotion_vectors_171.pt %s"
                 % (pass1["vectors_sha256"], vectors_sha))

    z = np.load(HIST_PATH, allow_pickle=True)
    counts = z["counts"]
    edges = np.linspace(float(z["lo"]), float(z["hi"]), int(z["bins"]) + 1)

    cache = {}

    def toks_proj(doc_id):
        if doc_id not in cache:
            zz = np.load(os.path.join(PT_DIR, doc_id + ".npz"), allow_pickle=True)
            cache[doc_id] = ([str(t) for t in zz["tokens"]], zz["proj"])
        return cache[doc_id]

    panels = {}
    used_shards = set()

    for panel in sel["panels"]:
        emo = panel["emotion"]
        if emo not in by_emotion:
            sys.exit("FATAL: emotion %r not in sweep" % emo)
        e_idx = emotions.index(emo)

        wins = by_emotion[emo]
        top1_value = float(wins[0]["value"])
        p90_floor = quantile_from_counts(counts[e_idx], edges, 0.90)

        rows = []
        for r in sorted(panel["rows"], key=lambda x: x["rank"]):
            rank = int(r["rank"])
            did = r["doc_id"]
            pp = int(r["peak_pos"])

            # selection must agree with the sweep at this rank
            w = wins[rank - 1]
            if w["doc_id"] != did or int(w["pos"]) != pp:
                sys.exit("FATAL: %s rank %d selection (%s,%d) != sweep (%s,%d)"
                         % (emo, rank, did, pp, w["doc_id"], int(w["pos"])))

            toks, proj = toks_proj(did)
            used_shards.add(did)

            # column-mapping check on every row
            got = float(proj[pp, e_idx])
            stored = float(w["value"])
            if abs(got - stored) > max(2.0, abs(stored) * 0.01):
                sys.exit("FATAL: column map %s rank %d: proj[%d,%d]=%.3f vs stored=%.3f"
                         % (emo, rank, pp, e_idx, got, stored))

            pj = proj[:, e_idx]
            a = max(0, pp - LEAD)
            b = min(len(toks), pp + SPAN)
            rows.append({
                "rank": rank,
                "doc_id": did,
                "source": r.get("source", w.get("source")),
                "peak_index_in_row": pp - a,
                "clipped_left": a > 0,
                "clipped_right": b < len(toks),
                "tokens": [str(t) for t in toks[a:b]],
                "values": [round(float(v), 3) for v in pj[a:b]],
            })

        panels[emo] = {
            "emotion": emo,
            "p90_floor": round(p90_floor, 3),
            "top1_value": round(top1_value, 3),
            "rows": rows,
        }

    out = {
        "figure": "Figure 1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": "build_fig1_render_data.py",
        "basis": os.path.relpath(VEC_PATH, ROOT),
        "basis_sha256": vectors_sha,
        "sweep": os.path.relpath(SWEEP_DIR, ROOT),
        "window": {"lead": LEAD, "span": SPAN},
        "shading_rule": (
            "per panel: token shaded iff value > p90_floor; "
            "intensity = (value-p90_floor)/(top1_value-p90_floor), clamped to [0,1]. "
            "top1_value = rank-1 peak; p90_floor = 90th percentile over all swept "
            "tokens for that emotion (histograms.npz)."
        ),
        "inputs": {
            "fig1_window_selection.json": sha256_file(SEL_PATH),
            "pass1_results.json": sha256_file(PASS1_PATH),
            "histograms.npz": sha256_file(HIST_PATH),
            "emotion_vectors_171.pt": vectors_sha,
            "n_pertoken_shards": len(used_shards),
        },
        "panels": panels,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    n_rows = sum(len(p["rows"]) for p in panels.values())
    print("wrote %s (%.2f MB) - %d panels, %d rows, %d shards"
          % (OUT_PATH, os.path.getsize(OUT_PATH) / 1e6, len(panels), n_rows, len(used_shards)))
    for emo in [p["emotion"] for p in sel["panels"]]:
        pp = panels[emo]
        print("  %-11s p90_floor=%8.1f  top1=%9.1f  rows=%d"
              % (emo, pp["p90_floor"], pp["top1_value"], len(pp["rows"])))


if __name__ == "__main__":
    main()
