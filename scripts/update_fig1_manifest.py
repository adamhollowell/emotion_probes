#!/usr/bin/env python3
"""
Refresh the Figure 1 (canonical max-activation panels) entry in
docs/e1_figures/FIGURES_MANIFEST.json.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/update_fig1_manifest.py

fig1 is PIL-based and consumes the maxact sweep's per-token projections, so it
differs from figs 2-7: it cannot hash a vectors .pt directly. This updater verifies the
recorded provenance of the whole chain (the sweep's pass1_results.json AND the render
data both stamp vectors_sha256 == the canonical basis 6de2e293), re-renders to a temp prefix,
and refuses to write unless the shipped PNG is byte-reproducible. It never claims a
from-scratch reprojection was done — recomputing the projections needs a GPU pass over
25M tokens. Writes to docs/e1_figures/ only.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = Path("docs/e1_figures/FIGURES_MANIFEST.json")
FIG_KEY = "E1_Figure_1_maxact.png"
PNG = Path("docs/e1_figures") / FIG_KEY
PDF = PNG.with_suffix(".pdf")
RENDER = "geometry_out/render_fig1.py"
BUILD = "geometry_out/build_fig1.py"
SWEEP = Path("data/heldout/maxact")
RENDER_DATA = SWEEP / "fig1_render_data.json"
SELECTION = SWEEP / "fig1_selection.json"
PASS1 = SWEEP / "pass1_results.json"
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    for p in (MANIFEST, PNG, PDF, RENDER, BUILD, RENDER_DATA, SELECTION, PASS1):
        if not Path(p).exists():
            sys.exit("FATAL: missing %s" % p)

    rd = json.loads(RENDER_DATA.read_text())
    p1 = json.loads(PASS1.read_text())
    if rd.get("vectors_sha256") != EXPECTED_VEC_SHA:
        sys.exit("FATAL: render data is not the canonical basis")
    if p1.get("vectors_sha256") != EXPECTED_VEC_SHA or p1.get("vector_set") != "cleaned":
        sys.exit("FATAL: sweep pass1 is not the canonical cleaned basis")

    png_sha = sha256_file(PNG)
    with tempfile.TemporaryDirectory() as td:
        probe = str(Path(td) / "probe")
        r = subprocess.run([sys.executable, RENDER, "--out-prefix", probe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("FATAL: producer failed on the verification re-render:\n" + r.stderr[-2000:])
        if sha256_file(probe + ".png") != png_sha:
            sys.exit("FATAL: PNG is not reproducible from the current producer")
    print("[verify] PNG byte-reproducible on re-render")

    import PIL
    man = json.loads(MANIFEST.read_text())
    man["figures"][FIG_KEY] = {
        "sha256": png_sha,
        "basis": "canonical",
        "producer": "render_fig1.py (+ build_fig1.py)",
        "producer_sha256": sha256_file(RENDER),
        "build_sha256": sha256_file(BUILD),
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": [
            "%s (render data; basis canonical, vectors_sha256 %s…)" % (RENDER_DATA, EXPECTED_VEC_SHA[:12]),
            "%s (panel selection)" % SELECTION,
            "%s (sweep computation record; vectors_sha256 %s…, cleaned, L%s, %s docs / %s tokens)"
            % (PASS1, EXPECTED_VEC_SHA[:12], p1.get("layer"), p1.get("n_docs"), p1.get("n_tokens_swept")),
        ],
        "input_sha256": {
            "fig1_render_data.json": sha256_file(RENDER_DATA),
            "fig1_selection.json": sha256_file(SELECTION),
            "pass1_results.json": sha256_file(PASS1),
        },
        "companion_files": ["docs/e1_figures/" + PDF.name],
        "companion_sha256": {PDF.name: sha256_file(PDF)},
        "basis_verification": (
            "PROVENANCE-RECORD gate, not from-scratch reprojection. fig1 reads the sweep's "
            "per-token projections; recomputing them needs a GPU pass over 25M tokens. The "
            "sweep's own record (pass1_results.json) and the render data both stamp "
            "vectors_sha256 %s (cleaned, L31), and magnitude_diagnostic.json independently "
            "stamps basis canonical. The render hard-exits if either stamp is wrong." % EXPECTED_VEC_SHA),
        "curation": rd.get("curation") or json.loads(SELECTION.read_text()).get("curation"),
        "lockable": True,
        "notes": (
            "Figure 1, parallel to Anthropic Figure 1: 12 emotion panels x 7 max-activating "
            "windows each, from the maxact sweep (whole-corpus per-token projection of "
            "resid_post[L31] onto the unit-normalized cleaned canonical vectors; 47,274 docs / "
            "25,089,246 tokens). Each token shaded iff its projection exceeds the vector's p90 "
            "floor; intensity = (value - p90_floor)/(top1_value - p90_floor), per panel, so the "
            "color scale is normalized to each vector's own corpus-max activation (HONEST scale — "
            "not capped). PIL-based; deterministic given the render data -> byte-reproducible "
            "(PNG and, after CreationDate pinning, PDF). "
            "ALL 12 PANELS HAND-CURATED: the 10 clean-emotion panels earlier, selected for "
            "printability; desperate & surprised (re)curated 2026-08-18 as a hybrid of "
            "top-of-plateau STRUCTURAL rows (the vector's strongest activations land on "
            "structural tokens -- a space in 'FIG. 6', 'According', a citation digit -- NOT on "
            "emotion words; the same 'FIG. 6' patent doc is the #1 desperate AND #2 surprised "
            "activation, so the top activation is not even emotion-specific) plus genuinely "
            "on-emotion HIGH-DENSITY rows (which peak lower on the plateau, ~2000-2800 vs the "
            "~5231 structural top, so they render less dark on the honest scale). This is the "
            "max-activation-confound story (>=52%% of vectors peak on structurally "
            "non-conceptual tokens) made visual. A whitespace-peak render fix ensures a "
            "space/newline peak token (e.g. the 'FIG. 6' space) still draws its highlight. "
            "Rendered with Pillow %s. Supersedes the prior lock Gemma_Figure_1.png (2ae66a68, dead "
            "under the single-basis policy)." % PIL.__version__),
    }
    MANIFEST.write_text(json.dumps(man, indent=1))
    print("[updated] %s -> %s" % (MANIFEST, FIG_KEY))
    print("          png      sha256 %s" % png_sha)
    print("          pdf      sha256 %s" % sha256_file(PDF))
    print("          producer sha256 %s (render) / %s (build)" % (sha256_file(RENDER)[:12], sha256_file(BUILD)[:12]))


if __name__ == "__main__":
    main()
