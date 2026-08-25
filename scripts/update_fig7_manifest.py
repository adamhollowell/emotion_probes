#!/usr/bin/env python3
"""
Refresh the Figure 7 (canonical cross-layer RSA heatmap) entry in
docs/e1_figures/FIGURES_MANIFEST.json.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/update_fig7_manifest.py

Re-runs the producer to a temp prefix and refuses to write unless the shipped PNG is
byte-reproducible from it. Records literal render-time hashes plus the audit JSON's
RSA facts (sampled layers, diagonal bounds, adjacent-min location), so a wrong-basis
render can never be described as a right one.

Writes to docs/e1_figures/ only. Never touches locked_figures_and_tables/ — the canonical
figure-lock freeze holds until the RSA data is independently confirmed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = Path("docs/e1_figures/FIGURES_MANIFEST.json")
FIG_KEY = "E1_Figure_7_crosslayer_rsa.png"
PNG = Path("docs/e1_figures") / FIG_KEY
AUDIT = PNG.with_suffix(".audit.json")
PRODUCER = "repo/github_repo/scripts/render_parallel_fig9_cross_layer_rsa.py"
FIGSTYLE = "repo/github_repo/scripts/figstyle.py"
VECTORS = "data/vectors/emotion_vectors_171.pt"
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    for p in (MANIFEST, PNG, AUDIT, PRODUCER, FIGSTYLE, VECTORS):
        if not Path(p).exists():
            sys.exit("FATAL: missing %s" % p)

    vec_sha = sha256_file(VECTORS)
    if vec_sha != EXPECTED_VEC_SHA:
        sys.exit("FATAL: %s is not the canonical basis" % VECTORS)

    png_sha = sha256_file(PNG)
    audit = json.loads(AUDIT.read_text())
    if audit.get("source_vectors_sha256") != vec_sha:
        sys.exit("FATAL: audit JSON records a different source basis")
    adj = audit["adjacent_pairs"]
    cg = audit["content_gate"]

    with tempfile.TemporaryDirectory() as td:
        probe = str(Path(td) / "probe")
        r = subprocess.run([sys.executable, PRODUCER, VECTORS, "--out-prefix", probe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("FATAL: producer failed on the verification re-render:\n" + r.stderr[-2000:])
        if sha256_file(probe + ".png") != png_sha:
            sys.exit("FATAL: PNG is not reproducible from the current producer")
    print("[verify] PNG byte-reproducible on re-render")

    env = audit["env"]
    man = json.loads(MANIFEST.read_text())
    man["figures"][FIG_KEY] = {
        "sha256": png_sha,
        "basis": "canonical",
        "producer": "render_parallel_fig9_cross_layer_rsa.py",
        "producer_sha256": sha256_file(PRODUCER),
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": [
            "%s (canonical basis; sha256 %s…; hash-gated at render — the filename "
            "emotion_vectors_171.pt denotes DIFFERENT bases in different directories, "
            "so only the hash identifies the basis)" % (VECTORS, vec_sha[:12]),
        ],
        "companion_files": ["docs/e1_figures/" + AUDIT.name],
        "companion_sha256": {AUDIT.name: sha256_file(AUDIT)},
        "sampled_layers": audit["sampled_layers"],
        "adjacent_min": {"value": adj["min_value"], "layer_pair": adj["min_layer_pair"],
                         "in_seam_region": adj["min_in_seam_region_L22_26"]},
        "lockable": True,
        "notes": (
            "Figure 7, parallel to Anthropic Figure 9. Finding P3.1 (cross-layer "
            "representational stability). Layer x layer cosine-of-cosine RSA between "
            "the CLEANED emotion-probe geometries (blob['vectors'], never vectors_raw) "
            "at 14 evenly spaced layers np.linspace(12,40).round() = %s, which includes "
            "L31. Per layer: upper triangle of the 171x171 pairwise-cosine matrix, "
            "L2-normalised, then cosine similarity across layers. viridis vmin 0.8 "
            "vmax 1.0; Early/Early-Mid/Mid-Late/Late band labels (matches Anthropic; "
            "the numeric-layer mode is a blog/diagnostic variant, P3.8, not the E1 "
            "figure). METHOD UNCHANGED from the earlier version — Anthropic-faithful; the Pearson "
            "variant differs only ~1e-5 (P3.1/P3.8) and is not used. Deterministic "
            "(pure geometry, no seed) — byte-reproducible on re-render. "
            "The vectors are identified by sha256, not by the filename; the "
            "producer is hash-gated because the filename alone cannot identify "
            "the vectors. "
            "figstyle is now a HARD import (the earlier producer wrapped it in a try/except "
            "that silently fell back to different fonts + a hardcoded rust colour on "
            "import failure — a silent different-bytes path, removed here). "
            "CONTENT GATE at render (catches a wrong-basis/wrong-layer load even if the "
            "hash passed): L31 cosines afraid/scared %.4f, joyful/happy %.4f, "
            "compassionate/empathetic %.4f (same spot-check as Figure 2, tol 0.002), "
            "and the RSA diagonal is ~1.0 (%.6f..%.6f). "
            "SEAM: the adjacent-pair minimum is %.4f at L%d->L%d, inside the L22-26 "
            "emergence seam — the sampled-sweep echo of P3.1. This is a DIFFERENT cut "
            "from the full-46-layer adjacent steps in adjacent_rsa.json (true "
            "L21->L22 0.9425, L43->L44 0.9329): same story, different numbers. The "
            "figure shows the L22-26 seam -> L27-40 stable plateau; it does NOT show "
            "the final-layer drop (L41-45), which is outside the 12-40 sampling window "
            "exactly as in Anthropic's figure — do not caption a 'final drop' here "
            "(that lives in the RSA-to-L31 trajectory, P3.1 text). INTENTIONALLY "
            "PARTITION-FREE. Rendered at numpy %s / matplotlib %s. NOT LOCKED: "
            "final lock awaits an independent RSA re-derivation; supersedes nothing yet "
            "— the prior lock "
            "Gemma_Figure_9.png (eca93a9a) stands until promotion."
            % (audit["sampled_layers"],
               cg["spot_cosines"]["afraid|scared"], cg["spot_cosines"]["joyful|happy"],
               cg["spot_cosines"]["compassionate|empathetic"],
               cg["rsa_diagonal_min"], cg["rsa_diagonal_max"],
               adj["min_value"], adj["min_layer_pair"][0], adj["min_layer_pair"][1],
               env["numpy"], env["matplotlib"])),
        "figstyle_sha256": sha256_file(FIGSTYLE),
    }
    MANIFEST.write_text(json.dumps(man, indent=1))
    print("[updated] %s -> %s" % (MANIFEST, FIG_KEY))
    print("          png      sha256 %s" % png_sha)
    print("          audit    sha256 %s" % sha256_file(AUDIT))
    print("          producer sha256 %s" % sha256_file(PRODUCER))
    print("          adjacent min %.4f at L%d->L%d (seam=%s)"
          % (adj["min_value"], adj["min_layer_pair"][0], adj["min_layer_pair"][1],
             adj["min_in_seam_region_L22_26"]))


if __name__ == "__main__":
    main()
