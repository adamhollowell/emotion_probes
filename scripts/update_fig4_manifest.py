#!/usr/bin/env python3
"""
Refresh the Figure 4 (canonical PC-projection bars) entry in docs/e1_figures/FIGURES_MANIFEST.json.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/update_fig4_manifest.py

Re-runs the producer to a temp prefix and refuses to write unless the shipped PNG is
byte-reproducible from it. Records literal render-time hashes plus the audit JSON's
variance shares, so a wrong-basis render can never be described as a right one.

Writes to docs/e1_figures/ only. Never touches locked_figures_and_tables/ — the canonical
figure-lock freeze holds until a figure's bytes are independently confirmed.
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
FIG_KEY = "E1_Figure_4_pc_variance.png"
PNG = Path("docs/e1_figures") / FIG_KEY
AUDIT = PNG.with_suffix(".audit.json")
PRODUCER = "repo/github_repo/scripts/render_parallel_fig7_pc_projections.py"
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
    evr = audit["explained_variance_percent"]
    if audit.get("source_vectors_sha256") != vec_sha:
        sys.exit("FATAL: audit JSON records a different source basis")

    with tempfile.TemporaryDirectory() as td:
        probe = str(Path(td) / "probe")
        r = subprocess.run([sys.executable, PRODUCER, VECTORS, "--out-prefix", probe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("FATAL: producer failed on the verification re-render:\n" + r.stderr[-2000:])
        if sha256_file(probe + ".png") != png_sha:
            sys.exit("FATAL: PNG is not reproducible from the current producer")
    print("[verify] PNG byte-reproducible on re-render")

    import sklearn, numpy, matplotlib
    man = json.loads(MANIFEST.read_text())
    man["figures"][FIG_KEY] = {
        "sha256": png_sha,
        "basis": "canonical",
        "producer": "render_parallel_fig7_pc_projections.py",
        "producer_sha256": sha256_file(PRODUCER),
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": [
            "%s (canonical basis; sha256 %s…; hash-gated at render — the filename "
            "emotion_vectors_171.pt denotes DIFFERENT bases in different directories, "
            "so only the hash identifies the basis)" % (VECTORS, vec_sha[:12]),
        ],
        "companion_files": ["docs/e1_figures/" + AUDIT.name],
        "companion_sha256": {AUDIT.name: sha256_file(AUDIT)},
        "explained_variance_percent": evr,
        "notes": (
            "Two stacked ranked-bar panels: each of the 171 emotions' projection onto "
            "PC1 and PC2 of a global PCA over the CLEANED L31 vectors "
            "(blob['vectors'][:, 31, :], never vectors_raw), z-scored for display, "
            "rank-sorted within each panel, 43 representative labels printed per panel. "
            "Deterministic (SVD, no seed; signs fixed by anchor groups) — "
            "byte-reproducible on re-render. Variance shares are computed from the data "
            "and printed in the y-labels: PC1 %.2f%% -> prints '27%% var', PC2 %.2f%% -> "
            "'13%% var', PC3 %.2f%%. Gate at render hard-exits unless these match the "
            "confirmed values within 0.05 (a wrong-basis load reads 26.07/13.35/11.87). "
            "ORIENTATION verified, not assumed: both PCs flipped to satisfy their anchor "
            "groups with clean margins (PC1 585.0, PC2 325.5), and the resulting poles "
            "are exactly optimistic/kind/cheerful (ranks 170/169/168) vs "
            "terrified/hysterical/frightened (0/1/2) on PC1, and "
            "playful/outraged/indignant (170/169/168) vs nostalgic/serene/depressed "
            "(0/1/2) on PC2. INTENTIONALLY PARTITION-FREE — no k=10 family coloring. "
            "Rendered at scikit-learn %s / numpy %s / matplotlib %s. NOT LOCKED — "
            "supersedes the earlier lock Gemma_Figure_7.png (5f823fda, dead under the "
            "single-basis policy); final lock awaits independent byte-confirmation."
            % (evr["PC1"], evr["PC2"], evr["PC3"],
               sklearn.__version__, numpy.__version__, matplotlib.__version__)),
        "figstyle_sha256": sha256_file(FIGSTYLE),
    }
    MANIFEST.write_text(json.dumps(man, indent=1))
    print("[updated] %s -> %s" % (MANIFEST, FIG_KEY))
    print("          png      sha256 %s" % png_sha)
    print("          audit    sha256 %s" % sha256_file(AUDIT))
    print("          producer sha256 %s" % sha256_file(PRODUCER))
    print("          variance PC1 %.3f  PC2 %.3f  PC3 %.3f" % (evr["PC1"], evr["PC2"], evr["PC3"]))


if __name__ == "__main__":
    main()
