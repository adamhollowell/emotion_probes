#!/usr/bin/env python3
"""
Refresh the Figure 2 (canonical cosine matrix) entry in docs/e1_figures/FIGURES_MANIFEST.json.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/update_fig2_manifest.py

Re-runs the producer to a temp prefix and refuses to write unless BOTH the shipped
PNG and PDF are byte-reproducible from it. Records the literal render-time hashes.

Writes to docs/e1_figures/ only. Never touches locked_figures_and_tables/ — per the
figure-lock freeze, Figure 2 is not promoted until its canonical bytes are independently
confirmed.
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
FIG_KEY = "E1_Figure_2_cosine_matrix.png"
PREFIX = Path("docs/e1_figures/E1_Figure_2_cosine_matrix")
PRODUCER = "repo/github_repo/scripts/render_parallel_fig5_cosine_matrix.py"
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
    png, pdf = PREFIX.with_suffix(".png"), PREFIX.with_suffix(".pdf")
    for p in (MANIFEST, png, pdf, PRODUCER, FIGSTYLE, VECTORS):
        if not Path(p).exists():
            sys.exit("FATAL: missing %s" % p)

    vec_sha = sha256_file(VECTORS)
    if vec_sha != EXPECTED_VEC_SHA:
        sys.exit("FATAL: %s is not the canonical basis" % VECTORS)

    png_sha, pdf_sha = sha256_file(png), sha256_file(pdf)

    # The manifest must describe what the producer actually emits right now.
    with tempfile.TemporaryDirectory() as td:
        probe = str(Path(td) / "probe")
        r = subprocess.run([sys.executable, PRODUCER, "--out-prefix", probe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("FATAL: producer failed on the verification re-render:\n" + r.stderr[-2000:])
        if sha256_file(probe + ".png") != png_sha:
            sys.exit("FATAL: PNG is not reproducible from the current producer")
        if sha256_file(probe + ".pdf") != pdf_sha:
            sys.exit("FATAL: PDF is not reproducible from the current producer")
    print("[verify] PNG and PDF both byte-reproducible on re-render")

    import scipy, numpy, matplotlib
    man = json.loads(MANIFEST.read_text())
    man["figures"][FIG_KEY] = {
        "sha256": png_sha,
        "basis": "canonical",
        "producer": "render_parallel_fig5_cosine_matrix.py",
        "producer_sha256": sha256_file(PRODUCER),
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": [
            "%s (canonical basis; sha256 %s…; hash-gated at render — the filename "
            "emotion_vectors_171.pt denotes DIFFERENT bases in different directories, "
            "so only the hash identifies the basis)" % (VECTORS, vec_sha[:12]),
        ],
        "companion_files": ["docs/e1_figures/E1_Figure_2_cosine_matrix.pdf"],
        "companion_sha256": {"E1_Figure_2_cosine_matrix.pdf": pdf_sha},
        "notes": (
            "171x171 pairwise cosine similarity of the CLEANED L31 emotion vectors "
            "(blob['vectors'][:, 31, :], never vectors_raw), ordered by "
            "figstyle.cluster_order(average linkage on cosine distance), RdBu_r, "
            "vlim +/-1. Deterministic: no GPU, no API, no k-means, no UMAP, no random "
            "seed — re-render is byte-identical (PNG and PDF; the PDF CreationDate is "
            "pinned so it can be byte-compared). Gates at render: shape 171x171, "
            "diagonal ~1.0, values in [-1,1], and the confirmed canonical neighbour cosines "
            "afraid<->scared 0.9867, joyful<->happy 0.9766, "
            "compassionate<->empathetic 0.9655. Positive block leads top-left via the "
            "deterministic orientation flip (not triggered on this basis: the "
            "positive anchors already lead, mean rank 43.5/171). INTENTIONALLY "
            "PARTITION-FREE — the block structure emerges from the ordering alone; do "
            "not annotate with the k=10 families. Leaf order is scipy-tie-sensitive: "
            "rendered at scipy %s / numpy %s / matplotlib %s. NOT LOCKED — supersedes "
            "the earlier lock Gemma_Figure_5.png (b26cf606, dead under the canonical-only "
            "policy); promotion to locked_figures_and_tables/ awaits independent "
            "confirmation of these bytes."
            % (scipy.__version__, numpy.__version__, matplotlib.__version__)),
        "figstyle_sha256": sha256_file(FIGSTYLE),
    }
    MANIFEST.write_text(json.dumps(man, indent=1))
    print("[updated] %s -> %s" % (MANIFEST, FIG_KEY))
    print("          png      sha256 %s" % png_sha)
    print("          pdf      sha256 %s" % pdf_sha)
    print("          producer sha256 %s" % sha256_file(PRODUCER))
    print("          figstyle sha256 %s" % sha256_file(FIGSTYLE))


if __name__ == "__main__":
    main()
