#!/usr/bin/env python3
"""
Refresh the Figure 3 (canonical) entry in docs/e1_figures/FIGURES_MANIFEST.json.

Run from the PROJECT ROOT, after the producer has re-rendered the PNG:

    python3 repo/github_repo/scripts/render_parallel_fig6_umap.py
    python3 repo/github_repo/scripts/update_fig3_manifest.py

Records the literal render-time hashes of everything the render consumed: the
frozen coords, the hand-placed layout, the certified partition, the canonical vectors,
the shared base builder, the producer, and figstyle. It re-derives every sha from
the files on disk — nothing is hardcoded — and refuses to write if the layout is
not stamped to the current coords or if the PNG on disk is not what the current
inputs produce.

Writes to docs/e1_figures/ only. Never touches locked_figures_and_tables/.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fig6_common as fc

MANIFEST = "docs/e1_figures/FIGURES_MANIFEST.json"
FIG_KEY = "E1_Figure_3_umap_kmeans.png"
FIG_PNG = "docs/e1_figures/" + FIG_KEY
VECTORS = "data/vectors/emotion_vectors_171.pt"
PRODUCER = "repo/github_repo/scripts/render_parallel_fig6_umap.py"
COMMON = "repo/github_repo/scripts/fig6_common.py"
FIGSTYLE = "repo/github_repo/scripts/figstyle.py"
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"


def main() -> None:
    for p in (MANIFEST, FIG_PNG, VECTORS, PRODUCER, COMMON, FIGSTYLE,
              fc.COORDS_CSV, fc.LAYOUT_JSON, fc.PARTITION_CSV):
        if not Path(p).exists():
            sys.exit("FATAL: missing input %s" % p)

    vec_sha = fc.sha256_file(VECTORS)
    if vec_sha != EXPECTED_VEC_SHA:
        sys.exit("FATAL: %s is not the canonical basis (sha %s)" % (VECTORS, vec_sha))

    coords_sha = fc.sha256_file(fc.COORDS_CSV)
    layout = json.loads(Path(fc.LAYOUT_JSON).read_text())
    if layout.get("coords_sha256") != coords_sha:
        sys.exit("FATAL: layout %s is stamped to a different coords file. Re-place "
                 "the labels against the current coords." % fc.LAYOUT_JSON)

    # The manifest must describe the PNG that these inputs actually produce, not a
    # stale render left in the tree. Re-render to a temp path and compare bytes.
    import tempfile
    df = fc.load_coords()
    fc.check_membership(df)
    with tempfile.TemporaryDirectory() as td:
        probe = str(Path(td) / "probe.png")
        fc.render_final(df, layout, probe)
        if fc.sha256_file(probe) != fc.sha256_file(FIG_PNG):
            sys.exit("FATAL: %s is not reproducible from the current inputs — re-run "
                     "the producer before refreshing the manifest." % FIG_PNG)

    png_sha = fc.sha256_file(FIG_PNG)
    man = json.loads(Path(MANIFEST).read_text())
    man["figures"][FIG_KEY] = {
        "sha256": png_sha,
        "basis": "canonical native partition",
        "producer": "render_parallel_fig6_umap.py",
        "producer_sha256": fc.sha256_file(PRODUCER),
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": [
            "%s (canonical basis; sha256 %s…; hash-gated at render)" % (VECTORS, vec_sha[:12]),
            "%s (certified native seed-0 KMeans k=10, rs=0, n_init=10, sklearn 1.8.0; "
            "sha256 %s…; ARI 1.0 vs a fresh seed-0 run)"
            % (fc.PARTITION_CSV, fc.sha256_file(fc.PARTITION_CSV)[:12]),
            "%s (FROZEN UMAP coords, computed once 2026-08-17; sha256 %s…)"
            % (fc.COORDS_CSV, coords_sha[:12]),
            "%s (author-placed label positions, stamped to the coords sha; sha256 %s…)"
            % (fc.LAYOUT_JSON, fc.sha256_file(fc.LAYOUT_JSON)[:12]),
            "%s (shared base-figure builder; sha256 %s…)" % (COMMON, fc.sha256_file(COMMON)[:12]),
        ],
        "notes": (
            "canonical UMAP scatter on the CERTIFIED NATIVE canonical partition (fresh seed-0 "
            "KMeans, not reused prior-basis membership). 10 families, valence-ordered "
            "sizes 8/24/16/15/14/32/12/17/4/29; family names per Appendix B. "
            "COORDINATES FROZEN: umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', "
            "random_state=42) on the cleaned L31 slice, umap-learn 0.5.12, computed ONCE "
            "into %s and never recomputed; the producer's --recompute hard-exits if that "
            "file exists. Faithfulness gate at freeze time: mean cosine of each point's 6 "
            "nearest 2-D neighbours = 0.6951 (random baseline 0.0045). DETERMINISTIC "
            "render: frozen coords + layout -> fig6_common.render_final; no UMAP recompute "
            "and no adjustText in the final path; the editor preview is byte-identical to "
            "this PNG. SOFTNESS: the partition is one soft realization (50-seed mean "
            "pairwise ARI 0.686, min 0.467; mean-vs-seed-0 0.622, i.e. the seed-0 draw is "
            "less central than average); membership differences from other partitions are "
            "within seed noise, NOT basis findings — do not caption any cross-model "
            "convergence as a result. Caption text: docs/e1_figures/"
            "E1_Figure_3_caption.md. NOT LOCKED: rendered to docs/e1_figures/ only."
            % fc.COORDS_CSV),
        "figstyle_sha256": fc.sha256_file(FIGSTYLE),
    }
    Path(MANIFEST).write_text(json.dumps(man, indent=1))
    print("[updated] %s -> %s" % (MANIFEST, FIG_KEY))
    print("          png      sha256 %s" % png_sha)
    print("          coords   sha256 %s" % coords_sha)
    print("          layout   sha256 %s" % fc.sha256_file(fc.LAYOUT_JSON))
    print("          producer sha256 %s" % fc.sha256_file(PRODUCER))
    print("          common   sha256 %s" % fc.sha256_file(COMMON))
    print("          figstyle sha256 %s" % fc.sha256_file(FIGSTYLE))


if __name__ == "__main__":
    main()
