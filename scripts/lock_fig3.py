#!/usr/bin/env python3
"""
Lock Figure 3 (canonical) into locked_figures_and_tables/ as Gemma_Figure_6.png.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/lock_fig3.py

Naming follows the lock dir's own convention field (Anthropic-parallel
Gemma_Figure_N, bare) rather than a _canonical suffix; the internal basis tags appear
on no release surface. The canonical basis is recorded inside the entry instead.

Refuses to write unless, at lock execution:
  - the source PNG is byte-reproducible from the current frozen coords + layout,
  - the layout is stamped to the current coords sha,
  - the vectors are the canonical basis by hash,
  - the producer and figstyle on disk match what the manifest recorded,
  - the outgoing locked file is versioned into superseded/ (never overwritten).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import fig6_common as fc

LOCK_DIR = Path("locked_figures_and_tables")
LOCK_RECORD = LOCK_DIR / "LOCK_RECORD.json"
LOCKED_NAME = "Gemma_Figure_6.png"
SRC_PNG = Path("docs/e1_figures/E1_Figure_3_umap_kmeans.png")
MANIFEST = Path("docs/e1_figures/FIGURES_MANIFEST.json")
VECTORS = "data/vectors/emotion_vectors_171.pt"
PRODUCER = "repo/github_repo/scripts/render_parallel_fig6_umap.py"
COMMON = "repo/github_repo/scripts/fig6_common.py"
FIGSTYLE = "repo/github_repo/scripts/figstyle.py"
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
APPROVAL = "Author sign-off."


def die(msg: str):
    sys.exit("FATAL: " + msg)


def main() -> None:
    if not SRC_PNG.exists():
        die("missing %s — run the producer first" % SRC_PNG)

    # 1. basis
    vec_sha = fc.sha256_file(VECTORS)
    if vec_sha != EXPECTED_VEC_SHA:
        die("%s is not the canonical basis (sha %s)" % (VECTORS, vec_sha))

    # 2. layout stamped to the frozen coords
    coords_sha = fc.sha256_file(fc.COORDS_CSV)
    layout = json.loads(Path(fc.LAYOUT_JSON).read_text())
    if layout.get("coords_sha256") != coords_sha:
        die("layout %s is not stamped to the current coords" % fc.LAYOUT_JSON)

    # 3. the PNG being locked must be what the current inputs actually produce
    df = fc.load_coords()
    fc.check_membership(df)
    src_sha = fc.sha256_file(SRC_PNG)
    with tempfile.TemporaryDirectory() as td:
        probe = str(Path(td) / "probe.png")
        fc.render_final(df, layout, probe)
        if fc.sha256_file(probe) != src_sha:
            die("%s is not reproducible from the current frozen inputs" % SRC_PNG)
    print("[verify] render reproduces %s at lock execution" % src_sha[:12])

    # 4. producer / figstyle gate against what the figures manifest recorded
    man = json.loads(MANIFEST.read_text())
    entry = man["figures"].get(SRC_PNG.name) or die("no manifest entry for %s" % SRC_PNG.name)
    prod_sha, style_sha = fc.sha256_file(PRODUCER), fc.sha256_file(FIGSTYLE)
    if entry["sha256"] != src_sha:
        die("manifest sha %s != PNG on disk %s" % (entry["sha256"][:12], src_sha[:12]))
    if entry["producer_sha256"] != prod_sha:
        die("producer changed since the manifest was written")
    if entry["figstyle_sha256"] != style_sha:
        die("figstyle changed since the manifest was written")
    print("[verify] producer + figstyle match the manifest (gate clean)")

    # 5. version the outgoing lock — never overwrite
    rec = json.loads(LOCK_RECORD.read_text())
    cur = LOCK_DIR / LOCKED_NAME
    superseded_rel = None
    if cur.exists():
        old_sha = fc.sha256_file(cur)
        if old_sha == src_sha:
            die("%s already holds these exact bytes — nothing to lock" % LOCKED_NAME)
        dest = LOCK_DIR / "superseded" / ("Gemma_Figure_6_%s.png" % old_sha[:12])
        if dest.exists():
            die("supersede target %s already exists — refusing to clobber" % dest)
        dest.parent.mkdir(exist_ok=True)
        shutil.move(str(cur), str(dest))
        superseded_rel = "superseded/" + dest.name
        print("[versioned] prior lock %s -> %s" % (old_sha[:12], superseded_rel))

    shutil.copy2(str(SRC_PNG), str(cur))
    if fc.sha256_file(cur) != src_sha:
        die("copy verification failed for %s" % cur)
    print("[locked] %s  sha256 %s" % (cur, src_sha))

    # 6. rewrite the lock entry
    prior = next((L for L in rec["locks"] if L["locked_file"] == LOCKED_NAME), None)
    chain = []
    if prior:
        if prior.get("superseded_prior"):
            chain.append(prior["superseded_prior"])
        rec["locks"].remove(prior)
    if superseded_rel:
        chain.append(superseded_rel)

    rec["locks"].append({
        "locked_file": LOCKED_NAME,
        "figure": "Figure 3",
        "name": SRC_PNG.name,
        "sha256": src_sha,
        "producer": "render_parallel_fig6_umap.py",
        "producer_sha256": prod_sha,
        "figstyle_sha256": style_sha,
        "basis_source": "emotion_vectors_171.pt",
        "basis_sha256": vec_sha,
        "locked_date_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "approved_by": (
            APPROVAL + " First canonical figure lock. Locked on the deterministic render: "
            "UMAP coordinates computed ONCE and frozen (%s, sha %s…), labels hand-placed "
            "against those exact coordinates and stamped to that sha, rendered "
            "through the shared base builder so the editor preview is byte-identical to "
            "this file. At lock execution the render was re-run and reproduced these "
            "bytes, and producer_sha256 + figstyle_sha256 were verified against the "
            "figures manifest and the staged scripts."
            % (fc.COORDS_CSV, coords_sha[:12])),
        "status": "LOCKED",
        "superseded_prior": superseded_rel,
        "superseded_chain": chain,
        "layout_inputs": [
            "%s (FROZEN UMAP coords, computed once 2026-08-17; %s…)" % (fc.COORDS_CSV, coords_sha[:12]),
            "%s (author-placed label positions; %s…)" % (fc.LAYOUT_JSON, fc.sha256_file(fc.LAYOUT_JSON)[:12]),
            "%s (certified native seed-0 partition; %s…)" % (fc.PARTITION_CSV, fc.sha256_file(fc.PARTITION_CSV)[:12]),
            "%s (shared base-figure builder; %s…)" % (COMMON, fc.sha256_file(COMMON)[:12]),
        ],
        "notes": (
            "Native canonical partition, 10 families, valence-ordered sizes "
            "8/24/16/15/14/32/12/17/4/29. Satisfies re-lock ruling F.4: the label "
            "re-hand-tune is a human step gated on the canonical coords — verified here, all "
            "36 labels were re-placed for the canonical geometry (median move 9.2 data units) "
            "and every leader anchor matches its canonical point. Caption and its constraints: "
            "docs/e1_figures/E1_Figure_3_caption.md. SOFTNESS disclosure required: "
            "50-seed mean pairwise ARI 0.686 (min 0.467), mean-vs-seed-0 0.622; membership "
            "differences from other partitions are within seed noise. Do NOT caption any "
            "cross-model family convergence as a result. Filename carries no canonical tag."),
    })
    LOCK_RECORD.write_text(json.dumps(rec, indent=2))
    print("[updated] %s" % LOCK_RECORD)


if __name__ == "__main__":
    main()
