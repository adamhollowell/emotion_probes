#!/usr/bin/env python3
"""
Lock Figure 2 (canonical cosine matrix) into locked_figures_and_tables/ as Gemma_Figure_5.png.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/lock_fig2.py

This figure was NOT independently re-rendered by a second party; the lock rests on a
hash-gated canonical basis and a byte-reproducible re-render at lock execution.

Naming follows the lock dir's convention field (Anthropic-parallel Gemma_Figure_N,
bare, no _canonical tag).

Refuses to write unless, at lock execution:
  - the vectors are the canonical basis by hash,
  - the PNG and PDF are byte-reproducible from the current producer,
  - the producer and figstyle on disk match what the figures manifest recorded,
  - the outgoing locked file is versioned into superseded/ (never overwritten).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

LOCK_DIR = Path("locked_figures_and_tables")
LOCK_RECORD = LOCK_DIR / "LOCK_RECORD.json"
LOCKED_NAME = "Gemma_Figure_5.png"
PREFIX = Path("docs/e1_figures/E1_Figure_2_cosine_matrix")
MANIFEST = Path("docs/e1_figures/FIGURES_MANIFEST.json")
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


def die(msg: str):
    sys.exit("FATAL: " + msg)


def main() -> None:
    png, pdf = PREFIX.with_suffix(".png"), PREFIX.with_suffix(".pdf")
    for p in (LOCK_RECORD, MANIFEST, png, pdf, PRODUCER, FIGSTYLE, VECTORS):
        if not Path(p).exists():
            die("missing %s" % p)

    vec_sha = sha256_file(VECTORS)
    if vec_sha != EXPECTED_VEC_SHA:
        die("%s is not the canonical basis (sha %s)" % (VECTORS, vec_sha))

    png_sha, pdf_sha = sha256_file(png), sha256_file(pdf)

    # The bytes being locked must be what the producer emits right now.
    with tempfile.TemporaryDirectory() as td:
        probe = str(Path(td) / "probe")
        r = subprocess.run([sys.executable, PRODUCER, "--out-prefix", probe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            die("producer failed on the lock-time re-render:\n" + r.stderr[-2000:])
        if sha256_file(probe + ".png") != png_sha:
            die("PNG is not reproducible from the current producer")
        if sha256_file(probe + ".pdf") != pdf_sha:
            die("PDF is not reproducible from the current producer")
    print("[verify] PNG + PDF reproduce at lock execution (%s / %s)" % (png_sha[:12], pdf_sha[:12]))

    man = json.loads(MANIFEST.read_text())
    entry = man["figures"].get(png.name) or die("no manifest entry for %s" % png.name)
    prod_sha, style_sha = sha256_file(PRODUCER), sha256_file(FIGSTYLE)
    if entry["sha256"] != png_sha:
        die("manifest sha != PNG on disk")
    if entry["producer_sha256"] != prod_sha:
        die("producer changed since the manifest was written")
    if entry["figstyle_sha256"] != style_sha:
        die("figstyle changed since the manifest was written")
    print("[verify] producer + figstyle match the manifest (gate clean)")

    rec = json.loads(LOCK_RECORD.read_text())
    cur = LOCK_DIR / LOCKED_NAME
    superseded_rel = None
    if cur.exists():
        old_sha = sha256_file(cur)
        if old_sha == png_sha:
            die("%s already holds these exact bytes — nothing to lock" % LOCKED_NAME)
        dest = LOCK_DIR / "superseded" / ("Gemma_Figure_5_%s.png" % old_sha[:12])
        if dest.exists():
            die("supersede target %s already exists — refusing to clobber" % dest)
        dest.parent.mkdir(exist_ok=True)
        shutil.move(str(cur), str(dest))
        superseded_rel = "superseded/" + dest.name
        print("[versioned] prior lock %s -> %s" % (old_sha[:12], superseded_rel))

    shutil.copy2(str(png), str(cur))
    if sha256_file(cur) != png_sha:
        die("copy verification failed for %s" % cur)
    print("[locked] %s  sha256 %s" % (cur, png_sha))

    prior = next((L for L in rec["locks"] if L["locked_file"] == LOCKED_NAME), None)
    chain = []
    if prior:
        if prior.get("superseded_prior"):
            chain.append(prior["superseded_prior"])
        if prior.get("superseded_chain"):
            chain = list(prior["superseded_chain"])
        rec["locks"].remove(prior)
    if superseded_rel:
        chain.append(superseded_rel)

    import scipy, numpy, matplotlib
    rec["locks"].append({
        "locked_file": LOCKED_NAME,
        "figure": "Figure 2",
        "name": png.name,
        "sha256": png_sha,
        "producer": "render_parallel_fig5_cosine_matrix.py",
        "producer_sha256": prod_sha,
        "figstyle_sha256": style_sha,
        "basis_source": "emotion_vectors_171.pt",
        "basis_sha256": vec_sha,
        "locked_date_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "approved_by": (
            "Locked from the canonical basis (sha 6de2e293) with a byte-reproducible "
            "re-render at lock execution. These bytes (%s…) were first computed for "
            "this lock and have no prior independent record -- the assurance is "
            "hash-gated basis + byte-reproducible re-render, NOT a second party's "
            "independent reproduction." % png_sha[:12]),
        "status": "LOCKED",
        "superseded_prior": superseded_rel,
        "superseded_chain": chain,
        "companion_files": ["docs/e1_figures/E1_Figure_2_cosine_matrix.pdf"],
        "companion_sha256": {"E1_Figure_2_cosine_matrix.pdf": pdf_sha},
        "notes": (
            "171x171 pairwise cosine similarity of the CLEANED L31 vectors "
            "(blob['vectors'][:, 31, :], never vectors_raw), ordered by average linkage "
            "on cosine distance, RdBu_r, vlim +/-1. Pure geometry: no GPU, no API, no "
            "k-means, no UMAP, no random seed — PNG and PDF both byte-reproducible (the "
            "PDF CreationDate is pinned so it can be byte-compared). The producer is "
            "hash-gated on sha256 6de2e293 because the filename alone cannot identify "
            "the vectors. Gates at render: shape, diagonal ~1.0, range [-1,1], and "
            "neighbour cosines afraid<->scared 0.9867, joyful<->happy 0.9766, "
            "compassionate<->empathetic 0.9655. Positive block leads top-left; the "
            "deterministic orientation flip did NOT fire (anchors already "
            "lead, mean rank 43.5/171). INTENTIONALLY "
            "PARTITION-FREE — block structure emerges from the ordering alone; no k=10 "
            "family annotation. Leaf order is scipy-tie-sensitive: rendered at scipy %s "
            "/ numpy %s / matplotlib %s."
            % (scipy.__version__, numpy.__version__, matplotlib.__version__)),
    })
    LOCK_RECORD.write_text(json.dumps(rec, indent=2))
    print("[updated] %s" % LOCK_RECORD)


if __name__ == "__main__":
    main()
