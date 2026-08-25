#!/usr/bin/env python3
"""
Lock Figure 4 (canonical PC-projection bars) into locked_figures_and_tables/ as Gemma_Figure_7.png.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/lock_fig4.py

The lock did not include an independent second-party reproduction of these bytes; the
assurance is a hash-gated canonical basis + byte-reproducible re-render at lock
execution.

Naming follows the lock dir's convention field (Anthropic-parallel Gemma_Figure_N,
bare, no _canonical tag).

Refuses to write unless, at lock execution: the vectors are the canonical basis by hash,
the PNG is byte-reproducible from the current producer, the producer and figstyle
match what the figures manifest recorded, and the outgoing locked file is versioned
into superseded/ rather than overwritten.
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
LOCKED_NAME = "Gemma_Figure_7.png"
PNG = Path("docs/e1_figures/E1_Figure_4_pc_variance.png")
AUDIT = PNG.with_suffix(".audit.json")
MANIFEST = Path("docs/e1_figures/FIGURES_MANIFEST.json")
PRODUCER = "repo/github_repo/scripts/render_parallel_fig7_pc_projections.py"
FIGSTYLE = "repo/github_repo/scripts/figstyle.py"
VECTORS = "data/vectors/emotion_vectors_171.pt"
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
EXPECTED_PNG_SHA = "bcd4bb584fb75015061c07394fdfa4b80166fc3c056421aacbc0e0b428615cd6"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def die(msg: str):
    sys.exit("FATAL: " + msg)


def main() -> None:
    for p in (LOCK_RECORD, MANIFEST, PNG, AUDIT, PRODUCER, FIGSTYLE, VECTORS):
        if not Path(p).exists():
            die("missing %s" % p)

    vec_sha = sha256_file(VECTORS)
    if vec_sha != EXPECTED_VEC_SHA:
        die("%s is not the canonical basis (sha %s)" % (VECTORS, vec_sha))

    png_sha = sha256_file(PNG)
    if png_sha != EXPECTED_PNG_SHA:
        die("source PNG sha %s != the sha named in the figures manifest %s"
            % (png_sha, EXPECTED_PNG_SHA))
    print("[verify] source PNG sha matches the figures manifest (%s)" % png_sha[:12])

    audit = json.loads(AUDIT.read_text())
    evr = audit["explained_variance_percent"]
    if audit.get("source_vectors_sha256") != vec_sha:
        die("audit JSON records a different source basis")

    with tempfile.TemporaryDirectory() as td:
        probe = str(Path(td) / "probe")
        r = subprocess.run([sys.executable, PRODUCER, VECTORS, "--out-prefix", probe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            die("producer failed on the lock-time re-render:\n" + r.stderr[-2000:])
        if sha256_file(probe + ".png") != png_sha:
            die("PNG is not reproducible from the current producer")
    print("[verify] PNG reproduces at lock execution")

    man = json.loads(MANIFEST.read_text())
    entry = man["figures"].get(PNG.name) or die("no manifest entry for %s" % PNG.name)
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
        dest = LOCK_DIR / "superseded" / ("Gemma_Figure_7_%s.png" % old_sha[:12])
        if dest.exists():
            die("supersede target %s already exists — refusing to clobber" % dest)
        dest.parent.mkdir(exist_ok=True)
        shutil.move(str(cur), str(dest))
        superseded_rel = "superseded/" + dest.name
        print("[versioned] superseded render %s -> %s" % (old_sha[:12], superseded_rel))

    shutil.copy2(str(PNG), str(cur))
    if sha256_file(cur) != png_sha:
        die("copy verification failed for %s" % cur)
    print("[locked] %s  sha256 %s" % (cur, png_sha))

    prior = next((L for L in rec["locks"] if L["locked_file"] == LOCKED_NAME), None)
    chain = []
    if prior:
        chain = list(prior.get("superseded_chain") or [])
        if prior.get("superseded_prior") and prior["superseded_prior"] not in chain:
            chain.append(prior["superseded_prior"])
        rec["locks"].remove(prior)
    if superseded_rel:
        chain.append(superseded_rel)

    import sklearn, numpy, matplotlib
    rec["locks"].append({
        "locked_file": LOCKED_NAME,
        "figure": "Figure 4",
        "name": PNG.name,
        "sha256": png_sha,
        "producer": "render_parallel_fig7_pc_projections.py",
        "producer_sha256": prod_sha,
        "figstyle_sha256": style_sha,
        "basis_source": "emotion_vectors_171.pt",
        "basis_sha256": vec_sha,
        "locked_date_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "approved_by": (
            "Locked from the canonical basis (sha 6de2e293) with a byte-reproducible "
            "re-render at lock execution. No independent second-party reproduction of "
            "these bytes is on record; the assurance is hash-gated basis + "
            "byte-reproducible re-render."),
        "status": "LOCKED",
        "superseded_prior": superseded_rel,
        "superseded_chain": chain,
        "companion_files": ["docs/e1_figures/" + AUDIT.name],
        "companion_sha256": {AUDIT.name: sha256_file(AUDIT)},
        "explained_variance_percent": evr,
        "notes": (
            "Two stacked ranked-bar panels: each of the 171 emotions' "
            "projection onto PC1 and PC2 of a global PCA over the CLEANED L31 vectors "
            "(blob['vectors'][:, 31, :], never vectors_raw), z-scored for display, "
            "rank-sorted, 43 representative labels per panel. Deterministic (SVD, no "
            "seed; signs fixed by anchor groups) - byte-reproducible on re-render. "
            "Variance computed from the data and printed in the y-labels: PC1 %.3f%% "
            "-> '27%% var', PC2 %.3f%% -> '13%% var', PC3 %.3f%%. The render gate "
            "hard-exits unless these match the confirmed values within 0.05 (a wrong-basis "
            "load reads 26.07/13.35/11.87). The filename "
            "the vectors are identified by sha256, not by the filename; the "
            "producer is hash-gated "
            "because the filename cannot identify the basis. ORIENTATION verified "
            "against full ranks, not the sparse printed ticks: PC1 poles "
            "optimistic/kind/cheerful (ranks 170/169/168) vs "
            "terrified/hysterical/frightened (0/1/2); PC2 poles "
            "playful/outraged/indignant (170/169/168) vs nostalgic/serene/depressed "
            "(0/1/2); anchor margins PC1 585.0, PC2 325.5. INTENTIONALLY "
            "PARTITION-FREE - no k=10 family coloring. Rendered at scikit-learn %s / "
            "numpy %s / matplotlib %s. Supersedes the prior lock 5f823fda (dead under the "
            "single-basis policy)."
            % (evr["PC1"], evr["PC2"], evr["PC3"],
               sklearn.__version__, numpy.__version__, matplotlib.__version__)),
    })
    LOCK_RECORD.write_text(json.dumps(rec, indent=2))
    print("[updated] %s" % LOCK_RECORD)


if __name__ == "__main__":
    main()
