#!/usr/bin/env python3
"""
Lock Figure 5 (canonical PCA vs Russell & Mehrabian human norms) into
locked_figures_and_tables/ as Gemma_Figure_8.png.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/lock_fig5.py

This figure was NOT independently re-rendered by a second party; the lock rests on a
hash-gated canonical basis and a byte-reproducible re-render at lock execution.

Naming follows the lock dir's convention field (Anthropic-parallel Gemma_Figure_N,
bare, no _canonical tag) — internal basis tags appear nowhere on a release surface.

Producer authority: LOCK_RECORD names render_parallel_fig8_russell_mehrabian.py. An
earlier reference to "fig8_rebuild_locked_audit.py" names NO FILE that has ever existed
in this repo; it is a documentation artifact and carries no authority.

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
LOCKED_NAME = "Gemma_Figure_8.png"
PREFIX = Path("docs/e1_figures/E1_Figure_5_human_norms")
MANIFEST = Path("docs/e1_figures/FIGURES_MANIFEST.json")
PRODUCER = "repo/github_repo/scripts/render_parallel_fig8_russell_mehrabian.py"
FIGSTYLE = "repo/github_repo/scripts/figstyle.py"
VECTORS = "data/vectors/emotion_vectors_171.pt"
RATINGS = "data/ratings/russell_mehrabian_1977.csv"
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
EXPECTED_RATINGS_SHA = "b8a0e952957e6c849d3507bdd0d6e848fcb8f308b628837e3c6dc8f77bb6cffd"


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
    for p in (LOCK_RECORD, MANIFEST, png, pdf, PRODUCER, FIGSTYLE, VECTORS, RATINGS):
        if not Path(p).exists():
            die("missing %s" % p)

    vec_sha = sha256_file(VECTORS)
    if vec_sha != EXPECTED_VEC_SHA:
        die("%s is not the canonical basis (sha %s)" % (VECTORS, vec_sha))

    # The human norms are basis-independent but were the subject of PROV.7(c): a
    # corrupted transcription (…_DISCARD_BAD.csv, 77 divergent cells) was deleted and
    # must never be used. Pin the canonical table by hash too.
    ratings_sha = sha256_file(RATINGS)
    if ratings_sha != EXPECTED_RATINGS_SHA:
        die("%s is not the canonical Russell & Mehrabian table (sha %s)"
            % (RATINGS, ratings_sha))

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
        dest = LOCK_DIR / "superseded" / ("Gemma_Figure_8_%s.png" % old_sha[:12])
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

    import numpy, pandas, matplotlib, torch
    rec["locks"].append({
        "locked_file": LOCKED_NAME,
        "figure": "Figure 5",
        "name": png.name,
        "sha256": png_sha,
        "producer": "render_parallel_fig8_russell_mehrabian.py",
        "producer_sha256": prod_sha,
        "figstyle_sha256": style_sha,
        "basis_source": "emotion_vectors_171.pt",
        "basis_sha256": vec_sha,
        "locked_date_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "approved_by": (
            "Locked from the canonical basis (sha 6de2e293) with a byte-reproducible "
            "re-render at lock execution (Figure 5 -> Gemma_Figure_8.png). These bytes "
            "(%s…) were first computed for this lock and have no prior independent "
            "record -- the assurance is hash-gated basis + byte-reproducible re-render, "
            "NOT a second party's independent reproduction." % png_sha[:12]),
        "status": "LOCKED",
        "superseded_prior": superseded_rel,
        "superseded_chain": chain,
        "companion_files": [
            "docs/e1_figures/E1_Figure_5_human_norms.pdf",
            "geometry_out/fig8_pca_vs_russell_mehrabian_summary.json",
            "geometry_out/fig8_pca_vs_russell_mehrabian_audit.txt",
            "geometry_out/fig8_pca_vs_russell_mehrabian_points.csv",
        ],
        "companion_sha256": {
            "E1_Figure_5_human_norms.pdf": pdf_sha,
            "fig8_pca_vs_russell_mehrabian_summary.json":
                sha256_file("geometry_out/fig8_pca_vs_russell_mehrabian_summary.json"),
            "fig8_pca_vs_russell_mehrabian_audit.txt":
                sha256_file("geometry_out/fig8_pca_vs_russell_mehrabian_audit.txt"),
            "fig8_pca_vs_russell_mehrabian_points.csv":
                sha256_file("geometry_out/fig8_pca_vs_russell_mehrabian_points.csv"),
        },
        "inputs_sha256": {
            "data/vectors/emotion_vectors_171.pt": vec_sha,
            "data/ratings/russell_mehrabian_1977.csv": ratings_sha,
        },
        "displayed_values": {"pc1_pleasure_r": 0.72, "pc2_arousal_r": 0.67,
                             "pc1_var_pct_printed": 27, "pc2_var_pct_printed": 13},
        "notes": (
            "Two scatter panels: global PCA over all 171 CLEANED L31 vectors "
            "(blob['vectors'][:, 31, :], never vectors_raw), sliced to the 45 emotions "
            "overlapping Russell & Mehrabian (1977), PC1 vs human Pleasure and PC2 vs human "
            "Arousal. Validates PC1<->pleasure r = 0.7203121 (displays 0.72) and "
            "PC2<->arousal r = 0.6675247 (displays 0.67); cross-terms PC1<->arousal -0.125193, "
            "PC2<->pleasure -0.014593; PCA variance 26.73 / 13.38 / 11.60 (same PCA object as "
            "Figures 4 and 7). PC scores are z-scored for display only — correlations are "
            "identical raw vs z-scored, asserted to 1e-12. PC signs oriented to positive "
            "correlation with their human axis. DETERMINISTIC and partition-free: SVD only, no "
            "UMAP, no k-means, no seed, no GPU, no API — PNG and PDF both byte-reproducible "
            "(PDF CreationDate pinned so it can be byte-compared, the Fig 2 gotcha). "
            "The producer is hash-gated on sha256 6de2e293 because the filename "
            "alone cannot identify the vectors, and the basis label it stamps is derived from "
            "that hash rather than asserted by the caller. Gates at render, all hard failures: "
            "full-precision correlations match geometry_out/s5_supplement.json `rm` "
            "to 1e-6 (the rounded 0.72/0.67 alone would not separate this basis from a near "
            "neighbour); PCA variance matches to 0.05; "
            "n_overlap == 45; and no plotted point falls outside the inherited hardcoded axis "
            "limits, which were tuned on an earlier basis — verified, not assumed. Human norms "
            "input is hash-pinned to the canonical table (b8a0e952…) per PROV.7(c); the "
            "corrupted …_DISCARD_BAD.csv variant was deleted and must never be used. "
            "PRODUCER AUTHORITY: the ratings PROVENANCE "
            "file long named 'fig8_rebuild_locked_audit.py' as this figure's producer. No such "
            "file has ever existed in this repo; it is a documentation artifact and carries "
            "no authority. "
            "render_parallel_fig8_russell_mehrabian.py is authoritative, as LOCK_RECORD has "
            "said throughout. Rendered at numpy %s / pandas %s / matplotlib %s / torch %s. "
            "Supersedes the prior lock f947f17a (0.71015 / 0.63023, dead under the canonical-only "
            "policy); the stale earlier summary/audit that carried those values, plus a self-check "
            "whose target was the older 0.71/0.62, are archived under "
            "geometry_out/_superseded_fig8_20260817/."
            % (numpy.__version__, pandas.__version__, matplotlib.__version__, torch.__version__)),
    })
    LOCK_RECORD.write_text(json.dumps(rec, indent=2))
    print("[updated] %s" % LOCK_RECORD)


if __name__ == "__main__":
    main()
