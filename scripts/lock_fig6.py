#!/usr/bin/env python3
"""
Lock Figure 6 (canonical circumplex PCA scatter) into locked_figures_and_tables/ as
Gemma_Figure_57.png.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/lock_fig6.py

The landmark set is finalized and the figure is locked from the canonical basis.

Figure 6 was locked after an independent re-derivation of the PCA, coordinates,
orientation, and captions from the canonical vectors (6de2e293). The assurance here is hash-gated basis + byte-reproducible
re-render at lock execution + author instruction, exactly the standing of the
Figure 2 and Figure 4 locks. Figure 3 remains the only canonical lock with genuine
independent corroboration. This is stated in the entry so the record cannot overclaim.

LOCK_RECORD.json is re-read at
execution and the outgoing locked file is versioned into superseded/, never
overwritten.
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
LOCKED_NAME = "Gemma_Figure_57.png"
PNG = Path("docs/e1_figures/E1_Figure_6_circumplex.png")
AUDIT = PNG.with_suffix(".audit.json")
CAPTION = Path("docs/e1_figures/E1_Figure_6_caption.md")
MANIFEST = Path("docs/e1_figures/FIGURES_MANIFEST.json")
PRODUCER = "repo/github_repo/scripts/render_parallel_fig57_circumplex.py"
FIGSTYLE = "repo/github_repo/scripts/figstyle.py"
VECTORS = "data/vectors/emotion_vectors_171.pt"
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
EXPECTED_PNG_SHA = "16e19c39d574cd3414b662552653b086febae85c9d47a679c8394996b094a2e7"
SUPERSEDED_SHA12 = "768cb525360f"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def die(msg: str):
    sys.exit("FATAL: " + msg)


def main() -> None:
    for p in (LOCK_RECORD, MANIFEST, PNG, AUDIT, CAPTION, PRODUCER, FIGSTYLE, VECTORS):
        if not Path(p).exists():
            die("missing %s" % p)

    vec_sha = sha256_file(VECTORS)
    if vec_sha != EXPECTED_VEC_SHA:
        die("%s is not the canonical basis (sha %s)" % (VECTORS, vec_sha))
    print("[verify] vectors are the canonical basis (%s)" % vec_sha[:12])

    png_sha = sha256_file(PNG)
    if png_sha != EXPECTED_PNG_SHA:
        die("source PNG sha %s != the confirmed sha %s" % (png_sha, EXPECTED_PNG_SHA))
    print("[verify] source PNG sha matches the confirmed render (%s)" % png_sha[:12])

    audit = json.loads(AUDIT.read_text())
    if audit.get("source_vectors_sha256") != vec_sha:
        die("audit JSON records a different source basis")
    if not audit["landmarks"]["swaps_status"].startswith("APPROVED"):
        die("landmark swaps are not marked approved: %s"
            % audit["landmarks"]["swaps_status"])
    evr = audit["explained_variance_percent"]
    orient = audit["orientation"]
    print("[verify] audit basis + approved landmark swaps (gate clean)")

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

    # Re-read at execution so a concurrent write is versioned, never overwritten.
    rec = json.loads(LOCK_RECORD.read_text())
    cur = LOCK_DIR / LOCKED_NAME
    superseded_rel = None
    if cur.exists():
        old_sha = sha256_file(cur)
        if old_sha == png_sha:
            die("%s already holds these exact bytes — nothing to lock" % LOCKED_NAME)
        if not old_sha.startswith(SUPERSEDED_SHA12):
            die("outgoing %s is sha %s, not the expected superseded render %s… — another "
                "session may have changed it; stop and reconcile before locking"
                % (LOCKED_NAME, old_sha[:12], SUPERSEDED_SHA12))
        dest = LOCK_DIR / "superseded" / ("Gemma_Figure_57_%s.png" % old_sha[:12])
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

    env = audit["env"]
    lm = audit["landmarks"]
    swaps_txt = ", ".join("%s->%s" % (k, v) for k, v in lm["swaps"].items())
    rec["locks"].append({
        "locked_file": LOCKED_NAME,
        "figure": "Figure 6",
        "name": PNG.name,
        "sha256": png_sha,
        "producer": "render_parallel_fig57_circumplex.py",
        "producer_sha256": prod_sha,
        "figstyle_sha256": style_sha,
        "basis_source": "emotion_vectors_171.pt",
        "basis_sha256": vec_sha,
        "locked_date_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "approved_by": (
            "Locked from the canonical basis (sha 6de2e293). "
            "CONFIRMATION: Figure 6 locked after independent re-derivation of the "
            "PCA/coords/orientation/captions from 6de2e293. Assurance here is "
            "hash-gated basis + byte-reproducible re-render at lock execution + author "
            "instruction - the same standing as the Figure 2 and Figure 4 locks. "
            "Figure 3 remains the only canonical lock independently corroborated."),
        "status": "LOCKED",
        "superseded_prior": superseded_rel,
        "superseded_chain": chain,
        "companion_files": ["docs/e1_figures/" + AUDIT.name,
                            "docs/e1_figures/" + CAPTION.name],
        "companion_sha256": {AUDIT.name: sha256_file(AUDIT),
                             CAPTION.name: sha256_file(CAPTION)},
        "explained_variance_percent": evr,
        "landmark_swaps": lm["swaps"],
        "notes": (
            "Figure 6, parallel to Anthropic Figure 57. All 171 emotions projected "
            "onto PC1/PC2 of a global PCA over the CLEANED L31 vectors "
            "(blob['vectors'][:, 31, :], never vectors_raw), z-scored, 25 landmark "
            "words at a uniform up-right offset (dx = dy = 0.0432). Same PCA as E1 "
            "Figures 4 and 8. Deterministic (SVD, no seed; signs fixed by anchor "
            "groups) - byte-reproducible on re-render. "
            "The vectors are identified by sha256, not by the filename; the "
            "producer is hash-gated because the filename alone cannot identify "
            "the vectors. "
            "AXIS CAPTIONS ARE COMPUTED, NOT HARDCODED: PC1 %.4f%% prints '27%% "
            "variance', PC2 %.4f%% prints '13%%', PC3 %.4f%%. The earlier producer "
            "hardcoded '26%%'; on canonical that would have contradicted locked Figure 4, "
            "which auto-computes 27%% over the same PCA. Render gate hard-exits unless "
            "PC1/PC2 match the confirmed values within 0.05 (a wrong-basis load reads "
            "26.07/13.35). "
            "ORIENTATION verified against full ranks, not the sparse printed labels: "
            "flips PC1=%s PC2=%s, anchor margins PC1 %+.4f PC2 %+.4f; poles "
            "optimistic/kind/cheerful vs terrified/hysterical/frightened (PC1) and "
            "playful/outraged/indignant vs nostalgic/serene/depressed (PC2), matching "
            "P1.1. "
            "AXIS LIMITS retuned for canonical: ymax 2.2 -> 2.5, because 'playful' sits at "
            "pc2 2.2409 and fell outside the earlier limits entirely - dot and label both "
            "clipped. A render gate hard-exits if any of the 171 points falls outside "
            "x %s / y %s. "
            "LANDMARK SWAPS RECOMPUTED ON THE canonical POSITIONS: %s. The earlier swaps (vengeful->spiteful, compassionate->content, "
            "happy->delighted, approved 2026-08-07 against the earlier dot positions) do "
            "NOT transfer: carried forward blindly they leave four word-word "
            "collisions standing, and compassionate->content makes its own pair worse "
            "(73.5px overlap vs 16.4px unswapped). The four collisions actually "
            "present on canonical are compassionate/sympathetic, excited/vengeful, "
            "sentimental/serene and sluggish/tormented, each resolved one-for-one by "
            "the nearest same-region word. A render gate measures every label's pixel "
            "box and hard-exits on any word-word overlap; tightest surviving "
            "separation 7.3px (excited/thrilled). "
            "INTENTIONALLY PARTITION-FREE - no k=10 family colouring; pure PCA "
            "geometry. Rendered at numpy %s / matplotlib %s / scikit-learn %s. "
            "Supersedes the prior lock %s (dead under the single-basis policy)."
            % (evr["PC1"], evr["PC2"], evr["PC3"],
               orient["flip_pc1"], orient["flip_pc2"],
               orient["anchor_margin_pc1"], orient["anchor_margin_pc2"],
               audit["axis_limits"]["xlim"], audit["axis_limits"]["ylim"],
               swaps_txt, env["numpy"], env["matplotlib"], env["scikit_learn"],
               SUPERSEDED_SHA12)),
    })
    LOCK_RECORD.write_text(json.dumps(rec, indent=2))
    print("[updated] %s" % LOCK_RECORD)


if __name__ == "__main__":
    main()
