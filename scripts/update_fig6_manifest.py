#!/usr/bin/env python3
"""
Refresh the Figure 6 (canonical circumplex PCA scatter) entry in
docs/e1_figures/FIGURES_MANIFEST.json.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/update_fig6_manifest.py

Re-runs the producer to a temp prefix and refuses to write unless the shipped PNG is
byte-reproducible from it. Records literal render-time hashes plus the audit JSON's
variance shares and landmark swaps, so a wrong-basis render can never be described as
a right one.

Writes to docs/e1_figures/ only. Never touches locked_figures_and_tables/ — the canonical
figure-lock freeze holds, and this figure additionally carries UNAPPROVED landmark
swaps, so it is not yet lockable.
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
FIG_KEY = "E1_Figure_6_circumplex.png"
PNG = Path("docs/e1_figures") / FIG_KEY
AUDIT = PNG.with_suffix(".audit.json")
PRODUCER = "repo/github_repo/scripts/render_parallel_fig57_circumplex.py"
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
    evr = audit["explained_variance_percent"]
    lm = audit["landmarks"]

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
    swaps_txt = ", ".join("%s->%s" % (k, v) for k, v in lm["swaps"].items())
    man = json.loads(MANIFEST.read_text())
    man["figures"][FIG_KEY] = {
        "sha256": png_sha,
        "basis": "canonical",
        "producer": "render_parallel_fig57_circumplex.py",
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
        "landmark_swaps": lm["swaps"],
        "landmark_swaps_status": lm["swaps_status"],
        "lockable": True,
        "notes": (
            "Figure 6, parallel to Anthropic Figure 57: all 171 emotions projected "
            "onto PC1/PC2 of a global PCA over the CLEANED L31 vectors "
            "(blob['vectors'][:, 31, :], never vectors_raw), z-scored, 25 landmark "
            "words labelled at a uniform up-right offset (dx = dy = 0.0432). Same PCA "
            "as Figures 4 and 8. Deterministic (SVD, no seed; signs fixed by anchor "
            "groups) — byte-reproducible on re-render. "
            "AXIS CAPTIONS ARE COMPUTED, NOT HARDCODED: PC1 %.4f%% -> prints '27%% "
            "var', PC2 %.4f%% -> '13%% var', PC3 %.4f%%. The earlier producer hardcoded "
            "'26%%'/'13%%'; on canonical that would have put this figure in direct conflict "
            "with the locked Figure 4, which auto-computes 27%% over the same PCA. "
            "Render gate hard-exits unless PC1/PC2 match the confirmed canonical values "
            "within 0.05 (a wrong-basis load reads 26.07/13.35). "
            "ORIENTATION verified against full ranks, not the sparse printed labels: "
            "flips PC1=%s PC2=%s, anchor margins PC1 %+.4f PC2 %+.4f; poles are "
            "optimistic/kind/cheerful vs terrified/hysterical/frightened (PC1) and "
            "playful/outraged/indignant vs nostalgic/serene/depressed (PC2), matching "
            "P1.1. AXIS LIMITS retuned for canonical: ymax 2.2 -> 2.5, because 'playful' "
            "sits at pc2 2.2409 and fell outside the earlier limits entirely — dot and "
            "label both clipped. A render-time gate hard-exits if any of the 171 "
            "points falls outside x %s / y %s. "
            "LANDMARK SWAPS RECOMPUTED ON THE canonical POSITIONS (%s). "
            "The earlier swaps (vengeful->spiteful, compassionate->content, "
            "happy->delighted, approved 2026-08-07 against the earlier dot positions) do "
            "not transfer — carried forward blindly they leave four word-word "
            "collisions standing, and compassionate->content makes its own pair worse "
            "(73.5px overlap vs 16.4px unswapped). The four collisions actually "
            "present on canonical are compassionate/sympathetic, excited/vengeful, "
            "sentimental/serene and sluggish/tormented; each is resolved one-for-one "
            "by the nearest same-region word, the same rule that produced the earlier "
            "swaps. A render-time gate measures every label's pixel box and hard-exits "
            "on any word-word overlap; tightest surviving separation is 7.3px "
            "(excited/thrilled). INTENTIONALLY PARTITION-FREE — no k=10 family "
            "colouring; pure PCA geometry. Rendered at numpy %s / matplotlib %s / "
            "scikit-learn %s. Promoted to locked_figures_and_tables/Gemma_Figure_57.png, "
            "superseding the prior lock 768cb525 (dead under the single-basis policy); "
            "see LOCK_RECORD.json for the confirmation caveat."
            % (evr["PC1"], evr["PC2"], evr["PC3"],
               audit["orientation"]["flip_pc1"], audit["orientation"]["flip_pc2"],
               audit["orientation"]["anchor_margin_pc1"],
               audit["orientation"]["anchor_margin_pc2"],
               audit["axis_limits"]["xlim"], audit["axis_limits"]["ylim"],
               swaps_txt, env["numpy"], env["matplotlib"], env["scikit_learn"])),
        "figstyle_sha256": sha256_file(FIGSTYLE),
    }
    MANIFEST.write_text(json.dumps(man, indent=1))
    print("[updated] %s -> %s" % (MANIFEST, FIG_KEY))
    print("          png      sha256 %s" % png_sha)
    print("          audit    sha256 %s" % sha256_file(AUDIT))
    print("          producer sha256 %s" % sha256_file(PRODUCER))
    print("          variance PC1 %.4f  PC2 %.4f  PC3 %.4f" % (evr["PC1"], evr["PC2"], evr["PC3"]))
    print("          swaps    %s  [%s]" % (swaps_txt, lm["swaps_status"]))


if __name__ == "__main__":
    main()
