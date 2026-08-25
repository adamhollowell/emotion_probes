#!/usr/bin/env python3
"""
Lock Figure 7 (canonical cross-layer RSA heatmap) into locked_figures_and_tables/ as
Gemma_Figure_9.png.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/lock_fig7.py

Figure 7 was locked after an independent re-derivation of the RSA matrix from the
canonical vectors (6de2e293; sampled layers, cosine-of-cosine, adjacent-min, diagonal).
The assurance here is
hash-gated basis + a render-time content gate (L31 cosine spot-check + RSA diagonal)
+ byte-reproducible re-render at lock execution + author instruction. This is the
standing of the Figure 2, 4 and 6 locks; Figure 3 remains the only canonical lock with
genuine independent corroboration. Stated in the entry so the record cannot overclaim.

LOCK_RECORD.json is re-read at
execution and the outgoing locked file is versioned into superseded/, never
overwritten. An existing supersession (f0be16fc) is preserved in the chain.
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
LOCKED_NAME = "Gemma_Figure_9.png"
PNG = Path("docs/e1_figures/E1_Figure_7_crosslayer_rsa.png")
AUDIT = PNG.with_suffix(".audit.json")
MANIFEST = Path("docs/e1_figures/FIGURES_MANIFEST.json")
PRODUCER = "repo/github_repo/scripts/render_parallel_fig9_cross_layer_rsa.py"
FIGSTYLE = "repo/github_repo/scripts/figstyle.py"
VECTORS = "data/vectors/emotion_vectors_171.pt"
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
EXPECTED_PNG_SHA = "296f798516881cf10cba69c0f98432f984c8413ee594db36723df4f7dd738ba5"
SUPERSEDED_SHA12 = "eca93a9a1ac2"


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
    print("[verify] vectors are the canonical basis (%s)" % vec_sha[:12])

    png_sha = sha256_file(PNG)
    if png_sha != EXPECTED_PNG_SHA:
        die("source PNG sha %s != the confirmed sha %s" % (png_sha, EXPECTED_PNG_SHA))
    print("[verify] source PNG sha matches the confirmed render (%s)" % png_sha[:12])

    audit = json.loads(AUDIT.read_text())
    if audit.get("source_vectors_sha256") != vec_sha:
        die("audit JSON records a different source basis")
    cg = audit.get("content_gate") or die("audit JSON has no content_gate block")
    if not (0.9999 <= cg["rsa_diagonal_min"] and cg["rsa_diagonal_max"] <= 1.0001):
        die("audit RSA diagonal not ~1.0 (%s..%s)"
            % (cg["rsa_diagonal_min"], cg["rsa_diagonal_max"]))
    adj = audit.get("adjacent_pairs") or die("audit JSON has no adjacent_pairs block")
    if not adj.get("min_in_seam_region_L22_26"):
        die("audit adjacent-min is not in the L22-26 seam region — refusing to lock")
    print("[verify] audit basis + content gate (diagonal ~1, seam min L%d->L%d) clean"
          % (adj["min_layer_pair"][0], adj["min_layer_pair"][1]))

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
        dest = LOCK_DIR / "superseded" / ("Gemma_Figure_9_%s.png" % old_sha[:12])
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
    spot = cg["spot_cosines"]
    rec["locks"].append({
        "locked_file": LOCKED_NAME,
        "figure": "Figure 7",
        "name": PNG.name,
        "sha256": png_sha,
        "producer": "render_parallel_fig9_cross_layer_rsa.py",
        "producer_sha256": prod_sha,
        "figstyle_sha256": style_sha,
        "basis_source": "emotion_vectors_171.pt",
        "basis_sha256": vec_sha,
        "locked_date_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "approved_by": (
            "Locked from the canonical basis (sha 6de2e293) after an independent "
            "re-derivation of the RSA matrix (sampled layers, cosine-of-cosine, "
            "adjacent-min, diagonal). Assurance here is hash-gated basis + a "
            "render-time content gate (L31 cosine spot-check + RSA diagonal ~1) + "
            "byte-reproducible re-render at lock execution. No independent second-party "
            "reproduction is on record."),
        "status": "LOCKED",
        "superseded_prior": superseded_rel,
        "superseded_chain": chain,
        "companion_files": ["docs/e1_figures/" + AUDIT.name],
        "companion_sha256": {AUDIT.name: sha256_file(AUDIT)},
        "sampled_layers": audit["sampled_layers"],
        "adjacent_min": {"value": adj["min_value"], "layer_pair": adj["min_layer_pair"]},
        "notes": (
            "Figure 7, parallel to Anthropic Figure 9. Finding P3.1 (cross-layer "
            "representational stability). Layer x layer cosine-of-cosine RSA between "
            "the CLEANED emotion-probe geometries (blob['vectors'], never vectors_raw) "
            "at 14 evenly spaced layers np.linspace(12,40).round() = %s (includes L31). "
            "Per layer: upper triangle of the 171x171 pairwise-cosine matrix, "
            "L2-normalised, then cosine similarity across layers. viridis vmin 0.8 vmax "
            "1.0; Early/Early-Mid/Mid-Late/Late band labels (matches Anthropic; the "
            "numeric-layer mode is a blog/diagnostic variant, P3.8, not this figure). "
            "METHOD UNCHANGED from the earlier version - Anthropic-faithful; the Pearson variant differs "
            "only ~1e-5 and is not used. Deterministic (pure geometry, no seed) - "
            "byte-reproducible on re-render. "
            "The vectors are identified by sha256, not by the filename; the "
            "producer is hash-gated because the filename alone cannot identify "
            "the vectors. "
            "figstyle is a HARD import here: the earlier producer wrapped it in a try/except "
            "that silently fell back to different fonts + a hardcoded rust colour on "
            "import failure (a silent different-bytes path), removed in the canonical "
            "rebuild. CONTENT GATE at render: L31 cosines afraid/scared %.4f, "
            "joyful/happy %.4f, compassionate/empathetic %.4f (same spot-check as "
            "Figure 2, tol 0.002); RSA diagonal ~1.0 (%.6f..%.6f). "
            "SEAM: adjacent-pair minimum %.4f at L%d->L%d, inside the L22-26 emergence "
            "seam - the sampled-sweep echo of P3.1. DIFFERENT cut from the "
            "full-46-layer adjacent steps in adjacent_rsa.json (true L21->L22 "
            "0.9425, L43->L44 0.9329): same story, different numbers. The figure shows "
            "the L22-26 seam -> L27-40 stable plateau; it does NOT show the final-layer "
            "drop (L41-45), which is outside the 12-40 sampling window exactly as in "
            "Anthropic's figure - do not caption a 'final drop' here (that lives in the "
            "RSA-to-L31 trajectory, P3.1 text). INTENTIONALLY PARTITION-FREE. Rendered "
            "at numpy %s / matplotlib %s. Supersedes the prior lock %s (dead under the "
            "single-basis policy); the earlier f0be16fc supersession is retained in "
            "superseded_chain."
            % (audit["sampled_layers"], spot["afraid|scared"], spot["joyful|happy"],
               spot["compassionate|empathetic"], cg["rsa_diagonal_min"],
               cg["rsa_diagonal_max"], adj["min_value"], adj["min_layer_pair"][0],
               adj["min_layer_pair"][1], env["numpy"], env["matplotlib"],
               SUPERSEDED_SHA12)),
    })
    LOCK_RECORD.write_text(json.dumps(rec, indent=2))
    print("[updated] %s" % LOCK_RECORD)


if __name__ == "__main__":
    main()
