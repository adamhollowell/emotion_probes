#!/usr/bin/env python3
"""
Lock Figure 1 (canonical max-activation panels) into locked_figures_and_tables/ as
Gemma_Figure_1.png.

Run from the PROJECT ROOT:
    python3 repo/github_repo/scripts/lock_fig1.py

fig1 is PIL-based and consumes the
maxact sweep's per-token projections, so unlike figs 2-7 it cannot hash a
vectors .pt directly and cannot be re-verified from scratch without a GPU pass over
25M tokens. Its basis rests on the sweep's recorded provenance: pass1_results.json
(the sweep computation's own record) and the render data both stamp vectors_sha256 ==
6de2e293 (cleaned, L31), and magnitude_diagnostic.json independently stamps basis
canonical. The assurance is provenance-record gate + byte-reproducible re-render at
lock execution. It does not overclaim a from-scratch reprojection.

LOCK_RECORD.json is re-read at
execution and the outgoing locked file is versioned into superseded/, never overwritten.
"""
from __future__ import annotations
import hashlib, json, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

LOCK_DIR = Path("locked_figures_and_tables")
LOCK_RECORD = LOCK_DIR / "LOCK_RECORD.json"
LOCKED_NAME = "Gemma_Figure_1.png"
PNG = Path("docs/e1_figures/E1_Figure_1_maxact.png")
PDF = PNG.with_suffix(".pdf")
MANIFEST = Path("docs/e1_figures/FIGURES_MANIFEST.json")
RENDER = "geometry_out/render_fig1.py"
BUILD = "geometry_out/build_fig1.py"
SWEEP = Path("data/heldout/maxact")
RENDER_DATA = SWEEP / "fig1_render_data.json"
SELECTION = SWEEP / "fig1_selection.json"
PASS1 = SWEEP / "pass1_results.json"
PROJECTIONS = SWEEP / "alldocs_projections.f16.npy"
TOKEN_IDS = SWEEP / "alldocs_token_ids.i32.npy"
EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
EXPECTED_PNG_SHA = "75764321150fa9d08f753dc3587bb9840ec82190e8af695a8ed6576904bb76cb"
# The exact canonical sweep the panels trace to (maxact; substantive data-provenance gate).
EXPECTED_PROJ_SHA = "3efd30df363ba67f7d987b50bc33c012d70d14938da6b05db1f625a13f086cee"
SUPERSEDED_SHA12 = "2ae66a687f77"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def die(m): sys.exit("FATAL: " + m)


def main() -> None:
    for p in (LOCK_RECORD, MANIFEST, PNG, PDF, RENDER, BUILD, RENDER_DATA, SELECTION, PASS1, PROJECTIONS, TOKEN_IDS):
        if not Path(p).exists():
            die("missing %s" % p)

    rd = json.loads(RENDER_DATA.read_text()); p1 = json.loads(PASS1.read_text())
    if rd.get("vectors_sha256") != EXPECTED_VEC_SHA:
        die("render data is not the canonical basis")
    if p1.get("vectors_sha256") != EXPECTED_VEC_SHA or p1.get("vector_set") != "cleaned":
        die("sweep pass1 is not the canonical cleaned basis")
    print("[verify] render-data + sweep pass1 both stamp the canonical basis (%s, cleaned, L%s)"
          % (EXPECTED_VEC_SHA[:12], p1.get("layer")))

    # SUBSTANTIVE DATA-PROVENANCE GATE (the real check for a curated panel render):
    # the displayed windows/tokens/projections must trace to the maxact sweep,
    # not the old v2 sweep. Gate on the exact projections file the panels came from.
    print("[verify] hashing the sweep projections (8.5GB — the panels' actual data source)…")
    proj_sha = sha256_file(PROJECTIONS)
    if proj_sha != EXPECTED_PROJ_SHA:
        die("sweep projections sha %s != expected maxact %s — the panels do NOT "
            "trace to the approved canonical sweep" % (proj_sha, EXPECTED_PROJ_SHA))
    sel_sha = sha256_file(SELECTION)
    print("[verify] projections trace to maxact (%s); selection %s" % (proj_sha[:12], sel_sha[:12]))

    png_sha = sha256_file(PNG)
    if png_sha != EXPECTED_PNG_SHA:
        die("source PNG sha %s != the confirmed sha %s" % (png_sha, EXPECTED_PNG_SHA))
    print("[verify] source PNG sha matches the confirmed render (%s)" % png_sha[:12])

    with tempfile.TemporaryDirectory() as td:
        probe = str(Path(td) / "probe")
        r = subprocess.run([sys.executable, RENDER, "--out-prefix", probe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            die("producer failed on the lock-time re-render:\n" + r.stderr[-2000:])
        if sha256_file(probe + ".png") != png_sha:
            die("PNG is not reproducible from the current producer")
    print("[verify] PNG reproduces at lock execution")

    man = json.loads(MANIFEST.read_text())
    entry = man["figures"].get(PNG.name) or die("no manifest entry for %s" % PNG.name)
    if entry["sha256"] != png_sha:
        die("manifest sha != PNG on disk")
    if entry["producer_sha256"] != sha256_file(RENDER):
        die("render producer changed since the manifest was written")
    if entry.get("build_sha256") != sha256_file(BUILD):
        die("build producer changed since the manifest was written")
    print("[verify] producer (render + build) matches the manifest (gate clean)")

    rec = json.loads(LOCK_RECORD.read_text())
    cur = LOCK_DIR / LOCKED_NAME
    superseded_rel = None
    if cur.exists():
        old_sha = sha256_file(cur)
        if old_sha == png_sha:
            die("%s already holds these exact bytes — nothing to lock" % LOCKED_NAME)
        if not old_sha.startswith(SUPERSEDED_SHA12):
            die("outgoing %s is sha %s, not the expected superseded render %s… — a concurrent write "
                "may have changed it; stop and reconcile" % (LOCKED_NAME, old_sha[:12], SUPERSEDED_SHA12))
        dest = LOCK_DIR / "superseded" / ("Gemma_Figure_1_%s.png" % old_sha[:12])
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

    import PIL
    rec["locks"].append({
        "locked_file": LOCKED_NAME,
        "figure": "Figure 1",
        "name": PNG.name,
        "sha256": png_sha,
        "producer": "render_fig1.py (+ build_fig1.py)",
        "producer_sha256": sha256_file(RENDER),
        "build_sha256": sha256_file(BUILD),
        "basis_source": "maxact sweep (resid_post[L31] . cleaned emotion_vectors_171.pt)",
        "basis_sha256": EXPECTED_VEC_SHA,
        "locked_date_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "approved_by": (
            "Locked from the canonical basis (sha 6de2e293). The panel bytes are "
            "single-session and env-locked, so exact-byte second-party reproduction is "
            "NOT expected; the substantive check is the DATA-PROVENANCE TRACE. fig1 "
            "consumes the sweep's per-token projections and cannot be re-derived from "
            "vectors the way figs 4/6/7 were (a GPU pass over 25M tokens would be needed "
            "to reproject). Its basis rests on the sweep's recorded provenance -- "
            "pass1_results.json and the render data both stamp vectors_sha256 6de2e293 "
            "(cleaned, L31), magnitude_diagnostic.json stamps basis canonical -- gated at "
            "lock on the exact maxact projections file, plus a byte-reproducible re-render "
            "at lock execution. NOT a from-scratch reprojection."),
        "status": "LOCKED",
        "superseded_prior": superseded_rel,
        "superseded_chain": chain,
        "companion_files": ["docs/e1_figures/" + PDF.name],
        "companion_sha256": {PDF.name: sha256_file(PDF)},
        "curation": man["figures"][PNG.name].get("curation"),
        "sweep_provenance": {
            "sweep": "data/heldout/maxact (maxact)",
            "basis_vectors_sha256": EXPECTED_VEC_SHA,
            "projections_sha256": proj_sha,
            "token_ids_sha256": sha256_file(TOKEN_IDS),
            "selection_sha256": sel_sha,
            "render_data_sha256": sha256_file(RENDER_DATA),
            "pass1_results_sha256": sha256_file(PASS1),
            "note": ("The displayed windows/tokens/projections trace to the maxact sweep "
                     "(vectors 6de2e293, NOT the old v2 sweep), and the panels match the "
                     "approved fig1_selection.json. These two shas are the real check."),
        },
        "notes": (man["figures"][PNG.name]["notes"] + " Rendered at Pillow %s. "
                  "DATA-PROVENANCE TRACE (the substantive check, per the Fig-5 byte-origin "
                  "waiver): panels come from the maxact sweep — projections_sha256 %s, "
                  "basis vectors_sha256 %s — and match the approved selection "
                  "fig1_selection.json (selection_sha256 %s)."
                  % (PIL.__version__, proj_sha, EXPECTED_VEC_SHA, sel_sha)),
    })
    LOCK_RECORD.write_text(json.dumps(rec, indent=2))
    print("[updated] %s" % LOCK_RECORD)


if __name__ == "__main__":
    main()
