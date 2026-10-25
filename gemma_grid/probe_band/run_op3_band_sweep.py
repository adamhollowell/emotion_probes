#!/usr/bin/env python3
"""OP.3 band sweep (GGE1-G3, pod P) — fresh base-27B extraction at the full band
{31, 42, 43, 44, 45}, two seeds, per OP3_BAND_SWEEP_SPEC.md.

Producer ONLY: extraction + gates. All graded analysis (per-layer ceilings
P(l)=sqrt(R_ff), attenuation-rule comparisons vs locked, cliff-vs-slope,
dimension attribution) is post-session Grid-side per the no-self-grading rule.

Reuses run_probe_l45.py's verbatim pipeline by IMPORT (loader regime, sampler,
pooling path) — the L45-probe-validated code path, extended only in the layer
list. Gates, in order:
  1. fingerprint gate: resolved HF SHA == hf_model_revisions_VERIFIED.json pin
     for google/gemma-2-27b, cross-checked against the probe-era constant.
  2. sampler reproduction gate (pre-GPU): this run's seed-0 story indices must
     equal the stored probe draw (probe_l45_means.pt::story_indices) EXACTLY.
  3. overlap-layer control (post-extraction, per seed): fresh means at the
     overlapping layers {31, 42, 45} vs the stored probe means for that seed —
     same stories, same pipeline, so rel <= 1e-3 required (expect ~1e-6
     same-GPU-class; M.5 cross-hardware calibration if the class differs, which
     must then be REPORTED prominently). FAIL -> STOP.

RUN ORDER (pod P):
    python gemma_grid/probe_band/run_op3_band_sweep.py --pilot        # show Adam
    python gemma_grid/probe_band/run_op3_band_sweep.py --seed 0
    python gemma_grid/probe_band/run_op3_band_sweep.py --seed 1

Outputs (NOTE to coordination: these names supersede the spec's per-layer
`means_L{31..45}` sketch — one file per seed, probe schema, layers axis):
    gemma_grid/probe_band/probe_band_means_seed0.pt   (171, 5, 4608) f32 + meta
    gemma_grid/probe_band/probe_band_means_seed1.pt
    gemma_grid/probe_band/probe_band_means_PILOT.pt   (pilot flag)
    gemma_grid/probe_band/run_op3_band_sweep_manifest_seed{N}.json  (SR.5)
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from gemma_grid.probe_l45.run_probe_l45 import load_model_pinned, build_probe_groups  # noqa: E402
from gemma_grid.probe_l45._sampling import PROBE_SEED, N_STORIES                      # noqa: E402
from gemma_grid.scripts.run_cell_extraction import extract_story_slice                # noqa: E402

MODEL = "google/gemma-2-27b"
PROBE_ERA_SHA = "938270f5272feb02779b55c2bb2fffdd0f53ff0c"   # cross-check constant
BAND_LAYERS = [31, 42, 43, 44, 45]                            # OP.3 band; 31 = control
OVERLAP_LAYERS = [31, 42, 45]                                 # shared with the L45 probe
SKIP_FIRST_N = 50
OUT_DIR = REPO / "gemma_grid" / "probe_band"
STORED_MEANS = {0: REPO / "gemma_grid/probe_l45/probe_l45_means.pt",
                1: REPO / "gemma_grid/probe_l45/probe_l45_means_seed1.pt"}
VERIFIED_PINS = REPO / "gemma_grid/handoff/hf_model_revisions_VERIFIED.json"
STORIES = REPO / "data/stories/stories_171_canonical_clean.jsonl"

MANIFEST: dict[str, str] = {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    MANIFEST[str(p.relative_to(REPO))] = h.hexdigest()
    return h.hexdigest()


def fingerprint_gate() -> str:
    from huggingface_hub import HfApi
    sha256(VERIFIED_PINS)
    pins = json.loads(VERIFIED_PINS.read_text())
    expected = pins[MODEL]["current_main_sha"]
    assert expected == PROBE_ERA_SHA, \
        f"VERIFIED pin {expected[:12]} != probe-era constant {PROBE_ERA_SHA[:12]} — STOP, report"
    resolved = HfApi().model_info(MODEL).sha
    print(f"[fingerprint] resolved={resolved}\n[fingerprint] expected={expected}")
    if resolved != expected:
        print("[fingerprint] MISMATCH — STOP. Report to Adam.")
        sys.exit(2)
    print("[fingerprint] PASS — pinning explicitly.")
    return resolved


def sampler_reproduction_gate(groups: dict, seed: int) -> dict:
    """Pre-GPU: seed-N draw must equal the stored probe draw exactly."""
    ref_path = STORED_MEANS[seed]
    sha256(ref_path)
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_idx = ref["story_indices"]
    ours = {e: [gi for gi, _ in members] for e, members in groups.items()}
    if ours != ref_idx:
        bad = [e for e in ours if ours.get(e) != ref_idx.get(e)][:5]
        sys.exit(f"[sampler gate] seed-{seed} indices differ from stored probe draw "
                 f"(first mismatches: {bad}) — STOP (spec: must reproduce exactly)")
    print(f"[sampler gate] PASS — seed-{seed} draw identical to stored probe draw "
          f"({len(ours)} emotions x {N_STORIES})")


def overlap_control_gate(fresh_means: torch.Tensor, emotions: list, seed: int, tol: float = 1e-3):
    """Post-extraction: fresh {31,42,45} vs stored probe means, same stories."""
    ref = torch.load(STORED_MEANS[seed], map_location="cpu", weights_only=False)
    assert ref["layers"] == OVERLAP_LAYERS, f"stored layers {ref['layers']} != {OVERLAP_LAYERS}"
    assert ref["emotions"] == emotions, "emotion order differs from stored probe file — STOP"
    band_pos = [BAND_LAYERS.index(l) for l in OVERLAP_LAYERS]
    worst = 0.0
    for rpos, bpos, layer in zip(range(3), band_pos, OVERLAP_LAYERS):
        g = ref["means"][:, rpos, :].float()
        f = fresh_means[:, bpos, :].float()
        rel = ((f - g).norm() / g.norm().clamp_min(1e-12)).item()
        worst = max(worst, rel)
        print(f"[overlap control] seed{seed} L{layer}: rel={rel:.2e}")
    if worst > tol:
        sys.exit(f"[overlap control] FAIL — worst rel {worst:.2e} > {tol:g}. If the GPU "
                 "class differs from H100, apply M.5 calibration and REPORT; else STOP.")
    print(f"[overlap control] PASS (worst rel {worst:.2e}, tol {tol:g})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true", help="3 emotions x 5 stories sanity run")
    ap.add_argument("--seed", type=int, default=PROBE_SEED, choices=(0, 1))
    a = ap.parse_args()
    lim_e, lim_s = (3, 5) if a.pilot else (None, None)

    sha256(STORIES)
    revision = fingerprint_gate()

    groups, sampling_meta = build_probe_groups(a.seed, lim_e, lim_s)
    print(f"sampling: {sampling_meta}")
    if not a.pilot:
        sampler_reproduction_gate(groups, a.seed)

    print(f"Loading {MODEL} (compat=False, bf16, revision pinned)...")
    bridge = load_model_pinned(revision)
    assert max(BAND_LAYERS) < bridge.cfg.n_layers

    cfg = SimpleNamespace(skip_first_n_tokens=SKIP_FIRST_N)
    emotions, means, story_indices = [], [], {}
    for e, members in groups.items():
        idxs = [gi for gi, _ in members]
        estories = [s for _, s in members]
        sl = extract_story_slice(estories, bridge, cfg, BAND_LAYERS)   # (n, 5, 4608)
        means.append(sl.mean(dim=0))
        story_indices[e] = idxs
        emotions.append(e)
    means = torch.stack(means, dim=0).float()                          # (n_em, 5, 4608)
    print(f"extracted band means: {tuple(means.shape)}")

    if not a.pilot:
        overlap_control_gate(means, emotions, a.seed)

    prov = {
        "job": "GGE1-G3 / OP.3 band sweep (pod P)",
        "hf_revision_resolved": revision,
        "torch": torch.__version__,
        "tl_compatibility_mode": False, "dtype": "bfloat16",
        "skip_first_n_tokens": SKIP_FIRST_N,
        "pod_gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(), "pilot": bool(a.pilot), "sampling": sampling_meta,
        "pooling": "mean over resid_post from token 50 (BOS-inclusive); "
                   "identical to _mean_resid_post_single via extract_story_slice",
        "basis_note": "RAW structures only — basis-immune; valid under HA and 100T alike",
    }
    payload = {"emotions": emotions, "layers": BAND_LAYERS, "means": means,
               "story_indices": story_indices, "provenance": prov}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = "probe_band_means_PILOT.pt" if a.pilot else f"probe_band_means_seed{a.seed}.pt"
    out = OUT_DIR / name
    tmp = out.with_suffix(out.suffix + ".tmp"); torch.save(payload, tmp); tmp.rename(out)
    (OUT_DIR / f"run_op3_band_sweep_manifest_seed{a.seed}{'_PILOT' if a.pilot else ''}.json").write_text(
        json.dumps({"rule": "SR.5 — every input hashed at run time",
                    "inputs_sha256": MANIFEST}, indent=1))
    print(f"wrote {out.relative_to(REPO)} + SR.5 manifest")
    if a.pilot:
        print("\n[PILOT] shapes sane. SHOW ADAM before the full seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
