#!/usr/bin/env python3
"""
L45 probe — fresh base-27B extraction at layers {31,42,45} to settle whether
base-27B's L45 cosine-structure divergence is GENUINE or a round-one vintage
artifact. See gemma_grid/PROBE_L45_CLAUDE_CODE_HANDOFF.md.

RUN ORDER (gate discipline — do NOT skip the pilot; §7 of the handoff):
    # dry-run first (no GPU): gemma_grid/probe_l45/dry_run_probe_l45.py
    python gemma_grid/probe_l45/run_probe_l45.py --pilot      # 3 emotions x 5 stories -> show Adam
    python gemma_grid/probe_l45/run_probe_l45.py              # full 171 x 120

Reuses the grid pipeline verbatim: model loader regime + the exact pooling
(emotion_probes.activations._mean_resid_post_single, via the mode-B
extract_story_slice helper — confirmed bit-identical). No neutrals, no cleanup,
no reproduction-gate. RAW vectors only (anomaly is larger on raw).
"""
from __future__ import annotations
import argparse, json, sys, platform
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from emotion_probes.generation import Story, load_stories            # noqa: E402
from emotion_probes.emotions import EMOTIONS_171                      # noqa: E402
from gemma_grid.scripts.run_cell_extraction import extract_story_slice  # noqa: E402
from gemma_grid.probe_l45._sampling import probe_positions, PROBE_SEED, N_STORIES  # noqa: E402

MODEL = "google/gemma-2-27b"
EXPECTED_SHA = "938270f5272feb02779b55c2bb2fffdd0f53ff0c"   # handoff §3.1 / hf_model_revisions_ASSUMED.json
PROBE_LAYERS = [31, 42, 45]                                  # L31 pipeline control, L42 near-band, L45 target
SKIP_FIRST_N = 50                                            # position-50 pooling, BOS-inclusive (invariant §3.2)
OUT = REPO / "gemma_grid" / "probe_l45" / "probe_l45_means.pt"

# SAMPLING NOTE (deviation from handoff §3.4, approved by Adam 2026-07-27):
# §3.4 specified "first 120 in corpus order", but the corpus is topic-ordered
# (12 stories x 100 topics/emotion), so first-120 covers only ~10 topics and its
# structure-RSA to the locked L31 is 0.9154 — it would FAIL the >=0.980 pipeline
# control and VOID the probe. The §5 calibration (0.9856) was measured on RANDOM
# 120-draws. Fix: seeded-random 120-of-N per emotion (gemma_grid/probe_l45/
# _sampling.py; deterministic given the seed, representative, self-consistent
# with the calibration). Positions recorded in story_indices.


def fingerprint_gate() -> str:
    """Resolve the HF revision that will be loaded; assert it equals the
    recorded SHA. STOP on mismatch (handoff §3.1 / RUN_MANIFEST gate)."""
    from huggingface_hub import HfApi
    resolved = HfApi().model_info(MODEL).sha
    print(f"[fingerprint gate] {MODEL} resolved main SHA = {resolved}")
    print(f"[fingerprint gate] expected (recorded)       = {EXPECTED_SHA}")
    if resolved != EXPECTED_SHA:
        print("[fingerprint gate] MISMATCH — STOP. Report to Adam. (handoff §7)")
        sys.exit(2)
    print("[fingerprint gate] PASS — pinning this revision explicitly.")
    return resolved


def load_model_pinned(revision: str):
    """Grid regime, exactly: raw HF numerics (compat=False), bf16, revision
    PINNED. Mirrors emotion_probes.activations.load_model + the deliberate
    gemma-27b tl_compatibility_mode=False override."""
    from transformer_lens.model_bridge import TransformerBridge
    bridge = TransformerBridge.boot_transformers(
        MODEL, device="cuda", dtype=torch.bfloat16, revision=revision,
    )
    # tl_compatibility_mode stays FALSE (do NOT call enable_compatibility_mode) —
    # matches the anchor; see gemma_grid/CLAUDE.md extraction regime.
    bridge.eval()
    assert int(bridge.cfg.n_layers) == 46, f"n_layers={bridge.cfg.n_layers} != 46"
    assert int(bridge.cfg.d_model) == 4608, f"d_model={bridge.cfg.d_model} != 4608"
    assert max(PROBE_LAYERS) < bridge.cfg.n_layers
    return bridge


def build_probe_groups(seed=PROBE_SEED, pilot_emotions=None, pilot_stories=None):
    """Return ({emotion: [(global_idx, Story), ...]}, sampling_meta).
    Full run: seeded-random N_STORIES per emotion (representative). Pilot: first
    few (plumbing check only — representativeness irrelevant for shape sanity)."""
    stories = load_stories(REPO / "data" / "stories" / "stories_171_canonical_clean.jsonl")
    stories = [s for s in stories if getattr(s, "variant", "canonical") == "canonical"]
    by: dict[str, list[tuple[int, Story]]] = {}
    for gi, s in enumerate(stories):
        by.setdefault(s.emotion, []).append((gi, s))
    if pilot_emotions is not None:
        ems = [e for e in EMOTIONS_171 if e in by][:pilot_emotions]
        return {e: by[e][:pilot_stories] for e in ems}, {"sampling": "pilot_first_n", "seed": None}
    avail = {e: len(v) for e, v in by.items()}
    pos = probe_positions(avail, seed, N_STORIES)
    groups = {e: [by[e][p] for p in pos[e]] for e in pos}
    return groups, {"sampling": "seeded_random", "seed": seed, "n_stories": N_STORIES}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true", help="3 emotions x 5 stories sanity run")
    ap.add_argument("--seed", type=int, default=PROBE_SEED,
                    help="sampling seed (default 0 = canonical probe; use 1 for the independent second draw)")
    a = ap.parse_args()
    lim_e, lim_s = (3, 5) if a.pilot else (None, None)

    revision = fingerprint_gate()
    print(f"Loading {MODEL} (compat=False, bf16, revision pinned)...")
    bridge = load_model_pinned(revision)
    print(f"asserted OK: n_layers={bridge.cfg.n_layers} d_model={bridge.cfg.d_model} "
          f"probe layers {PROBE_LAYERS}")

    groups, sampling_meta = build_probe_groups(a.seed, lim_e, lim_s)
    print(f"sampling: {sampling_meta}")
    cfg = SimpleNamespace(skip_first_n_tokens=SKIP_FIRST_N)
    emotions, means, story_indices = [], [], {}
    for e, members in groups.items():
        idxs = [gi for gi, _ in members]
        estories = [s for _, s in members]
        sl = extract_story_slice(estories, bridge, cfg, PROBE_LAYERS)   # (n_stories, 3, 4608)
        means.append(sl.mean(dim=0))                                    # (3, 4608)
        story_indices[e] = idxs
        emotions.append(e)
    means = torch.stack(means, dim=0).float()                          # (n_em, 3, 4608)
    print(f"extracted per-emotion means: {tuple(means.shape)} over "
          f"{len(next(iter(groups.values())))} stories/emotion")

    try:
        import transformer_lens
        tl_ver = getattr(transformer_lens, "__version__", "unknown")
    except Exception:
        tl_ver = "unknown"
    prov = {
        "hf_revision_resolved": revision,
        "hf_revision_expected": EXPECTED_SHA,
        "torch": torch.__version__,
        "transformer_lens": tl_ver,
        "tl_compatibility_mode": False,
        "dtype": "bfloat16",
        "skip_first_n_tokens": SKIP_FIRST_N,
        "n_stories_per_emotion": len(next(iter(groups.values()))),
        "pod_gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "pilot": bool(a.pilot),
        "sampling": sampling_meta,
        "sampling_note": "seeded-random 120/emotion; deviation from handoff §3.4 first-120 (topic-clustered, would void L31 control), Adam-approved 2026-07-27",
        "pooling": "mean over resid_post from token 50 (BOS-inclusive); identical to _mean_resid_post_single",
    }
    payload = {"emotions": emotions, "layers": PROBE_LAYERS,
               "means": means, "story_indices": story_indices, "provenance": prov}
    if a.pilot:
        out = OUT.with_name("probe_l45_means_PILOT.pt")
    elif a.seed != PROBE_SEED:
        out = OUT.with_name(f"probe_l45_means_seed{a.seed}.pt")
    else:
        out = OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp"); torch.save(payload, tmp); tmp.rename(out)
    print(f"wrote {out.relative_to(REPO)}  (provenance: rev {revision[:12]}, {prov['pod_gpu']})")
    if a.pilot:
        print("\n[PILOT] shapes sane. Show this to Adam before the full run (handoff §3.5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
