#!/usr/bin/env python3
"""
Neutral-dialogue activations for ONE grid cell, the GPU step behind
gemma_grid/data/<cell>/vectors/<cell>__neutral.pt.

Reads the 500 neutral dialogues of the replication study (data/stories/neutral_dialogues.jsonl,
sha256 3838a8f2..., checked before the model loads) and writes every layer's residual-stream
activations, mean-pooled from token 50 by the replication study's own pooling path
(emotion_probes.activations.extract_activations_for_stories). Gates: the model's resolved
Hugging Face revision must equal the pin in gemma_grid/handoff/hf_model_revisions_VERIFIED.json,
and the cell's shape and analysis layer are asserted at load (gemma_grid.scripts.cell_config).

USAGE
  # pilot (3 dialogues, shape check):
  python gemma_grid/scripts/extract_neutral.py --cell gemma_grid/configs/gemma-2-2b-base.yaml --pilot 3
  # full run, one cell:
  python gemma_grid/scripts/extract_neutral.py --cell gemma_grid/configs/gemma-2-9b-it.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CORPUS_SHA256 = "3838a8f23e726d2433135336477ab0ee0e21f4ecb8ca06613963d26de8a3a04f"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from emotion_probes.activations import extract_activations_for_stories  # noqa: E402
from emotion_probes.generation import load_stories                      # noqa: E402
from gemma_grid.scripts.cell_config import load_cell, assert_model_matches  # noqa: E402

REVISIONS = REPO / "gemma_grid" / "handoff" / "hf_model_revisions_VERIFIED.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expected_sha(model_id: str) -> str:
    revs = json.loads(REVISIONS.read_text())
    if model_id not in revs:
        sys.exit(f"[fingerprint] {model_id} not in {REVISIONS.name} — STOP.")
    return revs[model_id]["current_main_sha"]


def fingerprint_gate(model_id: str) -> str:
    """Resolve main's HF SHA; assert it equals the recorded (VERIFIED) SHA.
    STOP on mismatch (handoff §3.1). Returns the resolved revision to PIN."""
    from huggingface_hub import HfApi
    exp = expected_sha(model_id)
    resolved = HfApi().model_info(model_id).sha
    print(f"[fingerprint] {model_id}")
    print(f"[fingerprint]   resolved main SHA = {resolved}")
    print(f"[fingerprint]   expected (VERIFIED) = {exp}")
    if resolved != exp:
        print("[fingerprint] MISMATCH — STOP. Report to Adam (handoff §4).")
        sys.exit(2)
    print("[fingerprint] PASS — pinning this revision explicitly.")
    return resolved


def load_model_pinned(cfg, revision: str, spec):
    """Grid regime EXACTLY: raw HF numerics (compat=False), bf16, revision
    PINNED. Mirrors emotion_probes.activations.load_model but adds the pin and
    NEVER enables compatibility mode. Then assert the cell shape."""
    from transformer_lens.model_bridge import TransformerBridge
    assert cfg.tl_compatibility_mode is False, "compat_mode must be False (anchor regime)"
    bridge = TransformerBridge.boot_transformers(
        cfg.model_name, device="cuda", dtype=torch.bfloat16, revision=revision,
    )
    # Do NOT call enable_compatibility_mode — see gemma_grid/CLAUDE.md.
    bridge.eval()
    analysis_layer = assert_model_matches(bridge, spec)   # raises on any mismatch
    return bridge, analysis_layer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", type=Path, required=True, help="gemma_grid/configs/*.yaml")
    ap.add_argument("--neutral", type=Path,
                    default=REPO / "data" / "stories" / "neutral_dialogues.jsonl")
    ap.add_argument("--tag", choices=("100T",), default="100T")
    ap.add_argument("--pilot", type=int, default=None,
                    help="extract only the first N dialogues (plumbing/shape check)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .pt; default gemma_grid/data/<cell>/vectors/<cell>__neutral.pt")
    a = ap.parse_args()

    cfg, spec = load_cell(a.cell, REPO)
    if not a.neutral.exists():
        sys.exit(f"neutral input not found: {a.neutral}")
    src_sha = sha256(a.neutral)
    print(f"[input] {a.neutral}  sha256={src_sha[:16]}  tag={a.tag}")
    if src_sha != CORPUS_SHA256:   # release: the replication study's neutral corpus, checked by hash
        sys.exit(f"{a.neutral} has sha256 {src_sha}, expected {CORPUS_SHA256}")

    revision = fingerprint_gate(cfg.model_name)
    print(f"Loading {cfg.model_name} (compat=False, bf16, revision pinned)...")
    bridge, analysis_layer = load_model_pinned(cfg, revision, spec)
    n_layers, d_model = int(bridge.cfg.n_layers), int(bridge.cfg.d_model)
    print(f"asserted OK: n_layers={n_layers} d_model={d_model} analysis L{analysis_layer}")

    stories = load_stories(a.neutral)
    if a.pilot is not None:
        stories = stories[:a.pilot]
    first_turn = stories[0].text.splitlines()[0] if stories else "<empty>"
    print(f"[{a.tag}] n={len(stories)}  first turn: {first_turn!r}")

    acts = extract_activations_for_stories(stories, bridge, cfg)   # (n, n_layers, d) f32 CPU
    assert acts.shape == (len(stories), n_layers, d_model), \
        f"shape {tuple(acts.shape)} != {(len(stories), n_layers, d_model)}"
    assert acts.dtype == torch.float32 and not torch.isnan(acts).any()
    print(f"[{a.tag}] acts {tuple(acts.shape)} {acts.dtype}")

    out = a.out or (REPO / "gemma_grid" / "data" / spec.cell_id / "vectors"
                    / f"{spec.cell_id}__neutral.pt")
    if a.pilot is not None:
        out = out.with_name(out.stem + "_PILOT.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import transformer_lens
        tl_ver = getattr(transformer_lens, "__version__", None) or \
            __import__("importlib.metadata", fromlist=["version"]).version("transformer-lens")
    except Exception:
        tl_ver = "unknown"

    payload = {
        "acts": acts,
        "src": a.neutral.name,
        "src_sha256": src_sha,
        "model_name": cfg.model_name,
        "hf_revision_resolved": revision,
        "hf_revision_expected": expected_sha(cfg.model_name),
        "n_stories": len(stories),
        "config": {
            "cell_id": spec.cell_id,
            "tuning": spec.tuning,
            "tl_compatibility_mode": False,
            "dtype": "bfloat16",
            "skip_first_n_tokens": cfg.skip_first_n_tokens,
            "pca_variance_target": cfg.pca_variance_target,
            "n_layers": n_layers,
            "d_model": d_model,
            "analysis_layer": analysis_layer,
        },
        "provenance": {
            "torch": torch.__version__,
            "transformer_lens": tl_ver,
            "pod_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": platform.node(),
            "pilot": a.pilot,
            "pooling": "mean over resid_post from token 50 (BOS-inclusive) with "
                       "short-text fallback; identical to _mean_resid_post_single",
        },
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.rename(out)
    print(f"[{a.tag}] wrote {out.relative_to(REPO)}  "
          f"(rev {revision[:12]}, {payload['provenance']['pod_gpu']})")

    if a.pilot is not None:
        print("\n[PILOT] shapes sane. Show this to Adam before the full run (handoff §3.3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
