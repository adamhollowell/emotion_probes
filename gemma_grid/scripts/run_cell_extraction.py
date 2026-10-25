#!/usr/bin/env python3
"""
Config-driven extraction driver for ONE grid cell (Phase 0 wiring).

Reuses emotion_probes.{activations,probes,generation,utils} unchanged and adds
only the grid-specific pieces registered in gemma_grid/PREREGISTRATION.md:
  * load a cell YAML, ASSERT the model's shape at load, resolve the analysis
    layer from the model's OWN config           (gemma_grid.scripts.cell_config)
  * capture logit-lens ingredients (W_U, b_U, final-norm weight)          [C5]
  * persist the per-story analysis-layer(+band) slice   [extraction contract 3]
  * TWO modes:
      A  new cells   : full-depth extraction -> all-layer vectors (canonical.pt)
                       + lens ingredients + per-story analysis-layer slice
      B  anchor rerun: stop_at_layer truncation, per-story analysis-layer slice
                       + neutrals at the analysis layer, then recompute the L31
                       vectors and run the REPRODUCTION GATE vs vectors_171.pt

Model API verified against TransformerBridge (gpt2): b.W_U, b.ln_final.w, b.b_U,
run_with_cache(stop_at_layer=...) all present.

This module imports with NO GPU and its model-free helpers are unit-testable;
the model path runs at the Phase-2 pilot and Phase-3 full extraction. It does
NOT cross any GATE on its own — it is the harness the gates approve the use of.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from emotion_probes.activations import (          # noqa: E402
    extract_activations_for_stories, extract_all_activations,
    load_activation_batch, load_model,
)
from emotion_probes.generation import Story, load_stories          # noqa: E402
from emotion_probes.probes import (               # noqa: E402
    difference_of_means, project_out_top_pcs, save_vectors, EmotionVectors,
    compute_emotion_vectors,
)
from emotion_probes.emotions import EMOTIONS_171          # noqa: E402
from gemma_grid.scripts.cell_config import load_cell, assert_model_matches  # noqa: E402


# --------------------------------------------------------------------------
# Layer set for the per-story slice: analysis layer + any sensitivity band.
# --------------------------------------------------------------------------
def slice_layers(analysis_layer: int, band: tuple[int, ...] = ()) -> list[int]:
    """Sorted unique layers to persist per-story (analysis layer + band)."""
    return sorted({analysis_layer, *band})


def subset_stories(stories, limit_emotions, limit_stories):
    """Pilot subset: first `limit_emotions` emotions × first `limit_stories`
    each (either None = keep all). Preserves per-emotion order; used only for
    the Phase-2 plumbing pilot, never a real run."""
    if limit_emotions is None and limit_stories is None:
        return stories
    by: dict[str, list] = {}
    for s in stories:
        by.setdefault(s.emotion, []).append(s)
    emotions = list(by)[:limit_emotions] if limit_emotions is not None else list(by)
    out = []
    for e in emotions:
        rows = by[e]
        out.extend(rows[:limit_stories] if limit_stories is not None else rows)
    return out


# --------------------------------------------------------------------------
# Logit-lens ingredients (C5) — captured while the model is loaded.
# --------------------------------------------------------------------------
def capture_lens_ingredients(bridge) -> dict:
    """W_U, b_U, and the final-norm weight for the offline (1+w) Gemma lens."""
    lnf = getattr(bridge, "ln_final", None)
    w = getattr(lnf, "w", None) if lnf is not None else None
    b_U = getattr(bridge, "b_U", None)
    return {
        # Native dtype (bf16) — the §7.2 lens IS the raw-weights regime; upcasting
        # to float32 only doubles the file (W_U is 2304..4608 × 256000).
        "W_U": bridge.W_U.detach().cpu(),
        "b_U": None if b_U is None else b_U.detach().cpu(),
        "ln_final_w": None if w is None else w.detach().cpu(),
        "dtype": str(bridge.W_U.dtype),
        "n_layers": int(bridge.cfg.n_layers),
        "d_model": int(bridge.cfg.d_model),
        # Gemma-2 final RMSNorm is applied as (1+w) at readout (P4.2/P4.3);
        # these weights are RAW (tl_compatibility_mode=False).
        "note": "apply Gemma-2 final RMSNorm in (1+w) form at lens readout",
    }


# --------------------------------------------------------------------------
# Truncated per-story capture (mode B, and the slice for mode A can reuse it).
# --------------------------------------------------------------------------
def _pool_slice_single(text, bridge, layers, skip_first_n) -> torch.Tensor:
    toks = bridge.to_tokens(text)
    seq = toks.shape[1]
    start = skip_first_n if seq > skip_first_n + 1 else min(1, seq - 1)
    hooks = [f"blocks.{l}.hook_resid_post" for l in layers]
    with torch.no_grad():
        _, cache = bridge.run_with_cache(
            toks, names_filter=hooks, stop_at_layer=max(layers) + 1,
        )
    return torch.stack(
        [cache[h][0, start:].mean(dim=0).float().cpu() for h in hooks], dim=0,
    )  # (len(layers), d_model)


def extract_story_slice(stories, bridge, cfg, layers) -> torch.Tensor:
    """Per-story pooled activations at `layers` only, forward truncated.

    Returns (n_stories, len(layers), d_model). Cheaper than a full-depth pass
    because run_with_cache stops at max(layers)+1.
    """
    out = torch.empty(len(stories), len(layers), bridge.cfg.d_model, dtype=torch.float32)
    for i, s in enumerate(stories):
        out[i] = _pool_slice_single(s.text, bridge, layers, cfg.skip_first_n_tokens)
    return out


# --------------------------------------------------------------------------
# Reproduction gate (mode B) — recomputed L31 vectors vs locked vectors_171.pt,
# under verify_reproduction.py's own TOL bands. Layer-stability anchors are out
# of scope for an L31-only recapture (they need all layers) and are ignored, as
# registered in PREREGISTRATION.md (e).
# --------------------------------------------------------------------------
def reproduction_gate(
    recomputed_layer_vectors: torch.Tensor,   # (n_emotions, d_model) at analysis layer
    emotions: list[str],
    locked_vectors_path: Path,
    analysis_layer: int,
    verify_script: Path,
    locked_quick: Path,
    locked_fig8: Path,
    ratings: Path,
) -> bool:
    """Return True iff every NON-layer-stability anchor reproduces under TOL.

    Builds a full (n, L, d) tensor = locked vectors with the analysis layer
    REPLACED by the recomputed one, then runs the unchanged verifier and parses
    its PASS/FAIL lines. Only the L31-dependent anchors gate the rerun.
    """
    locked = torch.load(locked_vectors_path, map_location="cpu", weights_only=False)
    lem = list(locked["emotions"])
    V = locked["vectors"]
    V = V.clone() if torch.is_tensor(V) else torch.as_tensor(V).clone()
    V = V.float()
    # align recomputed rows to the locked emotion order
    idx = [emotions.index(e) for e in lem]
    V[:, analysis_layer, :] = recomputed_layer_vectors[idx].float()

    tmp = locked_vectors_path.parent / "_repro_gate_tmp.pt"
    torch.save({**locked, "vectors": V}, tmp)
    try:
        proc = subprocess.run(
            [sys.executable, str(verify_script),
             "--vectors", str(tmp),
             "--locked-quick", str(locked_quick),
             "--locked-fig8", str(locked_fig8),
             "--ratings", str(ratings),
             "--layer", str(analysis_layer)],
            capture_output=True, text=True,
        )
    finally:
        tmp.unlink(missing_ok=True)

    fails = [ln for ln in proc.stdout.splitlines() if "[FAIL]" in ln]
    non_layerstab_fails = [ln for ln in fails if "layer_stability" not in ln]
    print(proc.stdout)
    if non_layerstab_fails:
        print("REPRODUCTION GATE: FAIL —", len(non_layerstab_fails), "L31 anchor(s) did not reproduce")
        for ln in non_layerstab_fails:
            print("   ", ln.strip())
        return False
    ignored = [ln for ln in fails if "layer_stability" in ln]
    print(f"REPRODUCTION GATE: PASS (L31 anchors reproduced; "
          f"{len(ignored)} layer-stability anchor(s) ignored — out of scope for L31-only rerun)")
    return True


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_per_story_slice(path: Path, per_emotion: dict, neutral, layers, meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"per_emotion": per_emotion, "neutral": neutral,
                "layers": list(layers), **meta}, tmp)
    tmp.rename(path)


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


# --------------------------------------------------------------------------
# Mode B recompute: cleaned analysis-layer vectors from per-story slices.
# --------------------------------------------------------------------------
def recompute_cleaned_layer_vectors(per_emotion_slice, neutral_slice, cfg):
    """Diff-of-means + neutral-basis cleanup on a single-layer slice.

    per_emotion_slice: {emotion: (n_stories, 1, d)}   neutral_slice: (n_neu, 1, d)
    Returns (emotions, cleaned (n_emotions, 1, d)) — the SAME cleaned quantity
    the study reports, so the bootstrap/gate compare like with like.
    """
    raw, emotions = difference_of_means(per_emotion_slice)          # (n_em, 1, d)
    cleaned = project_out_top_pcs(raw, neutral_slice, cfg.pca_variance_target)
    return emotions, cleaned


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
def _save_reduced(path: Path, emotion: str, mean_all, slice_ps, story_indices) -> None:
    """Persist an emotion's REDUCED record: all-layer mean (for the vectors) +
    per-story slice at the pre-registered layers (for the bootstrap). Small."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"emotion": emotion, "mean": mean_all.cpu(),
                "slice": slice_ps.cpu(), "story_indices": list(story_indices)}, tmp)
    tmp.rename(path)


def run_mode_A(cfg, spec, bridge, analysis_layer, band, out_root: Path, pilot=None) -> None:
    """Disk-light new-cell extraction (inline prune-after-slice).

    Per emotion: forward all stories, then keep ONLY the emotion's all-layer
    MEAN (this reconstructs the full all-layer vectors exactly — mean-of-means)
    plus the per-story slice at the pre-registered layer(s) (bootstrap). The raw
    all-layer per-story tensor is reduced in RAM and NEVER written to disk, so
    the footprint is ~4-8 GB/cell, not 46-170 GB. Every reported number is
    identical to the full pipeline; only out-of-scope raw tensors are dropped.

    Resumable: skips emotions whose reduced file exists. Existing OLD-format
    full tensors ({emotion}__canonical.pt) are converted to reduced form (no
    GPU) and deleted, so a partially-extracted cell finishes without re-running.
    """
    layers = slice_layers(analysis_layer, band)
    print(f"[mode A] {spec.cell_id}: disk-light extraction, analysis L{analysis_layer}, slice layers {layers}")
    adir = cfg.activations_dir

    stories = load_stories(cfg.stories_dir / "stories_171_canonical_clean.jsonl")
    if pilot:
        stories = subset_stories(stories, pilot["emotions"], pilot["stories"])
        cfg.emotion_subset = sorted({s.emotion for s in stories})
        print(f"[PILOT] {len(stories)} stories, {len(cfg.emotion_subset)} emotions: {cfg.emotion_subset}")
    by_emotion: dict[str, list[Story]] = {}
    for s in stories:
        by_emotion.setdefault(s.emotion, []).append(s)

    for emotion, estories in by_emotion.items():
        reduced_path = adir / f"{emotion}__reduced.pt"
        full_path = adir / f"{emotion}__canonical.pt"   # old full-tensor checkpoint
        if reduced_path.exists():
            continue
        if full_path.exists():
            b = load_activation_batch(emotion, "canonical", cfg)
            acts = b.activations
            _save_reduced(reduced_path, emotion, acts.mean(dim=0), acts[:, layers, :].clone(), b.story_indices)
            full_path.unlink()
            print(f"  converted {emotion}: existing full tensor -> reduced (freed all-layer)")
            continue
        acts = extract_activations_for_stories(estories, bridge, cfg)   # (n_stories, n_layers, d) in RAM
        _save_reduced(reduced_path, emotion, acts.mean(dim=0), acts[:, layers, :].clone(), list(range(len(estories))))

    # Neutrals kept all-layer (small) — needed as the per-layer cleanup basis.
    neutrals = load_stories(cfg.stories_dir / "neutral_dialogues.jsonl")
    if pilot and pilot.get("neutrals"):
        neutrals = neutrals[:pilot["neutrals"]]
    neutral_all = extract_activations_for_stories(neutrals, bridge, cfg)   # (n_neu, n_layers, d)

    # Canonical vectors from the per-emotion means. This IS difference_of_means:
    #   raw = mean_e - mean_over_emotions(mean_e); cleanup reuses project_out_top_pcs.
    requested = cfg.emotion_subset or EMOTIONS_171
    emotions = [e for e in requested if (adir / f"{e}__reduced.pt").exists()]
    means, slices = [], {}
    for e in emotions:
        r = torch.load(adir / f"{e}__reduced.pt", weights_only=False)
        means.append(r["mean"].float())
        slices[e] = r["slice"]
    per_emotion_means = torch.stack(means, dim=0)                      # (n_em, n_layers, d)
    raw = per_emotion_means - per_emotion_means.mean(dim=0, keepdim=True)
    cleaned = project_out_top_pcs(raw, neutral_all.float(), cfg.pca_variance_target)
    vectors = EmotionVectors(variant="canonical", emotions=emotions, vectors=cleaned,
                             vectors_raw=raw, n_layers=raw.shape[1], d_model=raw.shape[2])
    save_vectors(vectors, cfg)

    neutral_slice = neutral_all[:, layers, :].clone()
    _atomic_save(capture_lens_ingredients(bridge), out_root / f"{spec.cell_id}__lens_ingredients.pt")
    save_per_story_slice(
        out_root / f"{spec.cell_id}__per_story_slice.pt",
        slices, neutral_slice, layers,
        meta={"cell_id": spec.cell_id, "analysis_layer": analysis_layer, "mode": "A", "slice_layers": layers},
    )
    print(f"[mode A] wrote canonical.pt, lens, per-story slice ({len(emotions)} emotions); "
          f"disk-light — no all-layer per-story tensors retained")


def run_mode_B(cfg, spec, bridge, analysis_layer, band, out_root: Path, gate_paths: dict, pilot=None) -> bool:
    """Anchor rerun: truncated capture, recompute L31, reproduction gate."""
    layers = slice_layers(analysis_layer, band)
    print(f"[mode B] {spec.cell_id}: truncated rerun (stop_at_layer={max(layers)+1}), "
          f"analysis L{analysis_layer}")

    stories = load_stories(cfg.stories_dir / "stories_171_canonical_clean.jsonl")
    if pilot:
        stories = subset_stories(stories, pilot["emotions"], pilot["stories"])
        print(f"[PILOT] {len(stories)} stories")
    by_emotion: dict[str, list[Story]] = {}
    for s in stories:
        by_emotion.setdefault(s.emotion, []).append(s)
    per_emotion_slice = {e: extract_story_slice(v, bridge, cfg, layers)
                         for e, v in by_emotion.items()}

    neutrals = load_stories(cfg.stories_dir / "neutral_dialogues.jsonl")
    if pilot and pilot.get("neutrals"):
        neutrals = neutrals[:pilot["neutrals"]]
    neutral_slice = extract_story_slice(neutrals, bridge, cfg, layers)

    save_per_story_slice(
        out_root / f"{spec.cell_id}__per_story_slice.pt",
        per_emotion_slice, neutral_slice, layers,
        meta={"cell_id": spec.cell_id, "analysis_layer": analysis_layer, "mode": "B"},
    )

    if pilot:
        shp = next(iter(per_emotion_slice.values())).shape
        print(f"[PILOT] mode-B plumbing OK: per-emotion slice {tuple(shp)}, "
              f"neutral slice {tuple(neutral_slice.shape)} — reproduction gate SKIPPED "
              f"(needs full 171 emotions; runs only in the full mode-B job).")
        return True

    ai = layers.index(analysis_layer)
    emotions, cleaned = recompute_cleaned_layer_vectors(
        {e: v[:, ai:ai + 1, :] for e, v in per_emotion_slice.items()},
        neutral_slice[:, ai:ai + 1, :], cfg,
    )
    ok = reproduction_gate(
        cleaned[:, 0, :], emotions,
        analysis_layer=analysis_layer, **gate_paths,
    )
    if not ok:
        print("[mode B] STOP — reproduction gate failed; bootstrap is void.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", type=Path, required=True, help="path to a gemma_grid/configs/*.yaml")
    ap.add_argument("--mode", choices=("A", "B"), required=True,
                    help="A=new cell (full depth); B=anchor rerun (truncated + repro gate)")
    ap.add_argument("--band", type=int, nargs="*", default=[],
                    help="extra sensitivity layers to persist (e.g. 18 for the 2B L18 check)")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="defaults to gemma_grid/data/<cell_id>/ (per-cell, no collisions)")
    # reproduction-gate inputs (mode B only)
    ap.add_argument("--locked-vectors", type=Path, default=REPO / "data" / "vectors" / "vectors_171.pt")
    ap.add_argument("--verify-script", type=Path, default=REPO / "scripts" / "verify_reproduction.py")
    ap.add_argument("--locked-quick", type=Path, default=REPO / "geometry_out" / "geometry_quick_summary.json")
    ap.add_argument("--locked-fig8", type=Path, default=REPO / "geometry_out" / "fig8_pca_vs_russell_mehrabian_REBUILD_summary.json")
    ap.add_argument("--ratings", type=Path, default=REPO / "data" / "ratings" / "russell_mehrabian_1977.csv")
    # Phase-2 pilot: subset the corpus for a plumbing check (never a real run).
    ap.add_argument("--pilot", action="store_true",
                    help="plumbing pilot: subset corpus (defaults 3 emotions × 3 stories × 10 neutrals), skip mode-B gate")
    ap.add_argument("--limit-emotions", type=int, default=None)
    ap.add_argument("--limit-stories", type=int, default=None)
    ap.add_argument("--limit-neutrals", type=int, default=None)
    a = ap.parse_args()

    pilot = None
    if a.pilot:
        pilot = {"emotions": a.limit_emotions or 3,
                 "stories": a.limit_stories or 3,
                 "neutrals": a.limit_neutrals or 10}

    cfg, spec = load_cell(a.cell, REPO)
    # Namespace ALL writes under gemma_grid/data/<cell_id>/ so nothing lands in
    # the first-round data/ dirs and the six cells never collide. The corpus is
    # still READ from the first-round data/stories (cfg.stories_dir, unchanged).
    cell_root = REPO / "gemma_grid" / "data" / spec.cell_id
    cfg.activations_dir = cell_root / "activations"
    cfg.vectors_dir = cell_root / "vectors"
    for d in (cfg.activations_dir, cfg.vectors_dir):
        d.mkdir(parents=True, exist_ok=True)
    if a.out_root is None:
        a.out_root = cell_root / "vectors"
    print(f"outputs -> {cell_root.relative_to(REPO)}/  (corpus read from {cfg.stories_dir.relative_to(REPO)}/)")
    print(f"Loading {cfg.model_name} (compat={cfg.tl_compatibility_mode})...")
    bridge = load_model(cfg)
    analysis_layer = assert_model_matches(bridge, spec)
    cfg.primary_layer = analysis_layer
    print(f"asserted OK: n_layers={bridge.cfg.n_layers} d_model={bridge.cfg.d_model} "
          f"analysis L{analysis_layer}")

    a.out_root.mkdir(parents=True, exist_ok=True)
    if a.mode == "A":
        run_mode_A(cfg, spec, bridge, analysis_layer, tuple(a.band), a.out_root, pilot=pilot)
        print(f"[{'PILOT' if pilot else 'DONE'}] mode A complete for {spec.cell_id}")
        return 0
    if pilot is None:   # release: stop with a message when the gate's inputs are absent
        missing = [str(p) for p in (a.locked_vectors, a.verify_script, a.locked_quick, a.locked_fig8) if not p.exists()]
        if missing:
            sys.exit("Mode B's reproduction gate compares the base 27B rerun with the replication study's "
                     "first-round anchors, through a verifier and anchor files this release does not hold ("
                     + ", ".join(missing) + "). The base 27B per-story slice ships as the record of that run. "
                     "Pass --pilot for a plumbing check, which skips the gate.")
    gate_paths = dict(locked_vectors_path=a.locked_vectors, verify_script=a.verify_script,
                      locked_quick=a.locked_quick, locked_fig8=a.locked_fig8, ratings=a.ratings)
    ok = run_mode_B(cfg, spec, bridge, analysis_layer, tuple(a.band), a.out_root, gate_paths, pilot=pilot)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
