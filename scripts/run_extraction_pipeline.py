"""
End-to-end emotion-vector extraction driver (GPU).

Stages, in order:
    1. Load the model (Gemma 2 27B by default).
    2. Extract residual-stream activations from the 14,400 emotion
       stories. Already file-checkpointed per (emotion, variant) in
       activations.extract_all_activations — worst-case rework on
       interrupt is the in-progress emotion (~20 min on A100).
    3. Extract residual-stream activations from the 500 neutral
       dialogues. Chunked at the driver level so an interrupt costs at
       most ~1 chunk (~50 dialogues, ~1 min on A100). Concatenated
       into one tensor once every chunk is on disk.
    4. Compute emotion vectors (diff-of-means + PCA cleanup).
    5. Save vectors. Write a validation report.

The whole driver is resumable: on each re-run we skip any stage whose
artifact exists. Mid-stage interrupts are recoverable via the
file-level granularity baked in to stages 2 and 3.

Usage:
    python scripts/run_extraction_pipeline.py
    python scripts/run_extraction_pipeline.py --config gpt2-local
    python scripts/run_extraction_pipeline.py --neutral-chunk-size 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from emotion_probes.activations import (
    extract_activations_for_stories,
    extract_all_activations,
    load_activation_batch,
    load_model,
)
from emotion_probes.config import Config
from emotion_probes.generation import Story, load_stories
from emotion_probes.probes import (
    compute_emotion_vectors,
    cosine_similarities,
    load_vectors,
    save_vectors,
)
from emotion_probes.utils import primary_layer
from emotion_probes.validation import logit_lens, validate_geometry


REPO = Path(__file__).resolve().parent.parent

NEUTRAL_FILENAME = "neutral_dialogues.jsonl"
NEUTRAL_TENSOR_FILENAME = "neutral.pt"
NEUTRAL_CHUNKS_DIRNAME = "neutral_chunks"

REPORT_FILENAME = "report.md"


# ---- Config selection ----

def build_config(name: str, project_root: Path) -> Config:
    if name == "gemma-27b":
        return Config(
            model_name="google/gemma-2-27b",
            device="cuda", dtype="bfloat16",
            tl_compatibility_mode=False,
            emotion_subset=None,
            project_root=project_root,
        )
    if name == "gemma-9b":
        return Config(
            model_name="google/gemma-2-9b",
            device="cuda", dtype="bfloat16",
            tl_compatibility_mode=True,
            project_root=project_root,
        )
    if name == "gpt2-local":
        return Config(
            model_name="gpt2",
            device="cpu", dtype="float32",
            tl_compatibility_mode=True,
            project_root=project_root,
        )
    raise ValueError(
        f"unknown --config {name!r}; try gemma-27b, gemma-9b, gpt2-local"
    )


# ---- Stage 2: story extraction ----

def stage_story_extraction(cfg: Config, bridge) -> None:
    """
    Extract activations for every story under stories.jsonl. Reuses the
    per-(emotion, variant) file checkpoint in activations.extract_all_activations.
    """
    stories_path = cfg.stories_dir / "stories_171_canonical_clean.jsonl"
    if not stories_path.exists():
        raise FileNotFoundError(
            f"missing {stories_path}; make it available on the GPU machine first"
        )
    stories = load_stories(stories_path)
    print(f"  loaded {len(stories)} stories")

    extract_all_activations(stories, bridge, cfg)

    # Verify the expected per-emotion files landed.
    emotions = cfg.emotion_subset or sorted({s.emotion for s in stories})
    for emotion in emotions:
        path = cfg.activations_dir / f"{emotion}__canonical.pt"
        if not path.exists():
            raise RuntimeError(f"expected output missing: {path}")
        batch = load_activation_batch(emotion, "canonical", cfg)
        assert batch.activations.ndim == 3
    print(f"  verified {len(emotions)} per-emotion files on disk")


# ---- Stage 3: neutral extraction (driver-side chunking) ----

def _neutral_chunks_dir(cfg: Config) -> Path:
    d = cfg.activations_dir / NEUTRAL_CHUNKS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _neutral_tensor_path(cfg: Config) -> Path:
    return cfg.activations_dir / NEUTRAL_TENSOR_FILENAME


def stage_neutral_extraction(
    cfg: Config, bridge, chunk_size: int,
) -> torch.Tensor:
    """
    Extract activations for every neutral dialogue under
    neutral_dialogues.jsonl. Chunked into chunk_size-sized batches.
    Each chunk's tensor is saved to disk; on re-run we skip chunks
    whose file exists. After every chunk is on disk we concatenate
    them into neutral.pt and remove the per-chunk files.

    If neutral.pt already exists, load and return it directly.
    """
    final_path = _neutral_tensor_path(cfg)
    if final_path.exists():
        print(f"  {final_path.name} present; loading")
        return torch.load(final_path, map_location="cpu", weights_only=False)

    path = cfg.stories_dir / NEUTRAL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run scripts/run_neutral_generation.py first "
            f"(or copy from your local box)"
        )
    neutrals = load_stories(path)
    print(f"  loaded {len(neutrals)} neutral dialogues")

    chunks_dir = _neutral_chunks_dir(cfg)
    n_chunks = (len(neutrals) + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        chunk_path = chunks_dir / f"chunk_{i:04d}.pt"
        if chunk_path.exists():
            continue
        start = i * chunk_size
        end = min(start + chunk_size, len(neutrals))
        chunk_stories = neutrals[start:end]
        print(f"  chunk {i+1}/{n_chunks}: stories {start}-{end-1}")
        t0 = time.time()
        acts = extract_activations_for_stories(chunk_stories, bridge, cfg)
        tmp = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
        torch.save(
            {"chunk": i, "start": start, "end": end, "activations": acts}, tmp,
        )
        tmp.rename(chunk_path)  # atomic
        print(f"    {time.time()-t0:.1f}s, shape={tuple(acts.shape)}")

    # Concatenate all chunks into one tensor.
    print("  concatenating chunks")
    pieces = []
    for i in range(n_chunks):
        chunk_path = chunks_dir / f"chunk_{i:04d}.pt"
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        pieces.append(payload["activations"])
    all_neutral = torch.cat(pieces, dim=0)
    assert all_neutral.shape[0] == len(neutrals), (
        f"concat mismatch: got {all_neutral.shape[0]}, expected {len(neutrals)}"
    )

    # Atomic write of the final tensor.
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    torch.save(all_neutral, tmp)
    tmp.rename(final_path)
    print(f"  wrote {final_path.name} shape={tuple(all_neutral.shape)}")

    # Clean up per-chunk files only after the final file is durable.
    for i in range(n_chunks):
        (chunks_dir / f"chunk_{i:04d}.pt").unlink(missing_ok=True)
    try:
        chunks_dir.rmdir()
    except OSError:
        pass  # not empty; fine.
    return all_neutral


# ---- Stage 4: vector computation ----

def stage_vectors(cfg: Config, neutral_acts: torch.Tensor):
    vec_path = cfg.vectors_dir / "canonical.pt"
    if vec_path.exists():
        print(f"  {vec_path.name} present; loading")
        return load_vectors("canonical", cfg)

    vectors = compute_emotion_vectors(
        cfg.activations_dir, neutral_acts, cfg, variant="canonical",
    )
    save_vectors(vectors, cfg)
    print(f"  wrote {vec_path.name}, "
          f"shape={tuple(vectors.vectors.shape)}")
    print(f"  cleanup delta: "
          f"{(vectors.vectors - vectors.vectors_raw).norm() / vectors.vectors_raw.norm() * 100:.1f}% "
          "of raw norm")
    return vectors


# ---- Stage 5: validation report ----

def stage_validation(cfg: Config, bridge, vectors) -> Path:
    layer = primary_layer(cfg, vectors.n_layers)
    print(f"  primary layer: {layer} / {vectors.n_layers}")

    print("  logit lens...")
    lens = logit_lens(vectors, bridge, layer=layer, top_k=10)

    print("  geometry (PCA across emotions)...")
    geom = validate_geometry(vectors, layer=layer)

    print("  cosine similarities...")
    cs = cosine_similarities(vectors, layer=layer)

    report_path = cfg.vectors_dir / REPORT_FILENAME
    _write_report(
        report_path=report_path, cfg=cfg, vectors=vectors,
        layer=layer, lens=lens, geom=geom, cs=cs,
    )
    print(f"  wrote {report_path.relative_to(REPO)}")
    return report_path


def _write_report(
    report_path: Path, cfg: Config, vectors,
    layer: int, lens, geom, cs,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = []
    lines.append(f"# Emotion vector extraction report")
    lines.append("")
    lines.append(f"- Model: `{cfg.model_name}`")
    lines.append(f"- Compatibility mode: `{cfg.tl_compatibility_mode}`")
    lines.append(f"- skip_first_n_tokens: `{cfg.skip_first_n_tokens}`")
    lines.append(f"- pca_variance_target: `{cfg.pca_variance_target}`")
    lines.append(f"- Layer reported: **{layer}** of {vectors.n_layers}")
    lines.append(f"- Emotions ({len(vectors.emotions)}): "
                 f"{', '.join(vectors.emotions)}")
    lines.append(f"- Generated: {ts}")
    lines.append("")

    lines.append("## Logit lens — top 10 tokens per emotion")
    lines.append("")
    for r in lens:
        top = ", ".join(f"`{t.strip()}` ({v:+.2f})" for t, v in r.top_tokens)
        lines.append(f"- **{r.emotion}**: {top}")
    lines.append("")

    lines.append("## Logit lens — bottom 10 tokens per emotion")
    lines.append("")
    for r in lens:
        bot = ", ".join(f"`{t.strip()}` ({v:+.2f})" for t, v in r.bottom_tokens)
        lines.append(f"- **{r.emotion}**: {bot}")
    lines.append("")

    ratio = geom["explained_variance_ratio"]
    lines.append("## Geometry (PCA across emotions)")
    lines.append("")
    lines.append(f"- PC1 explained variance: **{ratio[0].item():.1%}**")
    lines.append(f"- PC2 explained variance: **{ratio[1].item():.1%}**")
    lines.append("")
    lines.append("**PC1 ranking (high to low — typically valence axis):**")
    lines.append("")
    for e, v in geom["pc1_ranking"]:
        lines.append(f"- {e}: {v:+.3f}")
    lines.append("")
    lines.append("**PC2 ranking (high to low — typically arousal axis):**")
    lines.append("")
    for e, v in geom["pc2_ranking"]:
        lines.append(f"- {e}: {v:+.3f}")
    lines.append("")

    lines.append("## Cosine similarities")
    lines.append("")
    header = "| | " + " | ".join(vectors.emotions) + " |"
    sep = "|---|" + "|".join(["---"] * len(vectors.emotions)) + "|"
    lines.append(header)
    lines.append(sep)
    for i, e in enumerate(vectors.emotions):
        row = "| " + e + " | " + " | ".join(
            f"{cs[i, j].item():+.2f}" for j in range(cs.shape[1])
        ) + " |"
        lines.append(row)
    lines.append("")

    lines.append("## Run metadata")
    lines.append("")
    lines.append("```json")
    summary = {
        "model_name": cfg.model_name,
        "n_layers": vectors.n_layers,
        "d_model": vectors.d_model,
        "primary_layer": layer,
        "emotions": vectors.emotions,
        "cleanup_delta_fraction": float(
            (vectors.vectors - vectors.vectors_raw).norm() /
            vectors.vectors_raw.norm()
        ),
        "generated_utc": ts,
    }
    lines.append(json.dumps(summary, indent=2))
    lines.append("```")

    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text("\n".join(lines))
    tmp.rename(report_path)


# ---- Driver ----

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="gemma-27b",
        choices=("gemma-27b", "gemma-9b", "gpt2-local"),
        help="Which Config preset to use (default: gemma-27b)",
    )
    parser.add_argument(
        "--neutral-chunk-size", type=int, default=50,
        help="Neutral dialogues per chunk (default 50)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip stage 5 (validation report). Useful for batched runs.",
    )
    args = parser.parse_args()

    cfg = build_config(args.config, project_root=REPO)
    print("=" * 60)
    print(f"Pipeline run starting")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model: {cfg.model_name}")
    print(f"Device: {cfg.device}, dtype: {cfg.dtype}")
    print(f"Project root: {cfg.project_root}")
    print()

    print("Stage 1: load model")
    t0 = time.time()
    bridge = load_model(cfg)
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"(n_layers={bridge.cfg.n_layers}, d_model={bridge.cfg.d_model})")

    print()
    print("Stage 2: story activations")
    t0 = time.time()
    stage_story_extraction(cfg, bridge)
    print(f"  Stage 2 done in {(time.time()-t0)/60:.1f} min")

    print()
    print("Stage 3: neutral activations")
    t0 = time.time()
    neutral_acts = stage_neutral_extraction(
        cfg, bridge, chunk_size=args.neutral_chunk_size,
    )
    print(f"  Stage 3 done in {(time.time()-t0)/60:.1f} min")

    print()
    print("Stage 4: compute emotion vectors")
    t0 = time.time()
    vectors = stage_vectors(cfg, neutral_acts)
    print(f"  Stage 4 done in {time.time()-t0:.1f}s")

    if not args.skip_validation:
        print()
        print("Stage 5: validation report")
        t0 = time.time()
        report = stage_validation(cfg, bridge, vectors)
        print(f"  Stage 5 done in {time.time()-t0:.1f}s")
        print()
        print(f"Report: {report.relative_to(REPO)}")

    print()
    print("=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
