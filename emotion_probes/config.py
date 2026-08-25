"""
Central configuration for an experimental run.

Everything that varies between "develop locally on Pythia-160M" and
"run for real on Gemma 2 27B (GPU)" lives here. The rest of the
package takes a Config object and acts on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- Model ----
    model_name: str = "google/gemma-2-9b-it"
    # Where in the model to extract activations. Anthropic uses "about
    # two-thirds of the way through". For Gemma 2 9B (42 layers) that's
    # ~layer 27-28. We extract from ALL layers and pick later, but this
    # is the layer to feature in default analyses.
    primary_layer: int | None = None  # set after model load if None
    device: str = "cuda"  # "mps" locally, "cuda" on GPU
    dtype: str = "bfloat16"  # "float32" for tiny dev models

    # ---- Story generation ----
    # Which API to use for generating stories. If None, generate with
    # the same model we're probing (cheap, lower quality).
    generation_api: str | None = "anthropic"  # "anthropic" | "openai" | None
    generation_model: str = "claude-sonnet-4-5"
    n_topics: int = 100
    n_stories_per_topic_per_emotion: int = 12
    # If you want a quick smoke test, drop these to e.g. 5 and 2.

    # ---- Activation extraction ----
    # Skip the first N tokens before averaging - emotional content
    # may not be apparent immediately. Anthropic uses 50.
    skip_first_n_tokens: int = 50
    batch_size: int = 8  # forward-pass batch size; tune to VRAM
    # transformer_lens 3.0 numerics regime. True folds LN/RMSNorm into
    # weights and centers, matching legacy HookedTransformer (and the
    # convention used by most published interp replications). False keeps
    # raw HuggingFace values; logit lens then needs the final norm applied
    # manually. Diff-of-means vectors are essentially invariant either way.
    tl_compatibility_mode: bool = True

    # ---- Probe computation ----
    # Number of principal components from neutral text to project out
    # of the raw emotion vectors. Anthropic uses "enough to explain
    # 50% of variance" - we encode that as a target rather than a count.
    pca_variance_target: float = 0.50

    # ---- Emotions ----
    # The 12 highlighted in the paper's figures - good starting set.
    # Set to None to use the full 171-emotion list (see emotions.py).
    emotion_subset: list[str] | None = field(
        default_factory=lambda: [
            "happy", "sad", "angry", "afraid", "calm", "desperate",
            "nervous", "surprised", "proud", "guilty", "inspired", "loving",
        ]
    )

    # ---- Paths ----
    # All paths are relative to the project root by default.
    project_root: Path = field(default_factory=lambda: Path.cwd())
    stories_dir: Path = field(init=False)
    activations_dir: Path = field(init=False)
    vectors_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.stories_dir = self.project_root / "data" / "stories"
        self.activations_dir = self.project_root / "data" / "activations"
        self.vectors_dir = self.project_root / "data" / "vectors"
        for d in (self.stories_dir, self.activations_dir, self.vectors_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- Convenience factories ----
    @classmethod
    def for_local_dev(cls) -> "Config":
        """Tiny model on the Mac for testing the pipeline end-to-end."""
        return cls(
            model_name="gpt2",
            device="cpu",
            dtype="float32",
            generation_api=None,  # generate with the same model; fast
            n_topics=5,
            n_stories_per_topic_per_emotion=2,
            batch_size=16,
        )

    @classmethod
    def for_runpod(cls, model_name: str = "google/gemma-2-27b") -> "Config":
        """Real run on rented GPU."""
        return cls(
            model_name=model_name,
            device="cuda",
            dtype="bfloat16",
            generation_api="anthropic",
            n_topics=100,
            n_stories_per_topic_per_emotion=12,
            batch_size=8,
        )
