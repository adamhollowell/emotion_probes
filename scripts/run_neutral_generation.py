"""
Generate the neutral-dialogue corpus used for PCA cleanup of emotion
vectors.

Defaults: 50 topics x 10 dialogues = 500 neutral dialogues. With ~150
tokens each that's ~75k token positions for PCA, enough for stable
principal components on Gemma-2-27B-class d_model.

Cost estimate: roughly $1-3 on the Anthropic API (much smaller than
the main story-generation run, since this is one corpus, not 12
emotions worth).

Output: data/stories/neutral_dialogues.jsonl

Safe to interrupt — re-running picks up unfinished topics.

Run from the project root:
    python scripts/run_neutral_generation.py
"""

from pathlib import Path

from emotion_probes.config import Config
from emotion_probes.generation import (
    NEUTRAL_DIALOGUE_TOPICS,
    generate_neutral_dialogues,
)


def main():
    config = Config(
        project_root=Path(__file__).parent.parent,
        generation_api="anthropic",
        generation_model="claude-sonnet-4-5",
    )

    n_per_topic = 10
    total = len(NEUTRAL_DIALOGUE_TOPICS) * n_per_topic
    print("=" * 70)
    print("Neutral-dialogue generation run starting.")
    print(f"Topics: {len(NEUTRAL_DIALOGUE_TOPICS)}")
    print(f"Dialogues per topic: {n_per_topic}")
    print(f"Total target: {total} dialogues")
    print(f"Model: {config.generation_model}")
    print(f"Output: {config.stories_dir / 'neutral_dialogues.jsonl'}")
    print("=" * 70)
    print()
    print("Safe to ctrl-C and resume.")
    print()

    generate_neutral_dialogues(config, n_per_topic=n_per_topic, verbose=True)


if __name__ == "__main__":
    main()
