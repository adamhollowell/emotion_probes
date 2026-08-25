"""
Story generation: produce labeled emotion-laden text.

Following Anthropic's exact methodology:
    - 100 topics (verbatim from their appendix)
    - 12 stories per topic per emotion
    - Their exact system prompt, including the "NEVER use the word"
      instruction
    - Stories generated in batches of N per API call, parsed apart

Design notes:
    - Each story saved as one JSONL row with metadata.
    - The Story schema supports object tagging for later experiments.
    - Generation is resumable: re-running picks up where it left off.
    - One API call produces multiple stories at once (per Anthropic's
      prompt format), making the run substantially cheaper than
      one-story-per-call.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from emotion_probes.config import Config
from emotion_probes.emotions import EMOTIONS_12


# ---- Data structures ----

@dataclass
class Story:
    """One generated story with all metadata needed for later analysis."""
    emotion: str
    topic: str
    text: str
    emotion_object: str | None = None
    pair_id: str | None = None
    variant: str = "canonical"

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "Story":
        return cls(**json.loads(line))


# ---- Anthropic's verbatim story-generation prompt ----
# Source: appendix of "Emotion Vectors" (transformer-circuits.pub/2026/emotions/)

CANONICAL_STORY_PROMPT = """\
Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]

[story 2]

[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:

- The character's actions and behaviors

- Physical sensations and body language

- Dialogue and tone of voice

- Thoughts and internal reactions

- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""


# ---- Anthropic's verbatim neutral-dialogues prompt ----
# Used later for PCA cleanup in probe computation.
# We don't use this in generate_stories() but expose it so the
# probes module can call it when needed.

NEUTRAL_DIALOGUES_PROMPT = """\
Write {n_stories} different dialogues based on the following topic.

Topic: {topic}

The dialogue should be between two characters:

- Person (a human)

- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.

The first speaker turn should always be from Person.

Format the dialogues like so:

[optional system instructions]

Person: [line]

AI: [line]

Person: [line]

AI: [line]

[continue for 2-6 exchanges]

[dialogue 2]

etc.

IMPORTANT: Always put a blank line before each speaker turn. Each turn should start with "Person:" or "AI:" on its own line after a blank line.

Generate a diverse mix of dialogue types across the {n_stories} examples:

- Some, but not all should include a system prompt at the start. These should come before the first Person turn. No tag like "System:" is needed, just put the instructions at the top. You can use "you" or "The assistant" to refer to the AI in the system prompt.

- Some should be about code or programming tasks

- Some should be factual questions (science, history, math, geography)

- Some should be work-related tasks (writing, analysis, summarization)

- Some should be practical how-to questions

- Some should be creative but neutral tasks (brainstorming names, generating lists)

- If it's natural to do so given the topic, it's ok for the dialogue to be a single back and forth (Person asks a question, AI answers), but at least some should have multiple exchanges.

CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.

- NO emotional content whatsoever - not explicit, not implied, not subtle

- The Person should not express any feelings (no frustration, excitement, gratitude, worry, etc.)

- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)

- The system prompt, if present, should not mention emotions at all, nor contain any emotionally charged language

- Avoid emotionally-charged topics entirely

- Use matter-of-fact, neutral language throughout

- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)

- Focus purely on information exchange and task completion"""


# ---- Anthropic's verbatim 100 topics ----
# Source: appendix of "Emotion Vectors" (transformer-circuits.pub/2026/emotions/)

ANTHROPIC_100_TOPICS: list[str] = [
    "An artist discovers someone has tattooed their work",
    "A family member announces they're converting to a different religion",
    "Someone's childhood imaginary friend appears in their niece's drawings",
    "A person finds out their biography was written without their knowledge",
    "A neighbor starts a renovation project",
    "Someone finds their grandmother's engagement ring in a pawn shop",
    "A student learns their scholarship application was denied",
    "A person's online friend turns out to live in the same city",
    "A neighbor wants to install a fence",
    "An adult child moves back in with their parents",
    "An employee is asked to train their replacement",
    "An athlete is asked to switch positions",
    "A traveler's flight is delayed, causing them to miss an important event",
    "A student is accused of plagiarism",
    "A person discovers their mentor has retired without saying goodbye",
    "Two friends both apply for the same job",
    "A person runs into their ex at a mutual friend's wedding",
    "Someone discovers their friend has been lying about their job",
    "A person discovers their partner has been taking secret phone calls",
    "A person discovers their child has the same teacher they had",
    "A person's car is towed from their own driveway",
    "Two friends realize they remember a shared event completely differently",
    "Someone discovers their mother kept every school assignment",
    "A person discovers their teenage diary has been published online",
    "Someone finds out their medical records were mixed up with another patient's",
    "A person finds out their article was published under someone else's name",
    "An athlete doesn't make the team they expected to join",
    "An employee is transferred to a different department",
    "Someone receives a friend request from a childhood bully",
    "A person finds out their surprise party has been cancelled",
    "An employee finds out a junior colleague makes more money",
    "A person finds out their partner has been learning their native language",
    "A chef receives a harsh review from a food critic",
    "A person learns their favorite restaurant is closing",
    "Someone finds their childhood teddy bear at a yard sale",
    "A homeowner discovers previous residents left items in the attic",
    "Someone finds an unsigned birthday card in their mailbox",
    "Someone discovers a hidden room in their new house",
    "Two strangers realize they've been dating the same person",
    "A person finds a hidden letter in a used book",
    "Two siblings inherit their grandmother's house",
    "Someone finds a wallet containing a large sum of cash",
    "Someone receives an invitation to their high school reunion",
    "Someone discovers their recipe has become famous under another name",
    "A college student discovers their roommate has been reading their journal",
    "A person finds out they were adopted through a DNA test",
    "A family member wants to sell a cherished heirloom",
    "Someone receives a package intended for the previous tenant",
    "Someone's childhood home is about to be demolished",
    "A person's invention is already patented by someone else",
    "A neighbor's dog keeps escaping into their yard",
    "A coach has to cut a player from the team",
    "Someone learns their favorite author plagiarized their stories",
    "A student finds out their scholarship was meant for someone else",
    "Someone discovers their teenager has a secret social media account",
    "Two roommates disagree about getting a pet",
    "Two friends plan separate birthday parties on the same day",
    "A person learns their childhood best friend doesn't remember them",
    "A musician hears their song being performed by someone else",
    "A person's manuscript is rejected by their dream publisher",
    "A person finds old photos that contradict family stories",
    "A person is asked to give a speech at their parent's retirement party",
    "A student discovers their teacher follows them on social media",
    "A parent finds an old letter they wrote but never sent",
    "An employee discovers the company is being sold",
    "A person accidentally sends a text to the wrong recipient",
    "Two coworkers are stuck in an elevator for three hours",
    "A student learns their thesis advisor is leaving the university",
    "A person's longtime hobby becomes their child's obsession",
    "Two colleagues are both considered for the same promotion",
    "Two coworkers discover they went to the same summer camp",
    "A tenant receives an eviction notice",
    "Someone finds their parent's draft letter of resignation from decades ago",
    "Someone finds out their best friend is moving across the country",
    "A neighbor's tree falls on their property",
    "Someone receives an apology letter years after the incident",
    "A person discovers the tree they planted as a child has been cut down",
    "Two siblings discover different versions of their inheritance",
    "A person finds their childhood home listed for sale online",
    "A homeowner learns their house was a former crime scene",
    "Someone finds out they have a half-sibling they never knew about",
    "A person learns their childhood bully became a therapist",
    "Two people discover they've been working on identical projects",
    "A person finds their spouse's secret savings account",
    "A neighbor complains about noise levels",
    "Someone finds their deceased parent's bucket list",
    "A teacher receives an unexpected gift from a former student",
    "An artist's work is displayed without their permission",
    "Someone discovers their neighbor is secretly wealthy",
    "A student receives a much lower grade than expected",
    "A person learns their college is closing down",
    "A neighbor asks to cut down a tree on the property line",
    "Two strangers discover they share the same rare medical condition",
    "Someone receives flowers with no card attached",
    "Someone discovers their partner has been writing a novel about them",
    "Someone finds a time capsule they don't remember burying",
    "Someone finds their partner's bucket list",
    "A neighbor asks to use part of the yard for a garden",
    "A person learns their apartment building is going condo",
    "Someone finds their college application essay published as an example",
]

assert len(ANTHROPIC_100_TOPICS) == 100, (
    f"expected 100 topics, got {len(ANTHROPIC_100_TOPICS)}"
)


# ---- Neutral dialogue topics ----
# These ARE NOT from Anthropic — the paper describes the neutral-dialogue
# prompt but doesn't publish the topic list. We use a diverse set covering
# the categories the prompt asks for (code, factual questions, work tasks,
# practical how-to, neutral creative tasks) to give PCA a stable, varied
# substrate for computing cleanup components.

NEUTRAL_DIALOGUE_TOPICS: list[str] = [
    # code / programming
    "implementing a binary search in Python",
    "explaining the difference between a list and a tuple",
    "debugging a null pointer exception",
    "setting up a virtual environment for a project",
    "writing a regular expression to match dates",
    "converting a recursive function to iterative",
    "explaining what a hash table is and how it works",
    "configuring a basic CI workflow",
    "understanding the time complexity of common sorts",
    "reading a JSON file and printing its top-level keys",
    # factual / science / history / geography / math
    "the boiling point of water at different altitudes",
    "the structure of a plant cell",
    "the major rivers of the South American continent",
    "the dates of the Industrial Revolution",
    "computing compound interest over multiple years",
    "the difference between weather and climate",
    "the periodic table organization by group and period",
    "the path of light through a convex lens",
    "calculating the area of a trapezoid",
    "converting between Celsius and Fahrenheit",
    # work tasks (writing / analysis / summarization)
    "summarizing a long technical document into bullet points",
    "writing a short status update for a project",
    "outlining a memo about a process change",
    "formatting a spreadsheet of monthly figures",
    "drafting a meeting agenda with three discussion items",
    "comparing the features of two software options in a table",
    "writing a brief project description for a portfolio entry",
    "preparing release notes for a minor version bump",
    "summarizing a research abstract in plain language",
    "writing a one-paragraph executive summary",
    # practical how-to
    "changing the air filter in a home HVAC system",
    "calibrating a kitchen scale",
    "setting up a new printer on a home network",
    "how to safely store batteries for long-term use",
    "how to properly inflate a bicycle tire",
    "how to read a basic nutrition label",
    "how to file a change-of-address form",
    "how to fold a fitted sheet",
    "how to descale a coffee machine",
    "how to organize a closet by category",
    # neutral creative tasks
    "brainstorming names for a community library",
    "generating a list of fifteen synonyms for the word 'large'",
    "coming up with chapter titles for a how-to manual",
    "generating sample test data for a contacts database",
    "brainstorming names for a software project related to file sync",
    "generating a list of practice questions for a geography quiz",
    "drafting a list of items for a moving checklist",
    "brainstorming categories for sorting a wardrobe",
    "generating filler text variants for a placeholder paragraph",
    "drafting a list of topics for a weekly newsletter",
]

assert len(NEUTRAL_DIALOGUE_TOPICS) == 50, (
    f"expected 50 neutral topics, got {len(NEUTRAL_DIALOGUE_TOPICS)}"
)


# ---- Public API ----

def generate_stories(
    config: Config,
    topics: list[str] | None = None,
    emotions: list[str] | None = None,
    verbose: bool = True,
) -> Path:
    """
    Generate the full story dataset specified by config.

    Writes to {config.stories_dir}/stories.jsonl, one Story per line.
    Returns the path written.

    Resumable: if the output file already contains stories for some
    (emotion, topic) cells, those cells are skipped. Safe to ctrl-C
    and re-run.

    Args:
        config: settings (model, paths, counts).
        topics: list of topic strings. Defaults to ANTHROPIC_100_TOPICS,
            truncated to config.n_topics.
        emotions: list of emotion strings. Defaults to
            config.emotion_subset, or EMOTIONS_12 if that's None.
        verbose: print progress.

    Each (emotion, topic) cell is generated in a single API call that
    produces n_stories_per_topic_per_emotion stories at once, following
    Anthropic's batched-generation prompt format.
    """
    if topics is None:
        topics = ANTHROPIC_100_TOPICS[: config.n_topics]
    if emotions is None:
        emotions = config.emotion_subset or EMOTIONS_12

    output_path = config.stories_dir / "stories.jsonl"

    # Count what's already on disk per (emotion, topic) cell.
    counts: dict[tuple[str, str], int] = {}
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if not line.strip():
                continue
            s = Story.from_jsonl(line)
            counts[(s.emotion, s.topic)] = counts.get((s.emotion, s.topic), 0) + 1

    target_per_cell = config.n_stories_per_topic_per_emotion
    total_target = len(emotions) * len(topics) * target_per_cell
    total_done = sum(counts.values())

    if verbose:
        print(
            f"Target: {total_target} stories "
            f"({len(emotions)} emotions x {len(topics)} topics x "
            f"{target_per_cell} per cell)"
        )
        print(f"Already have: {total_done} stories")
        print(f"To generate: {total_target - total_done} stories")
        print()

    with output_path.open("a", encoding="utf-8") as f:
        for emotion in emotions:
            for topic in topics:
                already = counts.get((emotion, topic), 0)
                needed = target_per_cell - already
                if needed <= 0:
                    continue

                if verbose:
                    print(f"  [{emotion} / {topic[:50]}...] generating {needed}...")

                # One API call gets us all needed stories for this cell.
                stories_text = _generate_one(
                    prompt=CANONICAL_STORY_PROMPT.format(
                        n_stories=needed,
                        emotion=emotion,
                        topic=topic,
                    ),
                    config=config,
                )
                story_texts = _parse_batched_stories(stories_text, expected=needed)

                for text in story_texts:
                    story = Story(emotion=emotion, topic=topic, text=text)
                    f.write(story.to_jsonl() + "\n")
                    total_done += 1
                f.flush()  # save after each cell so ctrl-C is safe

    if verbose:
        print(f"\nDone. {total_done} stories at {output_path}")
    return output_path


def generate_neutral_dialogues(
    config: Config,
    topics: list[str] | None = None,
    n_per_topic: int = 10,
    output_filename: str = "neutral_dialogues.jsonl",
    verbose: bool = True,
) -> Path:
    """
    Generate the neutral-dialogue corpus used for PCA cleanup.

    Each dialogue is saved as one Story row with ``emotion="neutral"``
    and ``variant="neutral"`` so the same loaders work on this dataset.

    Args:
        topics: defaults to NEUTRAL_DIALOGUE_TOPICS.
        n_per_topic: dialogues generated per topic. With 50 topics and
            n_per_topic=10 the total is 500 dialogues — typically enough
            for stable principal components on Gemma-class models. Tune
            up if the cleanup PCs look noisy.
        output_filename: file under ``config.stories_dir`` to write to.

    Resumable: if the file already contains entries for some topics
    those cells are skipped. Safe to ctrl-C and re-run.
    """
    if topics is None:
        topics = NEUTRAL_DIALOGUE_TOPICS

    output_path = config.stories_dir / output_filename

    # Count what's already on disk per topic.
    counts: dict[str, int] = {}
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if not line.strip():
                continue
            s = Story.from_jsonl(line)
            counts[s.topic] = counts.get(s.topic, 0) + 1

    total_target = len(topics) * n_per_topic
    total_done = sum(counts.values())

    if verbose:
        print(
            f"Target: {total_target} neutral dialogues "
            f"({len(topics)} topics x {n_per_topic} per topic)"
        )
        print(f"Already have: {total_done}")
        print(f"To generate: {total_target - total_done}")
        print()

    with output_path.open("a", encoding="utf-8") as f:
        for topic in topics:
            already = counts.get(topic, 0)
            needed = n_per_topic - already
            if needed <= 0:
                continue
            if verbose:
                print(f"  [{topic[:60]}...] generating {needed}...")

            raw = _generate_one(
                prompt=NEUTRAL_DIALOGUES_PROMPT.format(
                    n_stories=needed, topic=topic,
                ),
                config=config,
            )
            texts = _parse_batched_stories(
                raw, expected=needed, header_word="dialogue",
            )
            for text in texts:
                story = Story(
                    emotion="neutral",
                    topic=topic,
                    text=text,
                    variant="neutral",
                )
                f.write(story.to_jsonl() + "\n")
                total_done += 1
            f.flush()

    if verbose:
        print(f"\nDone. {total_done} dialogues at {output_path}")
    return output_path


def load_stories(path: Path) -> list[Story]:
    """Load all stories from a JSONL file."""
    return [
        Story.from_jsonl(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def stories_by_emotion(stories: list[Story]) -> dict[str, list[Story]]:
    """Group stories by their emotion label."""
    out: dict[str, list[Story]] = {}
    for s in stories:
        out.setdefault(s.emotion, []).append(s)
    return out


# ---- Parsing batched output ----

def _parse_batched_stories(
    raw: str, expected: int, header_word: str = "story",
) -> list[str]:
    """
    Parse a batched-generation response into individual stories or dialogues.

    Anthropic's prompts ask for output formatted as:
        [story 1]

        [story 2]

        etc.
    (or [dialogue 1], [dialogue 2], ... for the neutral-dialogue prompt).

    Models format this several ways, so we handle:
        - Literal "[story N]" / "Story N:" / "[dialogue N]" headers
        - Numbered markdown headers ("1.", "## Story 1")
        - Pure paragraph separation (double newlines)

    Strategy: try header-based splitting first; if it yields the
    expected count, use it. Otherwise fall back to paragraph splitting.

    For paragraph fallback with multi-paragraph dialogues, paragraph
    splitting will over-fragment — pass the right header_word so the
    primary header-split path wins.
    """
    hw = re.escape(header_word)
    # Strict: only matches an explicit story/dialogue header like
    # "[story 1]", "Story 1:", "**Story 1**", "## Story 1".
    strict_pattern = re.compile(
        r"(?:^|\n)\s*"
        rf"(?:\[?{hw}\s*\d+\]?[:.]?"
        rf"|\*+\s*{hw}\s*\d+\s*\*+[:.]?"
        rf"|#+\s*{hw}\s*\d+[:.]?"
        r")",
        re.IGNORECASE,
    )
    # Loose: bare numbered list ("1. ", "(2) "). Used only as a fallback
    # because content can contain inline numbered lists (e.g. markdown
    # answers in a neutral dialogue) that this pattern would shred.
    loose_pattern = re.compile(r"(?:^|\n)\s*\d+[\.\)]\s")

    def _try_split(pattern: re.Pattern) -> list[str] | None:
        parts = [c.strip() for c in pattern.split(raw) if c.strip()]
        if len(parts) == expected:
            return parts
        if len(parts) == expected + 1:
            return parts[1:]  # discard preamble before item 1
        return None

    out = _try_split(strict_pattern)
    if out is not None:
        return out
    out = _try_split(loose_pattern)
    if out is not None:
        return out

    # Paragraph fallback (double newline). Filter only on the strict
    # header pattern — the loose pattern would falsely drop content
    # paragraphs that happen to start with a numbered list.
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    paragraphs = [p for p in paragraphs if not strict_pattern.match("\n" + p)]
    if len(paragraphs) == expected:
        return paragraphs

    if len(paragraphs) > expected:
        paragraphs.sort(key=len, reverse=True)
        return paragraphs[:expected]

    print(
        f"  WARNING: expected {expected} items, parsed {len(paragraphs)}. "
        f"Raw output was {len(raw)} chars."
    )
    return paragraphs


# ---- Backend dispatch ----

def _generate_one(prompt: str, config: Config, max_retries: int = 5) -> str:
    """
    Make one API call and return the raw text response.

    Retries with exponential backoff on transient errors.
    """
    backend = config.generation_api
    last_err = None
    for attempt in range(max_retries):
        try:
            if backend == "anthropic":
                return _generate_via_anthropic(prompt, config)
            elif backend == "openai":
                return _generate_via_openai(prompt, config)
            elif backend is None:
                return _generate_via_local_model(prompt, config)
            else:
                raise ValueError(f"Unknown generation_api: {backend}")
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  (retry {attempt + 1}/{max_retries} after {wait}s: {e})")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}")


def _generate_via_anthropic(prompt: str, config: Config) -> str:
    """Call the Anthropic API to generate stories."""
    from anthropic import Anthropic

    api_key = _get_anthropic_key(config)
    client = Anthropic(api_key=api_key)
    # Generous max_tokens because we may be generating 12 stories at once.
    # ~150 words/story * 12 stories * ~1.3 tokens/word = ~2400 tokens.
    # Add headroom for headers and variation.
    msg = client.messages.create(
        model=config.generation_model,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    text_parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "".join(text_parts).strip()


def _get_anthropic_key(config: Config) -> str:
    """Resolve the Anthropic API key from env or .env file."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    env_path = config.project_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                if api_key.startswith(('"', "'")) and api_key.endswith(api_key[0]):
                    api_key = api_key[1:-1]
                return api_key
    raise RuntimeError(
        "ANTHROPIC_API_KEY not set in environment or .env file"
    )


def _generate_via_openai(prompt: str, config: Config) -> str:
    """Call the OpenAI API to generate stories. Not yet implemented."""
    raise NotImplementedError("OpenAI backend not implemented yet")


def _generate_via_local_model(prompt: str, config: Config) -> str:
    """Generate using the same model we'll probe. Not yet implemented."""
    raise NotImplementedError("Local-model generation not implemented yet")
