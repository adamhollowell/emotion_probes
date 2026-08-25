#!/usr/bin/env python3
"""
Strip leading perspective labels ("First person:", "Third-person -", etc.) from
the 171 canonical corpus, writing a NEW cleaned file. The original is read-only
here and never modified.

INPUT (read-only):  data/stories/stories_171_canonical.jsonl
OUTPUT (new):       data/stories/stories_171_canonical_clean.jsonl

What it does, precisely:
- For each line, parse the JSON, look at the `text` field.
- If `text` begins with a perspective label (the exact forms observed: first/
  second/third [optional space or hyphen] person, then ':' '.' or '-' plus
  trailing whitespace), remove ONLY that prefix; the rest of the story is
  untouched.
- Lines with no such prefix are passed through; their `text` is unchanged.
- Re-serialize each record with the same field set used elsewhere
  (Story.to_jsonl) so schema stays consistent.

Refuses to overwrite stories.jsonl or the input. Verifies the output afterward.

Usage:
    python3 scripts/strip_perspective_labels.py            # dry run: counts only
    python3 scripts/strip_perspective_labels.py --write     # write the clean file
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

from emotion_probes.config import Config
from emotion_probes.emotions import EMOTIONS_171
from emotion_probes.generation import ANTHROPIC_100_TOPICS, Story

IN_NAME = "stories_171_canonical.jsonl"
OUT_NAME = "stories_171_canonical_clean.jsonl"

# Matches ONLY a leading perspective label + its punctuation + trailing space.
LABEL = re.compile(r"^\s*(?:first|second|third)[\s-]*person\s*[:.\-]+\s*", re.I)

N_PER_CELL = 12
NTOP = len(ANTHROPIC_100_TOPICS)


def strip_one(text: str) -> tuple[str, bool]:
    """Return (clean_text, was_stripped). Removes only a leading label."""
    m = LABEL.match(text)
    if not m:
        return text, False
    return text[m.end():], True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the cleaned file (otherwise dry-run counts only)")
    args = ap.parse_args()

    cfg = Config()
    src = cfg.stories_dir / IN_NAME
    out = cfg.stories_dir / OUT_NAME
    if out.name in ("stories.jsonl", IN_NAME):
        raise SystemExit("Refusing: output collides with a protected file.")
    if not src.exists():
        raise SystemExit(f"Missing input: {src}")

    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    stripped = Counter()
    cleaned_records = []
    for r in rows:
        text, hit = strip_one(r["text"])
        if hit:
            stripped[r["emotion"]] += 1
        cleaned_records.append((r["emotion"], r["topic"], text))

    total_stripped = sum(stripped.values())
    print(f"input rows: {len(rows)}")
    print(f"labels stripped: {total_stripped}")
    if total_stripped:
        print("  by emotion (top 10):")
        for e, n in stripped.most_common(10):
            print(f"    {n:4d}  {e}")

    if not args.write:
        print("\nDry run. Re-run with --write to create the cleaned file.")
        return 0

    if out.exists():
        raise SystemExit(f"{out} already exists; remove it first if you want to redo.")

    # Write atomically via Story.to_jsonl so schema matches the rest of the corpus.
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for emotion, topic, text in cleaned_records:
            f.write(Story(emotion=emotion, topic=topic, text=text).to_jsonl() + "\n")
    tmp.replace(out)

    # ---- Verify the cleaned file ----
    crows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    cells = Counter((r["emotion"], r["topic"]) for r in crows)
    remaining = sum(1 for r in crows if LABEL.match(r["text"]))
    ok = (
        len(crows) == len(rows)
        and len(set(r["emotion"] for r in crows)) == len(EMOTIONS_171)
        and len(cells) == len(EMOTIONS_171) * NTOP
        and all(v == N_PER_CELL for v in cells.values())
        and remaining == 0
    )
    print(f"\n--- cleaned file written: {out} ---")
    print(f"rows: {len(crows)} (expect {len(rows)})")
    print(f"emotions: {len(set(r['emotion'] for r in crows))} (expect {len(EMOTIONS_171)})")
    print(f"cells: {len(cells)} (expect {len(EMOTIONS_171)*NTOP})")
    print(f"cells != 12: {sum(1 for v in cells.values() if v != N_PER_CELL)} (expect 0)")
    print(f"labels remaining: {remaining} (expect 0)")
    sj = cfg.stories_dir / "stories.jsonl"
    print(f"original {IN_NAME} untouched (still on disk): {src.exists()}")
    print("RESULT:", "✓ clean corpus complete" if ok else "✗ see mismatches above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
