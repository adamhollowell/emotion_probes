#!/usr/bin/env python3
"""
Assemble the snapshot-consistent 171-emotion corpus into ONE file.

Inputs  (both must be complete):
  data/stories/stories_12_regenerated.jsonl   (the 12, regenerated under current snapshot)
  data/stories/stories_171_extension.jsonl    (the 159)
Output:
  data/stories/stories_171_canonical.jsonl    (205,200 stories = 171 x 100 x 12)

SAFETY: never reads or writes stories.jsonl (your original, essay-basis 12 stay
archived and untouched). Refuses to assemble unless BOTH inputs pass their
completeness checks, so it cannot emit a half-baked corpus. Source lines are
copied verbatim — no reserialization of story text.

Modes:
  verify     check each input file independently; report readiness
  assemble   verify both, then write the canonical file and verify the union
"""
from __future__ import annotations
import argparse, json
from collections import Counter

from emotion_probes.config import Config
from emotion_probes.emotions import EMOTIONS_12, EMOTIONS_171
from emotion_probes.generation import ANTHROPIC_100_TOPICS

REGEN_12 = "stories_12_regenerated.jsonl"
EXT_159 = "stories_171_extension.jsonl"
CANONICAL = "stories_171_canonical.jsonl"
NTOP = len(ANTHROPIC_100_TOPICS)
N_PER_CELL = 12
EXT_EMOTIONS = [e for e in EMOTIONS_171 if e not in set(EMOTIONS_12)]


def _lines(path):
    return [l for l in path.read_text().splitlines() if l.strip()]


def check_set(rows, expected_emotions, label) -> bool:
    emos = set(r["emotion"] for r in rows)
    cells = Counter((r["emotion"], r["topic"]) for r in rows)
    exp = set(expected_emotions)
    problems = []
    if emos != exp:
        if exp - emos: problems.append(f"missing emotions: {sorted(exp - emos)[:5]}")
        if emos - exp: problems.append(f"unexpected emotions: {sorted(emos - exp)[:5]}")
    expected_cells = len(expected_emotions) * NTOP
    if len(cells) != expected_cells:
        problems.append(f"{len(cells)} cells, expected {expected_cells}")
    off = [(k, v) for k, v in cells.items() if v != N_PER_CELL]
    if off:
        problems.append(f"{len(off)} cells not exactly {N_PER_CELL}: {off[:3]}")
    expected_stories = len(expected_emotions) * NTOP * N_PER_CELL
    status = "OK" if not problems else "PROBLEMS"
    print(f"[{label}] {len(rows)} stories (expect {expected_stories}), "
          f"{len(emos)} emotions, {len(cells)} cells -> {status}")
    for p in problems:
        print("    -", p)
    return not problems


def cmd_verify(args, cfg):
    p12, p159 = cfg.stories_dir / REGEN_12, cfg.stories_dir / EXT_159
    ok12 = ok159 = True
    if p12.exists():
        ok12 = check_set([json.loads(l) for l in _lines(p12)], EMOTIONS_12, "regenerated 12")
    else:
        print(f"[regenerated 12] MISSING ({p12}) — retrieve that batch first"); ok12 = False
    if p159.exists():
        ok159 = check_set([json.loads(l) for l in _lines(p159)], EXT_EMOTIONS, "extension 159")
    else:
        print(f"[extension 159] MISSING ({p159})"); ok159 = False
    print(f"\nready to assemble: {ok12 and ok159}")


def cmd_assemble(args, cfg):
    p12, p159 = cfg.stories_dir / REGEN_12, cfg.stories_dir / EXT_159
    out = cfg.stories_dir / CANONICAL
    if out.name == "stories.jsonl":
        raise SystemExit("Refusing: output must not be stories.jsonl")
    for p in (p12, p159):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}. Retrieve it first.")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; use --force to overwrite.")

    rows12 = [json.loads(l) for l in _lines(p12)]
    rows159 = [json.loads(l) for l in _lines(p159)]
    ok = check_set(rows12, EMOTIONS_12, "regenerated 12")
    ok = check_set(rows159, EXT_EMOTIONS, "extension 159") and ok
    overlap = set(r["emotion"] for r in rows12) & set(r["emotion"] for r in rows159)
    if overlap:
        ok = False; print(f"    - OVERLAP between inputs: {sorted(overlap)}")
    if not ok and not args.force:
        raise SystemExit("\nInputs incomplete — not assembling. (Override with --force.)")

    # Verbatim copy of both sources' lines.
    combined = _lines(p12) + _lines(p159)
    out.write_text("\n".join(combined) + "\n", encoding="utf-8")

    print("\n--- combined corpus ---")
    comb_rows = [json.loads(l) for l in _lines(out)]
    union_ok = check_set(comb_rows, EMOTIONS_171, "canonical 171")
    sj = cfg.stories_dir / "stories.jsonl"
    if sj.exists():
        print(f"stories.jsonl untouched: {sj.stat().st_size} bytes")
    print(f"\nwrote {out}")
    print("RESULT:", "✓ canonical 171 corpus complete" if union_ok else "✗ see problems above")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "assemble"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    {"verify": cmd_verify, "assemble": cmd_assemble}[args.mode](args, cfg)


if __name__ == "__main__":
    main()
