#!/usr/bin/env python3
"""
Top up the four imperfect cells from the 171-extension run to a clean 12 each.

Targets (emotion, topic):
  - hysterical  / A student is accused of plagiarism                      (was errored: 0)
  - sullen      / Someone's childhood home is about to be demolished      (was errored: 0)
  - shocked     / Someone receives a package intended for the previous tenant  (was 8)
  - suspicious  / A person finds out their surprise party has been cancelled   (was 11)

Same verbatim prompt / model / parsing as the main run. Submits a 4-request
batch. On retrieve, for each cell that comes back successfully it REPLACES that
cell's rows in stories_171_extension.jsonl with a fresh 12 (deleting the old
partial rows only after the new ones are in hand). Backs the file up first.
Never touches stories.jsonl.

Modes: build | submit (--confirm) | status | retrieve
"""
from __future__ import annotations
import argparse, json, re, shutil, time
from pathlib import Path

from emotion_probes.config import Config
from emotion_probes.emotions import EMOTIONS_12, EMOTIONS_171
from emotion_probes.generation import (
    CANONICAL_STORY_PROMPT, ANTHROPIC_100_TOPICS, Story, _parse_batched_stories,
)

MODEL_DEFAULT = "claude-sonnet-4-5"
MAX_TOKENS_DEFAULT = 4000
N_STORIES_PER_CELL = 12
OUTPUT_FILENAME = "stories_171_extension.jsonl"
CLEANUP_LOG = "stories_171_extension.cleanup.batches.json"
CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

TARGET_CELLS = [
    ("hysterical", "A student is accused of plagiarism"),
    ("sullen",     "Someone's childhood home is about to be demolished"),
    ("shocked",    "Someone receives a package intended for the previous tenant"),
    ("suspicious", "A person finds out their surprise party has been cancelled"),
]


def make_custom_id(e, t):
    return f"e{EMOTIONS_171.index(e):03d}_t{ANTHROPIC_100_TOPICS.index(t):02d}"


def parse_custom_id(cid):
    m = re.fullmatch(r"e(\d{3})_t(\d{2})", cid)
    return EMOTIONS_171[int(m.group(1))], ANTHROPIC_100_TOPICS[int(m.group(2))]


def _check_targets():
    remaining = set(EMOTIONS_171) - set(EMOTIONS_12)
    for e, t in TARGET_CELLS:
        assert e in remaining, f"{e} is not one of the 159"
        assert t in ANTHROPIC_100_TOPICS, f"topic not verbatim: {t!r}"


def build_requests():
    reqs = []
    for e, t in TARGET_CELLS:
        reqs.append({
            "custom_id": make_custom_id(e, t),
            "params": {
                "model": MODEL_DEFAULT, "max_tokens": MAX_TOKENS_DEFAULT,
                "messages": [{"role": "user",
                              "content": CANONICAL_STORY_PROMPT.format(
                                  n_stories=N_STORIES_PER_CELL, emotion=e, topic=t)}],
            },
        })
    return reqs


def _client():
    import os
    from anthropic import Anthropic
    if not os.environ.get("ANTHROPIC_API_KEY") and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                break
    return Anthropic()


def _log_path(cfg): return cfg.stories_dir / CLEANUP_LOG
def _latest(cfg):
    p = _log_path(cfg)
    if not p.exists(): raise SystemExit("No cleanup batch submitted yet.")
    return json.loads(p.read_text())[-1]["batch_id"]


def cmd_build(args, cfg):
    _check_targets()
    reqs = build_requests()
    print(f"{len(reqs)} cleanup requests:")
    for r in reqs:
        e, t = parse_custom_id(r["custom_id"])
        assert CUSTOM_ID_RE.fullmatch(r["custom_id"])
        print(f"  {r['custom_id']}  {e} / {t}")
    print("Build only — nothing submitted. (~$0.06 expected when you submit.)")


def cmd_submit(args, cfg):
    _check_targets()
    reqs = build_requests()
    cmd_build(args, cfg)
    if not args.confirm:
        print("\nDry run. Add --confirm to submit.")
        return
    client = _client()
    batch = client.messages.batches.create(requests=reqs)
    p = _log_path(cfg)
    log = json.loads(p.read_text()) if p.exists() else []
    log.append({"batch_id": batch.id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    p.write_text(json.dumps(log, indent=2))
    print(f"\n  ✓ submitted. batch_id = {batch.id}  status={batch.processing_status}")


def cmd_status(args, cfg):
    client = _client()
    b = client.messages.batches.retrieve(args.batch_id or _latest(cfg))
    print(f"  processing_status: {b.processing_status}")
    print(f"  request_counts:    {b.request_counts}")


def cmd_retrieve(args, cfg):
    client = _client()
    bid = args.batch_id or _latest(cfg)
    out = cfg.stories_dir / OUTPUT_FILENAME
    assert out.name != "stories.jsonl"

    # Gather fresh stories per succeeded target cell FIRST (nothing deleted yet).
    new_by_cell, errors = {}, []
    for result in client.messages.batches.results(bid):
        if result.result.type != "succeeded":
            errors.append((result.custom_id, result.result.type)); continue
        e, t = parse_custom_id(result.custom_id)
        raw = "".join(b.text for b in result.result.message.content
                      if getattr(b, "type", None) == "text").strip()
        new_by_cell[(e, t)] = _parse_batched_stories(raw, expected=N_STORIES_PER_CELL)

    if not new_by_cell:
        print("  No succeeded cells to write."); 
        if errors: print("  errors:", errors)
        return

    # Backup, then rebuild: keep untouched rows verbatim, swap in the new cells.
    backup = out.with_suffix(f".jsonl.bak.{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(out, backup)
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    targets = set(new_by_cell)
    kept = [l for l in lines if (lambda d: (d["emotion"], d["topic"]) not in targets)(json.loads(l))]
    new_lines = [Story(emotion=e, topic=t, text=txt).to_jsonl()
                 for (e, t), sts in new_by_cell.items() for txt in sts]
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(kept + new_lines) + "\n", encoding="utf-8")
    tmp.replace(out)

    print(f"  backed up -> {backup.name}")
    for (e, t), sts in new_by_cell.items():
        flag = "" if len(sts) == N_STORIES_PER_CELL else f"  <-- only {len(sts)}, re-run if needed"
        print(f"  replaced {e} / {t[:40]}...: {len(sts)} stories{flag}")
    if errors:
        print("  still not succeeded:", errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "submit", "status", "retrieve"])
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--batch-id", default=None)
    args = ap.parse_args()
    cfg = Config()
    {"build": cmd_build, "submit": cmd_submit,
     "status": cmd_status, "retrieve": cmd_retrieve}[args.mode](args, cfg)


if __name__ == "__main__":
    main()
