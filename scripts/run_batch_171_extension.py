#!/usr/bin/env python3
"""
Submit & retrieve the 171-emotion extension story batch (Anthropic Batches API).

Generates stories for the 159 emotions NOT in EMOTIONS_12, using the VERBATIM
CANONICAL_STORY_PROMPT and ANTHROPIC_100_TOPICS imported directly from
emotion_probes.generation (no copy of the prompt lives in this file). One batch
request per (emotion, topic) cell, each asking for 12 stories — identical in
shape to the original 12-emotion run.

Writes ONLY to data/stories/stories_171_extension.jsonl. Never touches
stories.jsonl.

Modes
-----
  build              Build + validate all requests locally. Does NOT submit.
  submit             Build + validate + submit a batch. Requires --confirm.
                     Use --pilot EMOTION to scope to a single emotion (100 reqs).
  status             Poll a submitted batch's processing status.
  retrieve           Stream a finished batch's results into the output JSONL.

Nothing in build/status/retrieve spends money. Only `submit --confirm` does,
and it prints the cost estimate and the exact request count first.

Examples
--------
  python scripts/run_batch_171_extension.py build
  python scripts/run_batch_171_extension.py submit --pilot alarmed --confirm
  python scripts/run_batch_171_extension.py status
  python scripts/run_batch_171_extension.py retrieve
  python scripts/run_batch_171_extension.py submit --confirm        # FULL run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from emotion_probes.config import Config
from emotion_probes.emotions import EMOTIONS_12, EMOTIONS_171
from emotion_probes.generation import (
    CANONICAL_STORY_PROMPT,
    ANTHROPIC_100_TOPICS,
    Story,
    _parse_batched_stories,
)

# ---- Constants ----
MODEL_DEFAULT = "claude-sonnet-4-5"
MAX_TOKENS_DEFAULT = 4000          # project floor; the API hard floor is 1
N_STORIES_PER_CELL = 12

OUTPUT_FILENAME = "stories_171_extension.jsonl"
MANIFEST_FILENAME = "stories_171_extension.manifest.json"
BATCH_LOG_FILENAME = "stories_171_extension.batches.json"

CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
UNSUPPORTED_PARAMS = {
    "stream", "speed", "store", "previous_thread_event_id",
    "cache_hint", "context_hint", "research_preview_2026_02",
}

# Sonnet 4.5 batch rates, $/MTok (50% of standard). For estimate only.
BATCH_IN_RATE, BATCH_OUT_RATE = 1.50, 7.50


# ---- Emotion / custom_id helpers ----
def remaining_emotions() -> list[str]:
    done = set(EMOTIONS_12)
    return [e for e in EMOTIONS_171 if e not in done]  # preserves 171 order


def make_custom_id(emotion: str, topic: str) -> str:
    ei = EMOTIONS_171.index(emotion)          # canonical 171 index
    ti = ANTHROPIC_100_TOPICS.index(topic)
    return f"e{ei:03d}_t{ti:02d}"


def parse_custom_id(cid: str) -> tuple[str, str]:
    m = re.fullmatch(r"e(\d{3})_t(\d{2})", cid)
    if not m:
        raise ValueError(f"unparseable custom_id: {cid!r}")
    return EMOTIONS_171[int(m.group(1))], ANTHROPIC_100_TOPICS[int(m.group(2))]


# ---- Build ----
def build_requests(emotions: list[str], model: str, max_tokens: int):
    """Return (requests, manifest). requests are plain dicts ready for the SDK."""
    requests, manifest = [], {}
    for e in emotions:
        for t in ANTHROPIC_100_TOPICS:
            cid = make_custom_id(e, t)
            requests.append({
                "custom_id": cid,
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{
                        "role": "user",
                        # VERBATIM template; only the 3 documented fields are filled.
                        "content": CANONICAL_STORY_PROMPT.format(
                            n_stories=N_STORIES_PER_CELL, emotion=e, topic=t),
                    }],
                },
            })
            manifest[cid] = {"emotion": e, "topic": t, "n_stories": N_STORIES_PER_CELL}
    return requests, manifest


# ---- Validate ----
def validate_request(req: dict, project_min_tokens: int = MAX_TOKENS_DEFAULT) -> list[str]:
    errs: list[str] = []
    cid = req.get("custom_id")
    if not isinstance(cid, str) or not CUSTOM_ID_RE.fullmatch(cid):
        errs.append(f"bad custom_id: {cid!r}")
    p = req.get("params")
    if not isinstance(p, dict):
        return errs + ["params missing/not an object"]
    if not isinstance(p.get("model"), str) or not p["model"]:
        errs.append("params.model missing/empty")
    mt = p.get("max_tokens")
    if not isinstance(mt, int) or isinstance(mt, bool) or mt < 1:
        errs.append(f"params.max_tokens invalid: {mt!r} (API floor is 1)")
    elif mt < project_min_tokens:
        errs.append(f"params.max_tokens={mt} < project min {project_min_tokens}")
    msgs = p.get("messages")
    if not isinstance(msgs, list) or not msgs:
        errs.append("params.messages missing/empty")
    else:
        if msgs[0].get("role") != "user":
            errs.append("first message role must be 'user'")
        for i, m in enumerate(msgs):
            if m.get("role") not in ("user", "assistant"):
                errs.append(f"message[{i}].role invalid")
            c = m.get("content")
            if not (isinstance(c, str) and c) and not (isinstance(c, list) and c):
                errs.append(f"message[{i}].content empty")
    for k in UNSUPPORTED_PARAMS:
        if k in p:
            errs.append(f"params.{k} unsupported in batch")
    try:
        if json.loads(json.dumps(req, ensure_ascii=False)) != req:
            errs.append("JSON round-trip mismatch")
    except (TypeError, ValueError) as exc:
        errs.append(f"not JSON-serializable: {exc}")
    return errs


def validate_all(requests: list[dict]) -> None:
    """Hard gate: raise SystemExit if anything is wrong."""
    problems = {r.get("custom_id", "?"): validate_request(r)
                for r in requests if validate_request(r)}
    ids = [r["custom_id"] for r in requests]
    total_bytes = sum(len(json.dumps(r, ensure_ascii=False).encode()) for r in requests)

    ok = True
    if problems:
        ok = False
        print(f"  ✗ {len(problems)} request(s) failed validation; first few:")
        for cid, errs in list(problems.items())[:5]:
            print(f"      {cid}: {errs}")
    if len(ids) != len(set(ids)):
        ok = False
        print("  ✗ duplicate custom_ids present")
    if len(requests) > 100_000:
        ok = False
        print(f"  ✗ {len(requests)} requests exceeds the 100,000 batch limit")
    if total_bytes > 256 * 1_048_576:
        ok = False
        print(f"  ✗ batch is {total_bytes/1_048_576:.1f} MB, over the 256 MB limit")

    print(f"  validated {len(requests)} requests, "
          f"{len(set(ids))} unique ids, {total_bytes/1_048_576:.2f} MB")
    if not ok:
        raise SystemExit("Validation failed — not submitting.")
    print("  ✓ all local validation passed")


def cost_estimate(requests: list[dict]) -> None:
    in_chars = sum(len(r["params"]["messages"][0]["content"]) for r in requests)
    in_tok = in_chars / 3.8 + 12 * len(requests)            # ~ midpoint
    out_tok_expected = 2400 * len(requests)
    out_tok_ceiling = requests[0]["params"]["max_tokens"] * len(requests)
    exp = in_tok/1e6*BATCH_IN_RATE + out_tok_expected/1e6*BATCH_OUT_RATE
    ceil = in_tok/1e6*BATCH_IN_RATE + out_tok_ceiling/1e6*BATCH_OUT_RATE
    print(f"  cost estimate (batch): expected ~${exp:.0f}, hard ceiling ~${ceil:.0f} "
          f"(input ~${in_tok/1e6*BATCH_IN_RATE:.0f})")


# ---- Client / key ----
def _client():
    try:
        from anthropic import Anthropic
    except ImportError:
        raise SystemExit("anthropic SDK not installed. `pip install anthropic`")
    # SDK reads ANTHROPIC_API_KEY from env automatically; fall back to .env.
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env = Path(".env")
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    os.environ["ANTHROPIC_API_KEY"] = val
                    break
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (export it or put it in .env).")
    return Anthropic()


# ---- Batch log (records submitted batch ids) ----
def _log_path(cfg: Config) -> Path:
    return cfg.stories_dir / BATCH_LOG_FILENAME


def _record_batch(cfg: Config, label: str, batch_id: str, n: int) -> None:
    path = _log_path(cfg)
    log = json.loads(path.read_text()) if path.exists() else []
    log.append({"label": label, "batch_id": batch_id, "n_requests": n,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    path.write_text(json.dumps(log, indent=2))


def _latest_batch_id(cfg: Config) -> str:
    path = _log_path(cfg)
    if not path.exists():
        raise SystemExit(f"No batch log at {path}. Submit a batch first, "
                         f"or pass --batch-id.")
    log = json.loads(path.read_text())
    if not log:
        raise SystemExit("Batch log is empty.")
    return log[-1]["batch_id"]


# ---- Commands ----
def cmd_build(args, cfg: Config) -> None:
    emotions = _select_emotions(args)
    requests, manifest = build_requests(emotions, args.model, args.max_tokens)
    print(f"Built {len(requests)} requests for {len(emotions)} emotion(s), "
          f"{N_STORIES_PER_CELL} stories/cell.")
    validate_all(requests)
    cost_estimate(requests)
    mpath = cfg.stories_dir / MANIFEST_FILENAME
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=0))
    print(f"  wrote manifest: {mpath}")
    print("Build only — nothing submitted.")


def cmd_submit(args, cfg: Config) -> None:
    emotions = _select_emotions(args)
    label = f"pilot:{args.pilot}" if args.pilot else "full_159"
    requests, manifest = build_requests(emotions, args.model, args.max_tokens)
    print(f"Preparing to submit [{label}]: {len(requests)} requests "
          f"({len(emotions)} emotion(s) x {len(ANTHROPIC_100_TOPICS)} topics).")
    validate_all(requests)
    cost_estimate(requests)

    # Always (re)write the manifest so retrieval can map results back.
    (cfg.stories_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=0))

    if not args.confirm:
        print("\nDry run. Re-run with --confirm to actually submit and spend.")
        return

    client = _client()
    print("Submitting batch ...")
    batch = client.messages.batches.create(requests=requests)
    _record_batch(cfg, label, batch.id, len(requests))
    print(f"  ✓ submitted. batch_id = {batch.id}")
    print(f"  status: {batch.processing_status}")
    print(f"  recorded in {_log_path(cfg)}")
    print("\nNext: poll with `status`, then `retrieve` once it has ended.")


def cmd_status(args, cfg: Config) -> None:
    client = _client()
    bid = args.batch_id or _latest_batch_id(cfg)
    b = client.messages.batches.retrieve(bid)
    print(f"batch {bid}")
    print(f"  processing_status: {b.processing_status}")
    print(f"  request_counts:    {b.request_counts}")


def cmd_retrieve(args, cfg: Config) -> None:
    client = _client()
    bid = args.batch_id or _latest_batch_id(cfg)
    out_path = cfg.stories_dir / OUTPUT_FILENAME
    if out_path.name == "stories.jsonl":
        raise SystemExit("Refusing: output must not be stories.jsonl")

    # Resumable: skip (emotion, topic) cells already written.
    done_cells = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                s = Story.from_jsonl(line)
                done_cells.add((s.emotion, s.topic))
        print(f"  output exists with {len(done_cells)} cells already present; "
              f"those will be skipped.")

    n_stories = n_cells = n_err = 0
    with out_path.open("a", encoding="utf-8") as f:
        for result in client.messages.batches.results(bid):
            rtype = result.result.type
            if rtype != "succeeded":
                n_err += 1
                print(f"  [{result.custom_id}] {rtype}")
                continue
            emotion, topic = parse_custom_id(result.custom_id)
            if (emotion, topic) in done_cells:
                continue
            text_blocks = [b.text for b in result.result.message.content
                           if getattr(b, "type", None) == "text"]
            raw = "".join(text_blocks).strip()
            stories = _parse_batched_stories(raw, expected=N_STORIES_PER_CELL)
            for txt in stories:
                f.write(Story(emotion=emotion, topic=topic, text=txt).to_jsonl() + "\n")
                n_stories += 1
            n_cells += 1
            f.flush()
    print(f"  wrote {n_stories} stories from {n_cells} cells to {out_path}")
    if n_err:
        print(f"  {n_err} non-succeeded result(s) — re-run retrieve to fill gaps "
              f"after re-submitting those cells if needed.")


# ---- emotion selection ----
def _select_emotions(args) -> list[str]:
    allowed = remaining_emotions()
    if args.pilot:
        if args.pilot not in allowed:
            raise SystemExit(
                f"--pilot '{args.pilot}' is not one of the 159 remaining emotions. "
                f"(It may be one of the 12 already done, or misspelled.)")
        return [args.pilot]
    return allowed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["build", "submit", "status", "retrieve"])
    ap.add_argument("--pilot", metavar="EMOTION",
                    help="scope to a single emotion (one of the 159 remaining)")
    ap.add_argument("--confirm", action="store_true",
                    help="actually submit (otherwise submit is a dry run)")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT, dest="max_tokens")
    ap.add_argument("--batch-id", default=None,
                    help="for status/retrieve; defaults to the latest submitted")
    args = ap.parse_args()

    if args.max_tokens < MAX_TOKENS_DEFAULT:
        raise SystemExit(f"--max-tokens {args.max_tokens} is below the project "
                         f"floor of {MAX_TOKENS_DEFAULT}.")

    cfg = Config()  # project_root = cwd; ensures data/stories exists
    {"build": cmd_build, "submit": cmd_submit,
     "status": cmd_status, "retrieve": cmd_retrieve}[args.mode](args, cfg)


if __name__ == "__main__":
    main()
