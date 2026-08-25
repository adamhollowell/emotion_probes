#!/usr/bin/env python3
"""
Regenerate the 12 original emotions under the CURRENT model snapshot, for
consistency with the 159-emotion extension (both then use whatever
claude-sonnet-4-5 resolves to today, e.g. claude-sonnet-4-5-20250929).

SAFETY: writes ONLY to data/stories/stories_12_regenerated.jsonl. It refuses
to write to stories.jsonl. Your original 12-emotion corpus (the basis of the
essays and the validated vectors) is left completely untouched; assemble the
final 171-set deliberately once you've verified this output.

Identical pipeline to the 159 run: verbatim CANONICAL_STORY_PROMPT and
ANTHROPIC_100_TOPICS, same model alias, max_tokens=4000, 12 stories/cell,
same index-based custom_ids (keyed to EMOTIONS_171, so 'loving' is e088 etc.).

Modes: build | submit (--confirm) | status | retrieve
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path

from emotion_probes.config import Config
from emotion_probes.emotions import EMOTIONS_12, EMOTIONS_171
from emotion_probes.generation import (
    CANONICAL_STORY_PROMPT, ANTHROPIC_100_TOPICS, Story, _parse_batched_stories,
)

MODEL_DEFAULT = "claude-sonnet-4-5"
MAX_TOKENS_DEFAULT = 4000
N_STORIES_PER_CELL = 12
OUTPUT_FILENAME = "stories_12_regenerated.jsonl"      # NOT stories.jsonl
BATCH_LOG_FILENAME = "stories_12_regenerated.batches.json"
CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
UNSUPPORTED_PARAMS = {"stream", "speed", "store", "previous_thread_event_id",
                      "cache_hint", "context_hint", "research_preview_2026_02"}
BATCH_IN_RATE, BATCH_OUT_RATE = 1.50, 7.50


def make_custom_id(e, t):
    return f"e{EMOTIONS_171.index(e):03d}_t{ANTHROPIC_100_TOPICS.index(t):02d}"

def parse_custom_id(cid):
    m = re.fullmatch(r"e(\d{3})_t(\d{2})", cid)
    if not m: raise ValueError(f"bad custom_id {cid!r}")
    return EMOTIONS_171[int(m.group(1))], ANTHROPIC_100_TOPICS[int(m.group(2))]

def build_requests():
    reqs, manifest = [], {}
    for e in EMOTIONS_12:
        for t in ANTHROPIC_100_TOPICS:
            cid = make_custom_id(e, t)
            reqs.append({"custom_id": cid, "params": {
                "model": MODEL_DEFAULT, "max_tokens": MAX_TOKENS_DEFAULT,
                "messages": [{"role": "user", "content": CANONICAL_STORY_PROMPT.format(
                    n_stories=N_STORIES_PER_CELL, emotion=e, topic=t)}]}})
            manifest[cid] = {"emotion": e, "topic": t, "n_stories": N_STORIES_PER_CELL}
    return reqs, manifest

def validate_request(req, mn=MAX_TOKENS_DEFAULT):
    errs = []
    cid = req.get("custom_id")
    if not isinstance(cid, str) or not CUSTOM_ID_RE.fullmatch(cid):
        errs.append(f"bad custom_id {cid!r}")
    p = req.get("params", {})
    if not isinstance(p, dict): return errs + ["params not object"]
    if not isinstance(p.get("model"), str) or not p["model"]: errs.append("model missing")
    mt = p.get("max_tokens")
    if not isinstance(mt, int) or isinstance(mt, bool) or mt < 1: errs.append(f"max_tokens {mt!r}")
    elif mt < mn: errs.append(f"max_tokens {mt} < {mn}")
    msgs = p.get("messages")
    if not isinstance(msgs, list) or not msgs: errs.append("messages empty")
    elif msgs[0].get("role") != "user": errs.append("first msg not user")
    for k in UNSUPPORTED_PARAMS:
        if k in p: errs.append(f"{k} unsupported")
    try:
        if json.loads(json.dumps(req, ensure_ascii=False)) != req: errs.append("roundtrip")
    except (TypeError, ValueError) as e: errs.append(f"unserializable: {e}")
    return errs

def validate_all(reqs):
    probs = {r.get("custom_id"): validate_request(r) for r in reqs if validate_request(r)}
    ids = [r["custom_id"] for r in reqs]
    nbytes = sum(len(json.dumps(r, ensure_ascii=False).encode()) for r in reqs)
    ok = True
    if probs: ok = False; print(f"  ✗ {len(probs)} failed:", list(probs.items())[:3])
    if len(ids) != len(set(ids)): ok = False; print("  ✗ duplicate ids")
    print(f"  validated {len(reqs)} requests, {len(set(ids))} unique ids, {nbytes/1048576:.2f} MB")
    if not ok: raise SystemExit("Validation failed.")
    print("  ✓ all local validation passed")

def cost_estimate(reqs):
    in_chars = sum(len(r["params"]["messages"][0]["content"]) for r in reqs)
    in_tok = in_chars/3.8 + 12*len(reqs)
    exp = in_tok/1e6*BATCH_IN_RATE + 2400*len(reqs)/1e6*BATCH_OUT_RATE
    ceil = in_tok/1e6*BATCH_IN_RATE + reqs[0]["params"]["max_tokens"]*len(reqs)/1e6*BATCH_OUT_RATE
    print(f"  cost estimate (batch): expected ~${exp:.0f}, hard ceiling ~${ceil:.0f}")

def _client():
    import os
    from anthropic import Anthropic
    if not os.environ.get("ANTHROPIC_API_KEY") and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                break
    return Anthropic()

def _log(cfg): return cfg.stories_dir / BATCH_LOG_FILENAME
def _latest(cfg):
    p = _log(cfg)
    if not p.exists(): raise SystemExit("No batch submitted yet.")
    return json.loads(p.read_text())[-1]["batch_id"]

def cmd_build(args, cfg):
    reqs, man = build_requests()
    print(f"Built {len(reqs)} requests for {len(EMOTIONS_12)} emotions ({', '.join(EMOTIONS_12)}).")
    validate_all(reqs); cost_estimate(reqs)
    (cfg.stories_dir / "stories_12_regenerated.manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=0))
    print("Build only — nothing submitted.")

def cmd_submit(args, cfg):
    reqs, man = build_requests()
    print(f"Preparing to submit: {len(reqs)} requests ({len(EMOTIONS_12)} emotions x 100 topics).")
    validate_all(reqs); cost_estimate(reqs)
    (cfg.stories_dir / "stories_12_regenerated.manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=0))
    if not args.confirm:
        print("\nDry run. Add --confirm to submit and spend."); return
    client = _client()
    batch = client.messages.batches.create(requests=reqs)
    p = _log(cfg); log = json.loads(p.read_text()) if p.exists() else []
    log.append({"batch_id": batch.id, "n_requests": len(reqs),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    p.write_text(json.dumps(log, indent=2))
    print(f"\n  ✓ submitted. batch_id = {batch.id}  status={batch.processing_status}")

def cmd_status(args, cfg):
    b = _client().messages.batches.retrieve(args.batch_id or _latest(cfg))
    print(f"  processing_status: {b.processing_status}")
    print(f"  request_counts:    {b.request_counts}")

def cmd_retrieve(args, cfg):
    client = _client()
    bid = args.batch_id or _latest(cfg)
    out = cfg.stories_dir / OUTPUT_FILENAME
    if out.name == "stories.jsonl":
        raise SystemExit("Refusing: output must not be stories.jsonl")
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                s = Story.from_jsonl(l); done.add((s.emotion, s.topic))
        print(f"  {len(done)} cells already present; skipping those.")
    ns = nc = ne = 0
    with out.open("a", encoding="utf-8") as f:
        for result in client.messages.batches.results(bid):
            if result.result.type != "succeeded":
                ne += 1; print(f"  [{result.custom_id}] {result.result.type}"); continue
            e, t = parse_custom_id(result.custom_id)
            if (e, t) in done: continue
            raw = "".join(b.text for b in result.result.message.content
                          if getattr(b, "type", None) == "text").strip()
            for txt in _parse_batched_stories(raw, expected=N_STORIES_PER_CELL):
                f.write(Story(emotion=e, topic=t, text=txt).to_jsonl() + "\n"); ns += 1
            nc += 1; f.flush()
    print(f"  wrote {ns} stories from {nc} cells to {out}")
    if ne: print(f"  {ne} non-succeeded result(s).")

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
