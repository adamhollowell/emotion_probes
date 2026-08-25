import json, random, sys
from pathlib import Path
from collections import defaultdict

CORPUS = "data/stories/stories_171_canonical_clean.jsonl"
SEED = 20260622          # frozen: change this only if you deliberately want a new draw
N_EMOTIONS = 30
N_STORIES = 10
OUTDIR = Path.home() / "emotion_validation"

rng = random.Random(SEED)

by_emotion = defaultdict(list)
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_emotion[rec["emotion"]].append(rec)

all_emotions = sorted(by_emotion.keys())
print(f"Loaded {sum(len(v) for v in by_emotion.values())} stories across {len(all_emotions)} emotions.")

chosen_emotions = sorted(rng.sample(all_emotions, N_EMOTIONS))

selection = []
for emo in chosen_emotions:
    stories = by_emotion[emo]
    by_topic = defaultdict(list)
    for s in stories:
        by_topic[s["topic"]].append(s)
    topics = sorted(by_topic.keys())
    if len(topics) < N_STORIES:
        sys.exit(f"ERROR: {emo} has only {len(topics)} distinct topics; cannot draw {N_STORIES}.")
    picked_topics = rng.sample(topics, N_STORIES)
    for t in picked_topics:
        chosen = rng.choice(by_topic[t])
        selection.append({
            "emotion": chosen["emotion"],
            "topic": chosen["topic"],
            "text": chosen["text"],
            "variant": chosen.get("variant"),
        })

meta = {
    "seed": SEED,
    "n_emotions": N_EMOTIONS,
    "n_stories_per_emotion": N_STORIES,
    "total_stories": len(selection),
    "sampling": "10 stories from 10 distinct topics per emotion",
    "source_file": CORPUS,
    "chosen_emotions": chosen_emotions,
}

(OUTDIR / "validation_set.json").write_text(
    json.dumps({"meta": meta, "stories": selection}, ensure_ascii=False, indent=2),
    encoding="utf-8")

lines = ["# Validation Sample — Manual Review Set", ""]
lines.append("This is the random subsample used for manual validation of the emotion corpus,")
lines.append("matching Anthropic's protocol (ten stories for thirty emotions). It may also serve")
lines.append("as the appendix sample of example stories.")
lines.append("")
lines.append("## Selection metadata")
lines.append("")
for k, v in meta.items():
    if k == "chosen_emotions":
        lines.append(f"- **{k}:** {', '.join(v)}")
    else:
        lines.append(f"- **{k}:** {v}")
lines.append("")
lines.append("## Stories")
lines.append("")
cur = None
for s in selection:
    if s["emotion"] != cur:
        cur = s["emotion"]
        lines.append(f"### {cur}")
        lines.append("")
    lines.append(f"**Topic:** {s['topic']}")
    lines.append("")
    lines.append(s["text"])
    lines.append("")
    lines.append("---")
    lines.append("")
(OUTDIR / "validation_set.md").write_text("\n".join(lines), encoding="utf-8")

print(f"Wrote {len(selection)} stories.")
print(f"  JSON  -> {OUTDIR / 'validation_set.json'}")
print(f"  MD    -> {OUTDIR / 'validation_set.md'}")
print(f"Chosen emotions: {', '.join(chosen_emotions)}")
