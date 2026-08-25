import json, re
from pathlib import Path

data = json.loads((Path.home() / "emotion_validation" / "validation_set.json").read_text(encoding="utf-8"))
stories = data["stories"]

def stem(word):
    w = word.lower().strip()
    if " " in w:
        return w.split()[0]
    for suf in ["ened","ressed","icted","ated","ied","ed","ing","ful","less","ous","ive","ic","al","ly","ness","ent","ant","y","d"]:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w

exact_hits, stem_hits = [], []
for s in stories:
    emo = s["emotion"].lower()
    text = s["text"].lower()
    if re.search(r"\b" + re.escape(emo) + r"\b", text):
        exact_hits.append(s)
    st = stem(s["emotion"])
    for m in re.finditer(r"\b" + re.escape(st) + r"\w*\b", text):
        if not re.fullmatch(r"\b" + re.escape(emo) + r"\b", m.group()):
            stem_hits.append((s, m.group()))
            break

print(f"Checked {len(stories)} stories.\n")
print(f"=== EXACT emotion-word hits: {len(exact_hits)} ===")
for s in exact_hits:
    snippet = re.sub(r"\s+", " ", s["text"])[:90]
    print(f"  [{s['emotion']}] {snippet}...")
print(f"\n=== STEM/inflection hits (APPROXIMATE — eyeball these): {len(stem_hits)} ===")
for s, form in stem_hits:
    print(f"  [{s['emotion']}] matched '{form}'")
