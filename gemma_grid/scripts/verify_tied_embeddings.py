#!/usr/bin/env python3
"""Assert tie_word_embeddings for all six Gemma-2 grid checkpoints — at the
weight level, per checkpoint, per pinned revision (not from the Gemma2Config
class default).

Method: the HF API reports each checkpoint's total parameter count parsed from
the actual safetensors headers. Tied vs untied differ by exactly one
vocab×d_model block (a separate lm_head). We recompute the architectural total
both ways and require the tied total to match EXACTLY and the untied total to
miss. Base and IT are checked independently at their pinned SHAs
(gemma_grid/handoff/hf_model_revisions_ASSUMED.json).

This does not read config.json (gated without a license-accepted token). With
a token, the belt-and-suspenders version is:
  huggingface_hub.hf_hub_download(repo, "config.json", revision=SHA)
  -> assert json["tie_word_embeddings"] is True  (or absent = default True)
  and confirm model.safetensors.index.json has no lm_head.weight entry.

Exit 0 = all six checkpoints tied; exit 1 otherwise.
"""
import json, sys, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REVS = json.loads((REPO / "gemma_grid" / "handoff" / "hf_model_revisions_ASSUMED.json").read_text())

V = 256000
ARCH = {  # d_model, n_layers, heads, kv_heads, head_dim, d_ff (Gemma-2 family)
    "2b":  dict(d=2304, L=26, h=8,  kv=4,  hd=256, ff=9216),
    "9b":  dict(d=3584, L=42, h=16, kv=8,  hd=256, ff=14336),
    "27b": dict(d=4608, L=46, h=32, kv=16, hd=128, ff=36864),
}

def expected(a, tied):
    per_layer = a["d"]*a["h"]*a["hd"] + 2*a["d"]*a["kv"]*a["hd"] + a["h"]*a["hd"]*a["d"] \
                + 3*a["d"]*a["ff"] + 4*a["d"]
    tot = V*a["d"] + a["L"]*per_layer + a["d"]
    return tot if tied else tot + V*a["d"]

fail = 0
for repo, meta in REVS.items():
    size = repo.split("gemma-2-")[1].split("-")[0]
    a = ARCH[size]
    url = f"https://huggingface.co/api/models/{repo}?revision={meta['current_main_sha']}"
    with urllib.request.urlopen(url, timeout=30) as r:
        api = json.load(r)
    reported = api["safetensors"]["total"]
    t, u = expected(a, True), expected(a, False)
    ok = (reported == t) and (reported != u)
    fail += (not ok)
    print(f"{'PASS' if ok else 'FAIL'} {repo} @ {meta['current_main_sha'][:8]}: "
          f"reported={reported:,} tied={t:,} untied={u:,}")
print("VERDICT:", "ALL TIED" if fail == 0 else f"{fail} FAILED")
sys.exit(1 if fail else 0)
