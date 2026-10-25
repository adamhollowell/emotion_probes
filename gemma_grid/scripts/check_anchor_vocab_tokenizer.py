"""Check the 27B base (anchor) vocabulary map against the tokenizer file at the pinned revision.

The grid exporter verified every tensor of the anchor's lens ingredients but could not rebuild its
vocabulary map, because transformers would not import in the analysis environment. This check needs
only `tokenizers` and `huggingface_hub`. It rebuilds the map exactly as the replication study's
exporter defines it (convert_ids_to_tokens, then U+2581 -> space and <0xHH> -> that byte decoded as
UTF-8 with replacement) from google/gemma-2-27b's tokenizer.json at revision 938270f5..., and compares
all 256,000 entries with the `vocab` field of data/vectors/lens_ingredients.pt.

Run from the project root:  .venv/bin/python gemma_grid/scripts/check_anchor_vocab_tokenizer.py
Writes gemma_grid/geometry_out/anchor_vocab_tokenizer_check.json.
"""
import hashlib, json, os, re, sys, time
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STORED = os.path.join(REPO, "data/vectors/lens_ingredients.pt")
OUT = os.path.join(REPO, "gemma_grid/geometry_out/anchor_vocab_tokenizer_check.json")
MODEL, REV, N = "google/gemma-2-27b", "938270f5272feb02779b55c2bb2fffdd0f53ff0c", 256000
BYTE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def surface(t):
    m = BYTE.match(t)
    if m:
        return bytes([int(m.group(1), 16)]).decode("utf-8", errors="replace")
    return t.replace("▁", " ")


def main():
    path = hf_hub_download(MODEL, "tokenizer.json", revision=REV)
    tok_sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    tk = Tokenizer.from_file(path)
    rebuilt = []
    for i in range(N):
        t = tk.id_to_token(i)
        if t is None:
            sys.exit(f"STOP: tokenizer has no token for id {i}")
        rebuilt.append(surface(t))
    stored = torch.load(STORED, map_location="cpu", weights_only=False, mmap=True).get("vocab")
    if stored is None:
        sys.exit("STOP: stored ingredients have no vocab field")
    stored_type = type(stored).__name__
    if isinstance(stored, dict):
        keys = list(stored)
        if all(isinstance(k, int) for k in keys[:100]):          # {id: token}
            stored = [stored[i] for i in range(N)] if len(stored) == N else sys.exit("STOP: id map size")
        else:                                                  # {token: id}
            inv = {v: k for k, v in stored.items()}
            if len(inv) != N: sys.exit(f"STOP: token map has {len(inv)} distinct ids")
            stored = [inv[i] for i in range(N)]
    else:
        stored = list(stored)
    if len(stored) != N:
        sys.exit(f"STOP: stored vocab has {len(stored)} entries, expected {N}")
    diffs = [i for i in range(N) if rebuilt[i] != stored[i]]
    res = {"record": "Anchor (27B base) vocabulary map rebuilt from tokenizer.json at the pinned revision "
                     "and compared with the stored lens ingredients' vocab field.",
           "model": MODEL, "revision": REV, "tokenizer_json_sha256": tok_sha,
           "stored_file": os.path.relpath(STORED, REPO), "stored_vocab_type": stored_type, "n_entries": N, "identical": not diffs,
           "n_diffs": len(diffs), "first_diffs": [{"id": i, "rebuilt": rebuilt[i], "stored": stored[i]}
                                                   for i in diffs[:20]],
           "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"identical: {not diffs}  ({len(diffs)} of {N} differ)  wrote {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
