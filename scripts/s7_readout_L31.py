#!/usr/bin/env python3
"""§7.1/§7.2 direct-unembedding readout at L31, streaming W_U from the zip.

Producer of record for geometry_out/s7_readout_L31.json (per-emotion promoted/
suppressed tokens + own-word ranks). Reads the canonical emotion vectors and the
logit-lens ingredients.

NOTE — lens_ingredients.pt is NOT redistributed (it is, to within rounding,
Gemma-2-27B's own unembedding matrix W_U). Regenerate it from the base model with
the recipe in the HF model repo before running this script:
    python export_lens_ingredients.py --out data/vectors/lens_ingredients.pt

Run from the repo root:
    python scripts/s7_readout_L31.py
"""
import sys, os, json, zipfile
import numpy as np
sys.path.insert(0, "outputs")  # torch-free load_pt ships here; falls back to torch
try:
    from load_pt import load_pt
except Exception:
    import torch
    def load_pt(p): return torch.load(p, map_location="cpu", weights_only=False)


def _stamp(vec_path, basis="canonical"):
    import hashlib, time
    sha = None
    if os.path.exists(vec_path):
        h = hashlib.sha256()
        with open(vec_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
    return {"basis": basis, "vectors": vec_path, "vectors_sha256": sha,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


VEC = "data/vectors/emotion_vectors_171.pt"
TABLE4 = ["happy","inspired","loving","proud","calm","desperate","angry","guilty","sad","afraid","nervous","surprised"]
PT = "data/vectors/lens_ingredients.pt"   # recipe-only; regenerate via export_lens_ingredients.py (see header)
zf = zipfile.ZipFile(PT)
info = {i.filename: i for i in zf.infolist()}
w = info["lens_ingredients/data/0"]
# data offset = header_offset + fixed 30 + fname + extra (read local header for extra len)
import struct
f = open(PT, "rb"); f.seek(w.header_offset)
hdr = f.read(30); fn_len, ex_len = struct.unpack("<HH", hdr[26:30])
off = w.header_offset + 30 + fn_len + ex_len
V_TOK, D = 256000, 4608
mm = np.memmap(PT, dtype=np.float32, mode="r", offset=off, shape=(V_TOK, D))  # token-major

# ln_final weight
import io
lnw = np.frombuffer(zf.read("lens_ingredients/data/1"), dtype=np.float32).copy()

# vocab
import pickle
from collections import OrderedDict
class _Lazy(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return lambda *a, **k: None
        if name.endswith("Storage"): return name
        if module == "collections" and name == "OrderedDict": return OrderedDict
        class D:
            def __getattr__(self, n): return n
        return getattr(D(), name) if module.startswith("torch") else super().find_class(module, name)
    def persistent_load(self, pid): return None
with zf.open("lens_ingredients/data.pkl") as fh:
    meta = _Lazy(fh).load()
vocab = meta["vocab"]

p = load_pt(VEC)
em = [str(e) for e in p["emotions"]]
V = np.asarray(p["vectors"], dtype=np.float32)[:, 31, :]           # (171, 4608) cleaned L31
# normalized-lens variant of the 12 (Gemma RMSNorm applied to the vector)
def rms(x): return x / np.sqrt((x.astype(np.float64)**2).mean(-1, keepdims=True) + 1e-6)
Vn = (rms(V) * (1.0 + lnw)).astype(np.float32)

scores  = np.zeros((171, V_TOK), dtype=np.float32)
scoresN = np.zeros((len(TABLE4), V_TOK), dtype=np.float32)
i12 = [em.index(e) for e in TABLE4]
CH = 8192
for s in range(0, V_TOK, CH):
    blk = np.asarray(mm[s:s+CH])           # (ch, 4608)
    scores[:, s:s+CH]  = V @ blk.T
    scoresN[:, s:s+CH] = Vn[i12] @ blk.T
out = {"tag": "canonical_L31", "direct": {}, "normalized": {}}
for row, e in enumerate(TABLE4):
    sc = scores[em.index(e)]
    o = np.argsort(-sc)
    out["direct"][e] = {"promoted": [vocab[int(t)] for t in o[:5]], "suppressed": [vocab[int(t)] for t in o[-5:][::-1]]}
    scn = scoresN[row]; on = np.argsort(-scn)
    out["normalized"][e] = {"promoted": [vocab[int(t)] for t in on[:5]], "suppressed": [vocab[int(t)] for t in on[-5:][::-1]]}
# own-word ranks at L31 (single-token, leading-space convention)
tok_of = {}
inv = {}
for tid, tstr in vocab.items(): inv.setdefault(tstr, tid)
ranks = {}
for i, e in enumerate(em):
    tid = inv.get(" " + e, inv.get(e))
    if tid is None: continue
    r_ = int((scores[i] > scores[i, tid]).sum()) + 1
    ranks[e] = r_
out["own_word_rank_L31"] = {"n_scored": len(ranks), "median": float(np.median(list(ranks.values())))}
out["_provenance"] = _stamp(VEC, "canonical")
json.dump(out, open("geometry_out/s7_readout_L31.json","w"), indent=1, ensure_ascii=False)
for e in TABLE4:
    print(f"{e:11s} ^ {out['direct'][e]['promoted']}  v {out['direct'][e]['suppressed']}")
print("own-word ranks:", out["own_word_rank_L31"])
