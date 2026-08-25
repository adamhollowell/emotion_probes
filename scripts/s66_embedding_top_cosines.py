#!/usr/bin/env python3
"""§6.6 — embedding-level (static input embedding, no forward pass) pairwise-cosine
ranking among the 171 emotion vectors: which pairs are most similar *before* any
transformer layer runs.

Producer of record for geometry_out/s66_embedding_top_cosines.json. Single-basis (canonical);
torch-free.

Input: embedding_null_vectors.pt (the bag-of-embeddings null; §6.6). Not committed as a
file — regenerate from the base model with `extract_embedding_null.py` (see HF model repo).

Run from the repo root:
    python scripts/s66_embedding_top_cosines.py
"""
import sys, os, json, hashlib, datetime
sys.path.insert(0, "outputs")  # torch-free load_pt ships here; falls back to torch
try:
    from load_pt import load_pt
except Exception:
    import torch
    def load_pt(p): return torch.load(p, map_location="cpu", weights_only=False)
import numpy as np

EMB = "data/vectors/embedding_null_vectors.pt"
OUT = "geometry_out/s66_embedding_top_cosines.json"
L31_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

blob = load_pt(EMB)
em = list(blob["emotions"])
n = len(em)

def ranked(matrix_key):
    V = np.asarray(blob[matrix_key], dtype=np.float64).reshape(n, -1)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    C = Vn @ Vn.T
    iu = np.triu_indices(n, k=1)
    pairs = sorted(
        ([em[i], em[j], round(float(C[i, j]), 4)] for i, j in zip(*iu)),
        key=lambda r: r[2], reverse=True,
    )
    Coff = C.copy(); np.fill_diagonal(Coff, -np.inf)
    nn = {em[i]: [em[int(np.argmax(Coff[i]))], round(float(np.max(Coff[i])), 4)] for i in range(n)}
    return pairs, nn

pairs_clean, nn_clean = ranked("vectors")
pairs_raw, _ = ranked("vectors_raw")

legacy = {}
lookup = {(a, b): c for a, b, c in pairs_clean}
for a, b in [("afraid", "scared"), ("joyful", "happy"), ("compassionate", "empathetic")]:
    legacy[f"{a}->{b}"] = lookup.get((a, b), lookup.get((b, a)))

out = {
    "tag": "canonical",
    "desc": "Embedding-level (static input embedding, no forward pass) pairwise cosine "
            "ranking among the 171 emotion vectors, cleaned. Answers 'which pairs are most "
            "similar before any transformer layer runs.' Companion to s66_embedding.json.",
    "n_emotions": n,
    "cleaned": {
        "max_pair": pairs_clean[0],
        "top50_pairs": pairs_clean[:50],
        "nearest_neighbor_per_emotion": nn_clean,
    },
    "raw_reference": {"max_pair": pairs_raw[0], "top10_pairs": pairs_raw[:10]},
    "legacy_example_pairs_cleaned": legacy,
    "legacy_example_note": "afraid->scared (0.960) is the essay's carried-over example; it is "
                           "NOT the embedding-level maximum. Top pair is "
                           f"{pairs_clean[0][0]}/{pairs_clean[0][1]} at {pairs_clean[0][2]}; "
                           "afraid/scared ranks 4th (ties enraged/furious).",
    "_provenance": {
        "basis": "canonical",
        "embedding_vectors": EMB,
        "embedding_vectors_sha256": sha256(EMB),
        "L31_canonical_basis_sha256": L31_SHA,
        "matrix": "blob['vectors'] (cleaned); raw = blob['vectors_raw']",
        "method": "row-normalize, C = Vn @ Vn.T, rank upper triangle",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "producer": "scripts/s66_embedding_top_cosines.py",
        "reproduces_s66_neighbor_cosines": "afraid->scared, joyful->happy, compassionate->empathetic to the digit",
    },
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", OUT, "| max pair:", out["cleaned"]["max_pair"])
