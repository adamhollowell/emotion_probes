#!/usr/bin/env python3
"""
Embedding-table valence and arousal correlations, raw against cleaned, per cell (paper section 5.1,
Table E.1).

Producer of record for geometry_out/c3_raw_cleaned_embedding.json, re-derived on
2026-10-14 from the method the artifact serializes (first computed 2026-10-06, coordinator memo
COORDINATOR_TO_DRAFTING_EMBEDDING_RAW_VS_CLEANED_100T.md, no script retained). The verify mode
checks every stored value.

Per cell, from geometry_out/<cell>/c3_vectors.pt: `vectors_raw` is the embedding-table
diff-of-means vector per emotion, and `vectors` is the same after the neutral-embedding cleanup basis
is projected out (the 100T basis). For each set: PCA (center, SVD, scores = U * S), PC1 and PC2
restricted to the 45 emotions with Russell and Mehrabian (1977) ratings, each oriented so its
correlation with the rating is non-negative, Pearson r. This is phase4_c3_embedding.signals applied to
each vector set. The raw values do not depend on any cleanup basis.

The .pt files are read with a numpy-only reader of the torch zip format, so no torch is needed.

Usage:
  python c3_raw_cleaned_embedding.py            # recompute and verify against the stored artifact
  python c3_raw_cleaned_embedding.py --write    # recompute and overwrite the stored artifact
"""
import argparse
import hashlib
import json
import pickle
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import csv

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

OUT = REPO / "gemma_grid" / "geometry_out"
ART = OUT / "c3_raw_cleaned_embedding.json"
RATINGS = REPO / "data" / "ratings" / "russell_mehrabian_1977.csv"
CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base", "gemma-2-9b-it", "gemma-2-27b-base", "gemma-2-27b-it"]

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "BFloat16Storage": None, "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}


def load_pt_numpy(path: Path) -> dict:
    """Read a torch.save zip archive without torch. Tensors come back as numpy arrays."""
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl = next(n for n in names if n.endswith("/data.pkl") or n == "data.pkl")
    prefix = pkl[: -len("data.pkl")]

    class Storage:
        def __init__(self, key, dtype_name, numel):
            self.key, self.dtype_name, self.numel = key, dtype_name, numel

    def rebuild(storage, offset, size, stride, *_):
        raw = zf.read(f"{prefix}data/{storage.key}")
        dt = _DTYPES[storage.dtype_name]
        if dt is None:  # bfloat16: widen to float32 exactly
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            flat = u16.view(np.float32)
        else:
            flat = np.frombuffer(raw, dtype=dt)
        itemsize = flat.itemsize
        strides = tuple(s * itemsize for s in stride)
        return np.lib.stride_tricks.as_strided(flat[offset:], shape=tuple(size), strides=strides).copy()

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name.endswith("Storage"):
                return name
            if module == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            return super().find_class(module, name)

        def persistent_load(self, pid):
            typename, storage_type, key, location, numel = pid[:5]
            return Storage(key, storage_type, numel)

    return U(zf.open(pkl)).load()


def pca(Vl, k=5):
    """noise_floor.pca, verbatim (copied so this script needs no torch)."""
    Vc = Vl - Vl.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Vc, full_matrices=False)
    return U[:, :k] * S[:k], (S[:k] ** 2) / (S ** 2).sum()


def load_ratings(path, em):
    """noise_floor.load_ratings, verbatim."""
    reader = csv.DictReader(open(path)); cols = reader.fieldnames or []
    def find(*subs):
        for c in cols:
            if all(s in c.lower() for s in subs): return c
        return None
    term_c = find("term") or find("emotion") or cols[0]
    pl_c = find("pleasure", "mean") or find("pleasure") or find("valence")
    ar_c = find("arousal", "mean") or find("arousal")
    rr = {}
    for row in reader:
        rr[row[term_c].strip().lower()] = (float(row[pl_c]), float(row[ar_c]))
    overlap = [e for e in em if e.lower() in rr]
    idx = [em.index(e) for e in overlap]
    pl = np.array([rr[e.lower()][0] for e in overlap])
    ar = np.array([rr[e.lower()][1] for e in overlap])
    return idx, pl, ar


def oriented_r(V: np.ndarray, idx, pl, ar):
    coords, _ = pca(V.astype(np.float64), 5)
    c1, c2 = coords[idx, 0].copy(), coords[idx, 1].copy()
    if np.corrcoef(c1, pl)[0, 1] < 0:
        c1 = -c1
    if np.corrcoef(c2, ar)[0, 1] < 0:
        c2 = -c2
    return float(np.corrcoef(c1, pl)[0, 1]), float(np.corrcoef(c2, ar)[0, 1])


def compute() -> dict:
    cells, shas = {}, {}
    n_overlap = None
    for c in CELLS:
        p = OUT / c / "c3_vectors.pt"
        shas[c] = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        d = load_pt_numpy(p)
        em = list(d["emotions"])
        idx, pl, ar = load_ratings(RATINGS, em)
        idx = np.asarray(idx)
        n_overlap = len(idx)
        raw = np.asarray(d["vectors_raw"])[:, 0, :]
        cln = np.asarray(d["vectors"])[:, 0, :]
        vr, ar_r = oriented_r(raw, idx, pl, ar)
        vc, ar_c = oriented_r(cln, idx, pl, ar)
        cells[c] = {"valence_r_raw": vr, "arousal_r_raw": ar_r, "valence_r_cleaned": vc, "arousal_r_cleaned": ar_c}
    return {"cells": cells, "n_overlap": n_overlap, "source_c3_vectors_sha256_12": shas}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    stored = json.loads(ART.read_text())
    new = compute()
    bad = [c for c in CELLS if stored["source_c3_vectors_sha256_12"][c] != new["source_c3_vectors_sha256_12"][c]]
    if bad:
        sys.exit(f"c3_vectors.pt hash mismatch against the artifact: {bad}")
    if a.write:
        out = dict(stored)
        out["cells"] = new["cells"]
        out["n_overlap"] = new["n_overlap"]
        out["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ART.write_text(json.dumps(out, indent=2))
        print("wrote", ART)
        sys.exit(0)
    ok, worst = True, 0.0
    for c in CELLS:
        for k, v in stored["cells"][c].items():
            n = new["cells"][c][k]
            if n != v:
                ok = False
                worst = max(worst, abs(n - v))
                print(f"MISMATCH {c}.{k}: stored {v!r} recomputed {n!r}")
    print("numpy", np.__version__)
    print("VERIFIED: every stored value reproduced exactly" if ok else f"VERIFICATION FAILED (largest abs diff {worst:.3e})")
    sys.exit(0 if ok else 1)
