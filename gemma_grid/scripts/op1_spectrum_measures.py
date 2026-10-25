#!/usr/bin/env python3
"""
op1_spectrum_measures.py — OP.1 cheap part (Adam's ruling 10, 2026-08-10).

Per cell, on the 171 emotion vectors at the analysis layer (canonical, cleaned
primary + raw sensitivity): (a) participation ratio PR = (Σλ)²/Σλ² over the
PCA eigenvalue spectrum (Gao et al. 2017 lineage) — the effective number of
dimensions carrying emotional variance; (b) TwoNN intrinsic dimension
(Facco et al. 2017; Valeriani et al. 2023 applied it to transformers) — MLE
form, ID = N / Σ log(r2/r1). Question served: G3.1 says PC1's share falls
with scale — do PR/ID rise correspondingly (freed variance = more effective
dimensions), and does the cleaned/raw contrast hold?

Gates: var_pc1 recomputed from the spectrum must match the stageA table
(≤5e-4 after rounding); N=171 per cell. SR.5 manifest beside the output.
Outputs: geometry_out/op1_spectrum_measures_v1.json (+ manifest).
STATUS: UNREGISTERED exploratory; Phase-5 check owed before paper contact.
"""
import csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

GG = Path(__file__).resolve().parent.parent
REPO = GG.parent
sys.path.insert(0, str(REPO))
import ptload  # torch-free .pt loader at repo root

CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base",
         "gemma-2-9b-it", "gemma-2-27b-base", "gemma-2-27b-it"]

def canon_path(c):
    if c == "gemma-2-27b-base":
        return REPO / "data" / "vectors" / "emotion_vectors_171.pt"
    return GG / "data" / c / "vectors" / "canonical.pt"

STAGEA = GG / "geometry_out" / "stageA_cross_cell_table.csv"
OUT = GG / "geometry_out" / "op1_spectrum_measures_v1.json"
MANIFEST = GG / "geometry_out" / "op1_spectrum_measures_v1_manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def get_vectors(obj):
    """Return dict variant->(emotions, (171,d) array) from a canonical .pt object.
    Handles both schemas: {emotions: list, vectors: array, vectors_raw: array}
    (grid cells) and {emotion: vector} dicts (anchor-era)."""
    out = {}
    if not isinstance(obj, dict):
        return out
    emos = obj.get("emotions")
    for key, tag in (("vectors", "cleaned"), ("vectors_clean", "cleaned"), ("vectors_raw", "raw")):
        v = obj.get(key)
        if v is None:
            continue
        if isinstance(v, dict) and len(v) == 171:
            es = sorted(v)
            out[tag] = (es, np.stack([np.asarray(v[e], dtype=np.float64) for e in es]))
        elif hasattr(v, "shape") and len(getattr(v, "shape", ())) in (2, 3) and v.shape[0] == 171:
            assert emos is not None and len(emos) == 171
            out[tag] = (list(emos), np.asarray(v, dtype=np.float64))  # (171,d) or (171,L,d); sliced by caller
    if not out:  # anchor-era: top-level {emotion: vector} or nested under a variant key
        cand = {k: v for k, v in obj.items() if hasattr(v, "shape")}
        if len(cand) == 171:
            es = sorted(cand)
            out["cleaned"] = (es, np.stack([np.asarray(cand[e], dtype=np.float64) for e in es]))
        else:
            for k, v in obj.items():
                if isinstance(v, dict) and len(v) == 171 and all(hasattr(x, "shape") for x in v.values()):
                    es = sorted(v)
                    out["cleaned" if "raw" not in k else "raw"] = (es, np.stack([np.asarray(v[e], dtype=np.float64) for e in es]))
    return out


def spectrum_stats(X):
    Xc = X - X.mean(0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    lam = lam[lam > 1e-12]
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    var_pc1 = float(100 * lam[0] / lam.sum())
    # TwoNN (MLE): mu_i = r2/r1 on unit-normalized rows (cosine-consistent)
    U = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    D = np.linalg.norm(U[:, None, :] - U[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    r = np.sort(D, axis=1)[:, :2]
    mu = r[:, 1] / r[:, 0]
    twonn = float(len(mu) / np.log(mu).sum())
    return pr, twonn, var_pc1


def main():
    stagea = {row["cell"]: float(row["var_pc1"]) for row in csv.DictReader(open(STAGEA))} \
        if STAGEA.exists() else {}
    rows, inputs = {}, {}
    for c in CELLS:
        pth = canon_path(c)
        inputs[str(pth.relative_to(REPO))] = sha256(pth)
        variants = get_vectors(ptload.load(str(pth)))
        assert "cleaned" in variants, (c, "no cleaned vectors found")
        ANALYSIS_LAYER = {"2b": 17, "9b": 28, "27b": 31}[c.split("-")[2]]
        cell = {"analysis_layer": ANALYSIS_LAYER}
        for tag, (emos, X) in variants.items():
            if X.ndim == 3:  # (171, n_layers, d) all-layer store -> analysis-layer slice
                X = X[:, ANALYSIS_LAYER, :]
            pr, twonn, v1 = spectrum_stats(X)
            cell[tag] = {"participation_ratio": round(pr, 3),
                         "twonn_id": round(twonn, 3), "var_pc1_check": round(v1, 3)}
        # gate vs stageA (cleaned)
        if stagea:
            expect = stagea.get(c)
            if expect is not None:
                got = cell["cleaned"]["var_pc1_check"]
                assert abs(got - expect) < 5e-2, (c, got, expect)
                cell["gate_var_pc1_vs_stageA"] = "PASS"
        rows[c] = cell
    out = {
        "provenance": {
            "script": "gemma_grid/scripts/op1_spectrum_measures.py",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ruling": "Adam 2026-08-10 (ruling 10): bolt the cheap OP.1 part on — PR + TwoNN ID per cell.",
            "definitions": "PR=(sum λ)^2/sum λ^2 over PCA eigenvalues of the 171 vectors (mean-centered); TwoNN MLE ID on unit-normalized vectors.",
            "status": "UNREGISTERED exploratory; Phase-5 check owed before paper contact.",
        },
        "cells": rows,
    }
    OUT.write_text(json.dumps(out, indent=1))
    if STAGEA.exists():
        inputs[str(STAGEA.relative_to(REPO))] = sha256(STAGEA)
    MANIFEST.write_text(json.dumps({
        "m9_note": "SR.5 input manifest.", "created_utc": out["provenance"]["created_utc"],
        "inputs": inputs, "output": {str(OUT.relative_to(REPO)): sha256(OUT)}}, indent=1))
    for c, r in rows.items():
        cl = r["cleaned"]
        print(f"{c}: PR={cl['participation_ratio']} ID={cl['twonn_id']} var_pc1={cl['var_pc1_check']} gate={r.get('gate_var_pc1_vs_stageA','n/a')}")
    print("wrote", OUT.name, "+ manifest")


if __name__ == "__main__":
    main()
