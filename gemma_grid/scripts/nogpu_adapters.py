#!/usr/bin/env python3
"""Rebuild the input adapters that three producers read when they ran on 2026-08-27.

Written for the release, after the run. The adapters were built in a session's scratch folder
that has since been cleared, so this script rebuilds them from released files by the same steps
as the run of record (the steps survive in that session's record):

  pmap_transport  each model's cleaned vectors at its analysis layer (2B 17, 9B 28, 27B 31),
                  sliced from canonical.pt (emotion_vectors_171.pt for base 27B) and saved as a
                  171 x d file, plus cells_adapters.json listing the six files. The files keep
                  their as-run names, <model>__canonical_100T_L<layer>_analysis2d.pt, because the
                  frozen loader in pmap_transport.py checks for "100T" in the path.
  g66             g66_fair_adapter.json: the fair_140 list of fair_emotions_v1.json under the key
                  fair_emotions. ownword_ranks.json: per_emotion_ownword_ranks.json without its
                  _source_note key (the run read the file before that note was added, and the
                  frozen loader reads every top-level key as a model).
  g54a            g54a_l0scalar.json: the layer 0 rank of each emotion from the same ranks file.
                  purity.json: each emotion's 10-nearest-neighbour purity under a k = 10 k-means
                  partition at the analysis layer, computed as in the run of record and checked
                  against the Stage A cohesion summaries.

The per-emotion own-word ranks need the models' unembedding matrices, so they ship as a file
(geometry_out/per_emotion_ownword_ranks.json) rather than being recomputed here.

Run from the repository root, in the analysis environment:
    python gemma_grid/scripts/nogpu_adapters.py --out-dir <scratch folder>
then the commands in README.md, "Producers written after the run".
"""
import argparse, json
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
GG = REPO / "gemma_grid"
GEO = GG / "geometry_out"
CELLS = ["gemma-2-2b-base", "gemma-2-2b-it", "gemma-2-9b-base", "gemma-2-9b-it", "gemma-2-27b-base", "gemma-2-27b-it"]
LAYER = {"gemma-2-2b-base": 17, "gemma-2-2b-it": 17, "gemma-2-9b-base": 28, "gemma-2-9b-it": 28,
         "gemma-2-27b-base": 31, "gemma-2-27b-it": 31}
VEC = {c: f"gemma_grid/data/{c}/vectors/canonical.pt" for c in CELLS}
VEC["gemma-2-27b-base"] = "data/vectors/emotion_vectors_171.pt"


def pmap_adapters(out):
    adir = out / "adapters_2d"; adir.mkdir(parents=True, exist_ok=True)
    cells = {}
    for c in CELLS:
        p = torch.load(REPO / VEC[c], map_location="cpu", weights_only=False)
        La = LAYER[c]; V2 = p["vectors"][:, La, :].contiguous()
        outp = adir / f"{c}__canonical_100T_L{La}_analysis2d.pt"
        torch.save({"vectors": V2, "emotions": list(p["emotions"]), "layer": La,
                    "source": f"analysis-layer slice of {VEC[c]} (rebuilt for the release by nogpu_adapters.py)"}, outp)
        cells[c] = str(outp)
        print(f"pmap adapter {c}: {tuple(V2.shape)}")
    (out / "cells_adapters.json").write_text(json.dumps(cells, indent=1))


def ranks_adapters(out):
    r = json.loads((GEO / "per_emotion_ownword_ranks.json").read_text())
    r = {c: v for c, v in r.items() if not c.startswith("_")}
    assert list(r) == CELLS, list(r)
    (out / "ownword_ranks.json").write_text(json.dumps(r, indent=1))
    f = json.loads((GEO / "fair_emotions_v1.json").read_text())
    json.dump({"fair_emotions": f["fair_140"], "_adapter": "key-rename fair_140->fair_emotions for frozen g66 loader"},
              open(out / "g66_fair_adapter.json", "w"))
    l0 = {c: {e: v["L0"] for e, v in r[c].items()} for c in r}
    json.dump(l0, open(out / "g54a_l0scalar.json", "w"))
    print(f"g66 and g54a adapters: fair_140 {len(f['fair_140'])}, ranks for {len(r)} models")


def purity(out):
    from sklearn.cluster import KMeans
    res = {}
    for c in CELLS:
        p = torch.load(REPO / VEC[c], map_location="cpu", weights_only=False)
        em = list(p["emotions"]); Vl = p["vectors"].float().numpy()[:, LAYER[c], :]; n = len(em)
        nrm = np.clip(np.linalg.norm(Vl, axis=1, keepdims=True), 1e-12, None); Vn = Vl / nrm; C = Vn @ Vn.T
        lab = KMeans(n_clusters=10, n_init=10, random_state=0).fit(Vl).labels_
        np.fill_diagonal(C, -np.inf); order = np.argsort(-C, axis=1)
        res[c] = {em[i]: float((lab[order[i, :10]] == lab[i]).mean()) for i in range(n)}
        sa = json.loads((GEO / c / "stageA_summary.json").read_text())
        mine = float(np.mean(list(res[c].values()))); ref = sa["cohesion"]["mean_purity_top10"]
        print(f"purity {c}: mean {mine:.5f} vs Stage A {ref:.5f} {'OK' if abs(mine - ref) < 1e-4 else 'DIFF'}")
    (out / "purity.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pmap_adapters(out); ranks_adapters(out); purity(out)
    print(f"adapters in {out}")


if __name__ == "__main__":
    main()
