#!/usr/bin/env python3
"""Z3 — Macro-geometry checks (Zhang KDE-structure claims at 171-emotion resolution).

Pre-stated Ekman lexical counterparts: angry, disgusted, afraid, joyful, sad,
surprised. Per cell (cleaned vectors, analysis layer): counterpart cosine
matrix; angry↔disgusted cosine + percentile among all pairs; mutual NN ranks;
joyful↔surprised the same; k=10 co-membership for the three Zhang groups; signed
PC1–PC2 coordinates of the six.

Clustering conventions are the stage-A ones (k=10, seed 0, n_init=10, on the
cleaned vectors at the analysis layer). If sklearn is unavailable or the stage-A
script's conventions visibly differ, STOP and report (handoff rule 4).
Gate Z3-G1: emotions lists identical across all six cells.
"""
import json, sys
import numpy as np
from zhang_common import (REPO, OUT, CELLS, Manifest, load_vectors, pca_coords,
                          sign_orient)

EKMAN = ["angry", "disgusted", "afraid", "joyful", "sad", "surprised"]
GROUPS = {"negative": ("angry", "disgusted"),
          "positive": ("joyful", "surprised"),
          "downcast": ("afraid", "sad")}


def main():
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        sys.exit("STOP: sklearn unavailable — stage-A clustering conventions require it")
    man = Manifest()
    out = {"counterparts": EKMAN}
    em_ref = None
    for cell, cfg in CELLS.items():
        em, V = load_vectors(cell, man, "vectors")
        if em_ref is None:
            em_ref = em
        assert em == em_ref, f"STOP: emotions list differs at {cell} (Z3-G1 FAIL)"
        for e in EKMAN:
            assert e in em, f"STOP: counterpart '{e}' missing from emotion list"
        Vl = V[:, cfg["layer"], :]
        Vn = Vl / np.linalg.norm(Vl, axis=1, keepdims=True)
        S = Vn @ Vn.T
        idx = {e: em.index(e) for e in EKMAN}

        cos6 = {a: {b: round(float(S[idx[a], idx[b]]), 4) for b in EKMAN} for a in EKMAN}

        iu = np.triu_indices(len(em), 1)
        allpairs = S[iu]
        def pair_stats(a, b):
            c = float(S[idx[a], idx[b]])
            pct = float((allpairs < c).mean()) * 100
            rank_ab = int((-S[idx[a]]).argsort().tolist().index(idx[b]))  # 1 = nearest (0 = self)
            rank_ba = int((-S[idx[b]]).argsort().tolist().index(idx[a]))
            return {"cosine": round(c, 4), "percentile_all_pairs": round(pct, 1),
                    "nn_rank_a_to_b": rank_ab, "nn_rank_b_to_a": rank_ba}
        out_cell = {"cosine_matrix_6": cos6,
                    "angry_disgusted": pair_stats("angry", "disgusted"),
                    "joyful_surprised": pair_stats("joyful", "surprised"),
                    "afraid_sad": pair_stats("afraid", "sad")}

        km = KMeans(n_clusters=10, random_state=0, n_init=10).fit(Vl)
        lab = {e: int(km.labels_[idx[e]]) for e in EKMAN}
        out_cell["kmeans_labels"] = lab
        out_cell["group_comembership"] = {
            g: bool(lab[a] == lab[b]) for g, (a, b) in GROUPS.items()}

        coords = sign_orient(pca_coords(Vl), em)
        out_cell["pc12_coords"] = {e: [round(float(coords[idx[e], 0]), 3),
                                       round(float(coords[idx[e], 1]), 3)] for e in EKMAN}
        out[cell] = out_cell
        print(f"[{cell}] angry↔disgusted cos {out_cell['angry_disgusted']['cosine']} "
              f"(pctl {out_cell['angry_disgusted']['percentile_all_pairs']}); "
              f"co-membership {out_cell['group_comembership']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zhang_z3_results.json").write_text(json.dumps(out, indent=1))
    man.write("zhang_z3_manifest.json")
    print("Z3 done; results + manifest written.")


if __name__ == "__main__":
    main()
