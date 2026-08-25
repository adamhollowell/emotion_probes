#!/usr/bin/env python3
"""
Static producer for Figure 3 (the UMAP cluster scatter).

Default (final) path — deterministic, no UMAP, no adjustText:
    python3 repo/github_repo/scripts/render_parallel_fig6_umap.py

  Reads the frozen coords (geometry_out/fig3_coords_AUTH.csv) + the hand-placed
  label layout (geometry_out/fig3_label_layout_AUTH.json, produced by
  render_parallel_fig6_interactive.py) and renders the canonical PNG via the SAME
  fig6_common.render_final code path the editor's preview uses. So the PNG shown in the editor and the PNG this script produces are the same file. The
  layout's stamped coords_sha256 must equal the current coords sha or this exits.
  Output: docs/e1_figures/E1_Figure_3_umap_kmeans.png  (override --out PATH).

  Run from the PROJECT ROOT so the relative paths resolve to the verified main copy
  of the coords (NOT the stale staged copy under repo/github_repo/).

Coordinate-generation path — RUN ONCE, ALREADY RUN (2026-08-17):
    python3 repo/github_repo/scripts/render_parallel_fig6_umap.py --recompute

  Builds geometry_out/fig3_coords_AUTH.csv from the canonical vectors and hard-exits
  if that file already exists. UMAP is non-deterministic, so a second draw would
  produce valid-but-different geometry and silently void every label layout stamped
  to the first. The coordinates are frozen; do not regenerate them.
"""
import sys

import fig6_common as fc

OUT_PNG = "docs/e1_figures/E1_Figure_3_umap_kmeans.png"


def final_render(out_png: str) -> None:
    df = fc.load_coords()
    fc.check_membership(df)
    layout = fc.load_layout()
    if layout is None:
        sys.exit(
            "No label layout found at %s.\n"
            "Place the labels first:\n"
            "  python3 repo/github_repo/scripts/render_parallel_fig6_interactive.py        (drag, press S)\n"
            "  python3 repo/github_repo/scripts/render_parallel_fig6_interactive.py --export (headless seed)\n"
            % fc.LAYOUT_JSON)
    if layout.get("coords_sha256") and layout["coords_sha256"] != fc.sha256_file(fc.COORDS_CSV):
        sys.exit("FATAL: layout was placed against a different coords file "
                 "(coords_sha256 mismatch). Re-run the editor against the current coords.")
    import os
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fc.render_final(df, layout, out_png)
    print("[wrote] %s" % out_png)
    print("        sha256 %s" % fc.sha256_file(out_png))
    print("        labels %d, from %s" % (len(layout["labels"]), fc.LAYOUT_JSON))


def recompute_coords() -> None:
    """Original UMAP-generation path. Rewrites geometry_out/fig6_cleaned_coords.csv.
    Heavy deps imported here only, so the final path never needs umap-learn."""
    import os
    import numpy as np
    import pandas as pd
    import torch
    import umap
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    VECTORS = "data/vectors/emotion_vectors_171.pt"
    PARTITION_CSV = fc.PARTITION_CSV
    LAYER = 31
    OUT_CSV = fc.COORDS_CSV

    # COMPUTE ONCE. UMAP is non-deterministic across environments, and a second
    # coordinate file is how a layout silently ends up placed against geometry that
    # no longer matches the render. If the authoritative coords already exist, stop.
    if os.path.exists(OUT_CSV):
        sys.exit("REFUSING to recompute: %s already exists (sha256 %s).\n"
                 "The coordinates are frozen; every render reads this file. If they "
                 "genuinely must be rebuilt, move the existing file aside deliberately "
                 "— and know that every label layout stamped to the old sha is void."
                 % (OUT_CSV, fc.sha256_file(OUT_CSV)))

    # Gate on bytes, not filename: the canonical basis is sha256 6de2e293…; a stray working
    # copy of the canonically-named vectors file can still be the prior basis, which is
    # exactly how a wrong render happens. Hard-exit unless the loaded file is the canonical basis.
    EXPECTED_VEC_SHA = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
    got_sha = fc.sha256_file(VECTORS)
    if got_sha != EXPECTED_VEC_SHA:
        sys.exit("FATAL: %s sha256 %s != expected canonical basis %s (wrong basis or stray working copy)"
                 % (VECTORS, got_sha, EXPECTED_VEC_SHA))
    print("[basis] vectors sha256 %s == canonical basis (verified)" % got_sha[:12])

    p = torch.load(VECTORS, map_location="cpu", weights_only=False)
    emo = list(p["emotions"])
    V = p["vectors"][:, LAYER, :].float().numpy()      # 'vectors' = CLEANED
    idx = {e: i for i, e in enumerate(emo)}

    part = fc.read_partition_map(PARTITION_CSV)   # native seed-0 membership (s4_clusters.json)
    # The stored partition IS the certified fresh seed-0 KMeans on the canonical vectors, so the strong
    # self-check holds again: a fresh seed-0 run reproduces it (ARI 1.0). (Restored 2026-08-17; the
    # interim "covers-171 only" weakening was built on the reused-prior-basis premise, now void.)
    assert set(part) == set(emo), "partition must cover all 171 emotions"
    assert sorted(set(part.values())) == list(range(10)), "partition must be a valid k=10 labeling"
    fresh = KMeans(n_clusters=10, random_state=0, n_init=10).fit(V).labels_
    ari = adjusted_rand_score([part[e] for e in emo], fresh)
    print("[integrity] stored partition vs fresh seed-0 KMeans on the canonical vectors: ARI = %.4f (expect 1.0000)" % ari)
    assert ari > 0.999, "stored partition does not match a fresh seed-0 run on the canonical vectors"

    Xc = V - V.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S
    sgn = 1.0 if (np.mean([scores[idx[e], 0] for e in ["optimistic", "kind", "cheerful"]])
                  > np.mean([scores[idx[e], 0] for e in ["sleepy", "afraid", "disgusted"]])) else -1.0
    pc1 = scores[:, 0] * sgn

    # cluster id -> family name, keyed by CLUSTER SIZE: the native sizes are all distinct, so
    # size uniquely identifies each family. Family names per Appendix B.
    SIZE_NAMES = {8: "Compassionate Warmth", 24: "Exuberant Joy", 16: "Hopeful Serenity",
                  15: "Contemptuous Vigilance", 14: "Guilt and Self-Reproach", 32: "Low Mood and Depletion",
                  12: "Stunned Astonishment", 17: "Hostile Anger", 4: "Weary Lethargy", 29: "Fear and Overwhelm"}
    from collections import Counter as _Counter
    sizes = _Counter(part.values())
    assert sorted(sizes.values()) == sorted(SIZE_NAMES), \
        "partition sizes %s != expected native sizes %s" % (sorted(sizes.values()), sorted(SIZE_NAMES))
    name_for = {cid: SIZE_NAMES[sz] for cid, sz in sizes.items()}
    assert len(name_for) == 10, "size keying produced %d names for 10 clusters" % len(name_for)

    cl_pc1 = {cid: np.mean([pc1[idx[e]] for e in emo if part[e] == cid]) for cid in set(part.values())}
    order = sorted(cl_pc1, key=lambda c: -cl_pc1[c])
    rank = {cid: i for i, cid in enumerate(order)}

    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                    random_state=42).fit_transform(V)
    df = pd.DataFrame([{
        "emotion": e, "x": float(emb[idx[e], 0]), "y": float(emb[idx[e], 1]),
        "cluster_idx": rank[part[e]], "cluster_name": name_for[part[e]],
    } for e in emo])
    df.to_csv(OUT_CSV, index=False)
    print("[wrote] %s" % OUT_CSV)
    print("[FROZEN] coords sha256 %s" % fc.sha256_file(OUT_CSV))

    # ---- faithfulness gate ------------------------------------------------
    # The 2-D layout has to reflect the high-dimensional geometry it came from. For
    # each emotion take its 6 nearest neighbours in the embedding and measure their
    # mean cosine similarity in the cleaned L31 vectors. Expect ~0.6-0.7 (a random
    # baseline on these vectors is ~0.0); near zero means the coordinates were built
    # from the wrong basis or are corrupt.
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    COS = Vn @ Vn.T
    P = np.stack([emb[idx[e]] for e in emo])
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn6 = np.argsort(D, axis=1)[:, :6]
    per_emotion = np.array([COS[i, nn6[i]].mean() for i in range(len(emo))])
    faith = float(per_emotion.mean())
    rng = np.random.default_rng(0)
    rand_base = float(np.mean([COS[i, rng.choice([j for j in range(len(emo)) if j != i], 6, replace=False)].mean()
                               for i in range(len(emo))]))
    print("[faithfulness] mean cosine of the 6 nearest 2-D neighbours = %.4f "
          "(random baseline %.4f; expect ~0.60-0.70)" % (faith, rand_base))
    if not (0.50 <= faith <= 0.80):
        sys.exit("FATAL: faithfulness %.4f outside the expected band — the coords do not "
                 "reflect the cleaned L31 geometry. Do NOT place labels against this file." % faith)

    print("Next: place labels with render_parallel_fig6_interactive.py, then re-run "
          "this script (no flag) for the deterministic final PNG.")


def main() -> None:
    out_png = OUT_PNG
    if "--out" in sys.argv:
        out_png = sys.argv[sys.argv.index("--out") + 1]
    if "--recompute" in sys.argv:
        recompute_coords()
    else:
        final_render(out_png)


if __name__ == "__main__":
    main()
