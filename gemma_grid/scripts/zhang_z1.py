#!/usr/bin/env python3
"""Z1 — Cross-validated nearest-class-mean decodability at the analysis layer.

The dissociation experiment (Zhang memo §3.1): does 171-way decodability RISE with
scale while var_pc1 FALLS? Classifier = the study's own diff-of-means object
(class mean − grand mean), cosine rule, 5-fold CV, seed 0. No trained probe.

Gate Z1-G1: full-data per-emotion means − grand-of-means must reproduce canonical
vectors_raw at the analysis layer (rel ≤ 1e-4) per cell, BEFORE any CV runs.

Design is pre-stated in ZHANG_PARALLELS_CLAUDE_CODE_HANDOFF.md — do not vary.
"""
import csv, json, sys, time
import numpy as np
from zhang_common import (REPO, OUT, CELLS, Manifest, load_pt, as_np, load_vectors,
                          HAVE_TORCH)

SEED, FOLDS, TOPK = 0, 5, 5


def slice_path(cell):
    return REPO / f"gemma_grid/data/{cell}/vectors/{cell}__per_story_slice.pt"


def basis_path(cell):
    return REPO / f"gemma_grid/data/{cell}/vectors/{cell}__cleanup_basis.pt"


def get_layer_index(meta, cell):
    layers = meta.get("layers", meta.get("slice_layers"))
    if layers is None:
        if meta.get("analysis_layer") is not None:
            layers = [meta["analysis_layer"]]
        else:
            sys.exit(f"STOP: {cell} slice has no layer metadata — see handoff rule 4")
    layers = [int(x) for x in (layers.tolist() if hasattr(layers, "tolist") else layers)]
    target = CELLS[cell]["layer"]
    assert target in layers, f"STOP: analysis layer {target} not in slice layers {layers} for {cell}"
    return layers.index(target), layers


def load_cleanup_basis(cell, man, layer):
    """Schema (verified 2026-08-10): dict with 'bases_per_layer' (list of (k_l, d)
    float32 arrays, one per absolute layer), 'k_per_layer', 'analysis_layer'."""
    p = basis_path(cell)
    man.sha(p)
    d = load_pt(p)
    assert isinstance(d, dict) and "bases_per_layer" in d, \
        f"STOP: cleanup_basis_100T schema for {cell} not as expected — keys: " \
        f"{list(d.keys()) if isinstance(d, dict) else type(d).__name__} (handoff rule 4)"
    B = as_np(d["bases_per_layer"][layer])
    assert B.ndim == 2, f"STOP: basis at layer {layer} has shape {B.shape}"
    return B  # (k, d), orthonormal rows (project_out_top_pcs convention)


def project_out(X, B):
    return X - (X @ B.T) @ B


def run_cell(cell, man):
    t0 = time.time()
    sp = slice_path(cell)
    man.sha(sp)
    d = load_pt(sp, mmap=True)
    li, layers = get_layer_index(d, cell)
    per = d["per_emotion"]
    emotions = sorted(per.keys())
    em_canon, raw_canon = load_vectors(cell, man, "vectors_raw")
    assert sorted(em_canon) == emotions, f"STOP: emotion sets differ for {cell}"
    L = CELLS[cell]["layer"]

    # ---- pass 1: full-data means + gate --------------------------------------
    n_e = len(emotions)
    means = None
    for i, e in enumerate(emotions):
        X = as_np(per[e][:, li, :])                     # (1200, d)
        if means is None:
            means = np.zeros((n_e, X.shape[1]), dtype=np.float64)
        means[i] = X.mean(0)
    grand = means.mean(0)
    dirs_full = (means - grand).astype(np.float32)
    canon_at_L = raw_canon[[em_canon.index(e) for e in emotions], L, :]
    rel = float(np.linalg.norm(dirs_full - canon_at_L) / np.linalg.norm(canon_at_L))
    gate = {"rel_vs_vectors_raw": rel, "pass": rel <= 1e-4}
    print(f"  [{cell}] Z1-G1 rel={rel:.2e} -> {'PASS' if gate['pass'] else 'FAIL'}")
    if not gate["pass"]:
        return {"gate": gate, "aborted": True}

    # ---- CV folds (seed 0, per-emotion permutation, 5 x 240) -----------------
    rng = np.random.default_rng(SEED)
    perms = {e: rng.permutation(1200) for e in emotions}
    fold_of = {e: np.empty(1200, dtype=int) for e in emotions}
    for e in emotions:
        for f in range(FOLDS):
            fold_of[e][perms[e][f * 240:(f + 1) * 240]] = f

    # accumulate per-fold sums to build train means without a second full pass:
    # sum over stories per (emotion, fold), then train_mean = (total - fold_sum)/960
    dim = means.shape[1]
    fold_sums = np.zeros((n_e, FOLDS, dim), dtype=np.float64)
    for i, e in enumerate(emotions):
        X = as_np(per[e][:, li, :])
        for f in range(FOLDS):
            fold_sums[i, f] = X[fold_of[e] == f].sum(0)
    totals = fold_sums.sum(1)                                    # (n_e, d)

    B = load_cleanup_basis(cell, man, L)

    results = {}
    for variant in ("raw", "cleaned"):
        top1_f, top5_f = [], []
        per_emotion_correct = np.zeros(n_e)
        for f in range(FOLDS):
            train_means = (totals - fold_sums[:, f]) / (1200 - 240)
            g = train_means.mean(0)
            D = (train_means - g).astype(np.float32)
            if variant == "cleaned":
                D = project_out(D, B)
            Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
            c1 = c5 = n_test = 0
            for i, e in enumerate(emotions):
                X = as_np(per[e][:, li, :])[fold_of[e] == f] - g.astype(np.float32)
                if variant == "cleaned":
                    X = project_out(X, B)
                Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
                S = Xn @ Dn.T                                    # (240, n_e)
                pred = S.argmax(1)
                c1 += int((pred == i).sum())
                per_emotion_correct[i] += int((pred == i).sum())
                c5 += int((np.argsort(-S, axis=1)[:, :TOPK] == i).any(1).sum())
                n_test += X.shape[0]
            top1_f.append(c1 / n_test)
            top5_f.append(c5 / n_test)
        results[variant] = {
            "top1_mean": round(float(np.mean(top1_f)), 5),
            "top1_sd": round(float(np.std(top1_f)), 5),
            "top5_mean": round(float(np.mean(top5_f)), 5),
            "top5_sd": round(float(np.std(top5_f)), 5),
            "per_fold_top1": [round(x, 5) for x in top1_f],
        }
        if variant == "raw":
            results["per_emotion_top1_raw"] = {
                e: round(float(per_emotion_correct[i] / 1200), 4)
                for i, e in enumerate(emotions)}
    results["gate"] = gate
    results["layers_meta"] = layers
    results["runtime_s"] = round(time.time() - t0, 1)
    print(f"  [{cell}] raw top1 {results['raw']['top1_mean']:.4f}±{results['raw']['top1_sd']:.4f} "
          f"top5 {results['raw']['top5_mean']:.4f} | cleaned top1 {results['cleaned']['top1_mean']:.4f} "
          f"({results['runtime_s']}s)")
    return results


def main():
    man = Manifest()
    out = {"design": "171-way nearest-class-mean, cosine, 5-fold CV seed 0; "
                     "chance=1/171=0.00585; variants raw + HA-cleaned",
           "torch": HAVE_TORCH}
    # var_pc1 pairing from the stage-A table (the dissociation partner)
    gp = REPO / "gemma_grid/geometry_out/stageA_cross_cell_table.csv"
    man.sha(gp)
    out["var_pc1"] = {r["cell"]: float(r["var_pc1"]) for r in csv.DictReader(open(gp))}
    for cell in CELLS:
        print(f"== {cell} ==")
        out[cell] = run_cell(cell, man)
    for chain, (a, b) in {"base": ("gemma-2-2b-base", "gemma-2-27b-base"),
                          "IT": ("gemma-2-2b-it", "gemma-2-27b-it")}.items():
        if not out[a].get("aborted") and not out[b].get("aborted"):
            out[f"delta_top1_2b_to_27b_{chain}_raw"] = round(
                out[b]["raw"]["top1_mean"] - out[a]["raw"]["top1_mean"], 5)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zhang_z1_results.json").write_text(json.dumps(out, indent=1))
    man.write("zhang_z1_manifest.json")
    print("Z1 done; results + manifest written.")


if __name__ == "__main__":
    main()
