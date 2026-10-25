#!/usr/bin/env python3
"""P-map: cross-cell linear transport of emotion vectors (FROZEN analysis).

Spec of record: gemma_grid/P_MAP_TRANSPORT_SPEC_FROZEN_100T.md
(sha256 09b4dacdafa6e3b0e9aa4b0c4d75b4a44d2aaa74b2c041303e7919ae5833ad38).
FROZEN constants below must not be changed by the runner. Path arguments are free.
Producer does not grade. Output: geometry_out_100T/pmap_transport_100T.json
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

# ---------------- FROZEN constants (do not edit) ----------------
FOLD_SEED = 0
N_FOLDS = 5
N_PERM = 10_000
PERM_SEED = 0
RIDGE_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
PRIMARY_SOURCE, PRIMARY_TARGET = "gemma-2-2b-base", "gemma-2-27b-base"
ANALYSIS_LAYERS = {"2b": 17, "9b": 28, "27b": 31}
# Grading bar (applied by the Ledger Chat, echoed here for the record):
#   CONFIRMED: median held-out rank <= 2 AND permutation p < .001 (primary pair)
#   NULL: p < .001 but median rank > 2 ; DISCONFIRMED: not resolved from floor.
# ----------------------------------------------------------------

def load_vectors(path):
    """Load a cell's cleaned 100T 171xd matrix + emotion list. 100T guard enforced."""
    p = str(path)
    assert "100T" in p, f"100T guard: cleaned-vector path lacks '100T': {p}"
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    # Expected keys per house convention; fail loudly otherwise.
    vecs = blob["vectors"] if isinstance(blob, dict) and "vectors" in blob else None
    emos = blob.get("emotions") if isinstance(blob, dict) else None
    if vecs is None or emos is None:
        raise SystemExit(f"unexpected .pt schema at {p}: keys={list(blob.keys()) if isinstance(blob,dict) else type(blob)}")
    V = np.asarray(vecs, dtype=np.float64)
    assert V.shape[0] == 171, f"expected 171 rows, got {V.shape}"
    return V, [str(e) for e in emos]

def center(V):
    return V - V.mean(axis=0, keepdims=True)  # idempotent: cleaned vectors are per-cell mean-centered

def fit_procrustes(X, Y):
    """Semi-orthogonal W (dX->dY) + global scale s minimizing ||s XW - Y||_F."""
    C = X.T @ Y                       # dX x dY cross-covariance
    U, _, Vt = np.linalg.svd(C, full_matrices=False)
    W = U @ Vt                        # orthonormal columns (rank = min(dX,dY))
    num = float(np.sum((X @ W) * Y)); den = float(np.sum((X @ W) ** 2))
    s = num / den if den > 0 else 1.0
    return W, s

def fit_ridge(X, Y, lam):
    dX = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(dX), X.T @ Y)

def cosine_rows(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T

def eval_transport(Xs, Yt, fold_ids, map_kind="procrustes"):
    """5-fold CV; returns per-emotion (rank, cosine) pooled across folds."""
    n = Xs.shape[0]; ranks = np.zeros(n, int); coss = np.zeros(n)
    for f in range(N_FOLDS):
        tr, te = fold_ids != f, fold_ids == f
        if map_kind == "procrustes":
            W, s = fit_procrustes(Xs[tr], Yt[tr]); pred = s * (Xs[te] @ W)
        else:
            best = None
            rs = np.random.RandomState(FOLD_SEED)
            inner = rs.randint(0, N_FOLDS, size=tr.sum())
            for lam in RIDGE_GRID:
                errs = []
                Xtr, Ytr = Xs[tr], Yt[tr]
                for g in range(N_FOLDS):
                    itr, ite = inner != g, inner == g
                    B = fit_ridge(Xtr[itr], Ytr[itr], lam)
                    errs.append(float(np.linalg.norm(Xtr[ite] @ B - Ytr[ite])))
                m = float(np.mean(errs))
                if best is None or m < best[0]: best = (m, lam)
            B = fit_ridge(Xs[tr], Yt[tr], best[1]); pred = Xs[te] @ B
        S = cosine_rows(pred, Yt)                     # held-out preds vs ALL 171 targets
        idx = np.where(te)[0]
        for row, j in enumerate(idx):
            order = np.argsort(-S[row])
            ranks[j] = int(np.where(order == j)[0][0]) + 1
            coss[j] = float(S[row, j])
    return ranks, coss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="JSON mapping cell name -> vectors .pt path (all six)")
    ap.add_argument("--out", default="geometry_out_100T/pmap_transport_100T.json")
    ap.add_argument("--replicate-a", default=None, help="Branch A only: replicate vector set A for a primary-pair cell")
    ap.add_argument("--replicate-b", default=None, help="Branch A only: replicate vector set B (same cell)")
    ap.add_argument("--ceiling-branch", choices=["A", "B"], required=True,
                    help="Selected by the Coordinator's artifact inventory BEFORE execution")
    args = ap.parse_args()

    cells = json.loads(Path(args.cells).read_text())
    data = {}
    emos_ref = None
    for cell, path in cells.items():
        V, emos = load_vectors(path)
        if emos_ref is None: emos_ref = emos
        assert emos == emos_ref, f"emotion order mismatch at {cell}"
        data[cell] = center(V)

    rs = np.random.RandomState(FOLD_SEED)
    fold_ids = rs.permutation(np.arange(171) % N_FOLDS)

    results = {"spec_sha256": "09b4dacdafa6e3b0e9aa4b0c4d75b4a44d2aaa74b2c041303e7919ae5833ad38",
               "frozen": {"fold_seed": FOLD_SEED, "n_folds": N_FOLDS, "n_perm": N_PERM,
                          "perm_seed": PERM_SEED, "ridge_grid": RIDGE_GRID},
               "ceiling_branch": args.ceiling_branch, "pairs": {}}

    def run_pair(src, tgt, with_perm=False, with_ridge=False):
        Xs, Yt = data[src], data[tgt]
        ranks, coss = eval_transport(Xs, Yt, fold_ids, "procrustes")
        out = {"median_rank": float(np.median(ranks)),
               "top1_acc": float(np.mean(ranks == 1)), "top5_acc": float(np.mean(ranks <= 5)),
               "median_cosine": float(np.median(coss))}
        if with_ridge:
            r2, c2 = eval_transport(Xs, Yt, fold_ids, "ridge")
            out["ridge_secondary"] = {"median_rank": float(np.median(r2)),
                                      "top1_acc": float(np.mean(r2 == 1)),
                                      "median_cosine": float(np.median(c2))}
        if with_perm:
            prs = np.random.RandomState(PERM_SEED)
            obs = float(np.mean(ranks == 1)); count = 0
            for _ in range(N_PERM):
                perm = prs.permutation(171)
                pr, _ = eval_transport(Xs, Yt[perm], fold_ids, "procrustes")
                if float(np.mean(pr == 1)) >= obs: count += 1
            out["perm_p_top1"] = (1 + count) / (N_PERM + 1)
            out["perm_note"] = "add-one estimator; statistic = top-1 accuracy; frozen"
        return out

    # Primary pair (graded), full treatment
    results["pairs"][f"{PRIMARY_SOURCE}->{PRIMARY_TARGET}"] = run_pair(
        PRIMARY_SOURCE, PRIMARY_TARGET, with_perm=True, with_ridge=True)
    # Reverse primary + nine registered pairs, descriptive
    results["pairs"][f"{PRIMARY_TARGET}->{PRIMARY_SOURCE}"] = run_pair(PRIMARY_TARGET, PRIMARY_SOURCE)
    names = list(cells)
    for a in names:
        for b in names:
            if a >= b: continue
            key = f"{a}->{b}"
            if key not in results["pairs"]: results["pairs"][key] = run_pair(a, b)

    if args.ceiling_branch == "A":
        if not (args.replicate_a and args.replicate_b):
            raise SystemExit("Branch A requires --replicate-a/--replicate-b")
        Ra, _ = load_vectors(args.replicate_a); Rb, _ = load_vectors(args.replicate_b)
        rr, rc = eval_transport(center(Ra), center(Rb), fold_ids, "procrustes")
        results["ceiling_replicate"] = {"median_rank": float(np.median(rr)),
                                        "top1_acc": float(np.mean(rr == 1)),
                                        "median_cosine": float(np.median(rc))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("sha256:", hashlib.sha256(Path(args.out).read_bytes()).hexdigest())
    print(json.dumps(results["pairs"][f"{PRIMARY_SOURCE}->{PRIMARY_TARGET}"], indent=1))

if __name__ == "__main__":
    main()
