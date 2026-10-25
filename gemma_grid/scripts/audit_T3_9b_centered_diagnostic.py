# -*- coding: utf-8 -*-
"""Independent recomputation of the T.3 9B centered-correlation diagnostic.

Written against the METHOD DESCRIPTION in the 2026-10-09 dispatch only.
The producing script (research/diagnostics/diag_T3_9b.py) was not opened.

Method as given:
  profile = mean over dialogues of the cosine between each raw probe and the
            dialogue's pooled L28 activation
  centered = subtract the mean activation across dialogues before the cosine
  drop     = exclude the named dimensions from both probe and activation
             before normalizing

Reads .pt files without torch (tools/ptload.py). Prints a JSON verdict block.
"""
import hashlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.environ["HOME"], "tools"))
import ptload

ROOT = os.path.join(os.environ["HOME"], "mnt", "emotion_probes_project", "gemma_grid")
L = 28
PROBES = os.path.join(ROOT, "data/gemma-2-9b-it/vectors/canonical_100T.pt")
ACT = {"base": os.path.join(ROOT, "data/gemma-2-9b-base/vectors/gemma-2-9b-base__neutral_100T.pt"),
       "it":   os.path.join(ROOT, "data/gemma-2-9b-it/vectors/gemma-2-9b-it__neutral_100T.pt")}


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def profile(P, A, drop=None):
    """mean over dialogues of cos(probe, activation); drop dims from BOTH, then normalize."""
    if drop is not None and len(drop):
        keep = np.setdiff1d(np.arange(A.shape[1]), np.asarray(drop))
        P, A = P[:, keep], A[:, keep]
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    return (Pn @ An.T).mean(axis=1)          # (n_emotions,)


def r(a, b):
    return float(np.corrcoef(np.asarray(a, dtype=np.float64),
                             np.asarray(b, dtype=np.float64))[0, 1])


def main():
    out = {"inputs": {}, "claims": {}, "extras": {}}

    pb = ptload.load(PROBES)
    P = np.array(pb["vectors_raw"][:, L, :], dtype=np.float64)
    out["inputs"]["probes"] = {"path": os.path.relpath(PROBES, ROOT), "sha256": sha256(PROBES),
                              "key": "vectors_raw", "layer": L, "shape": list(P.shape),
                              "d_model_declared": pb.get("d_model"),
                              "n_emotions": len(pb.get("emotions", []))}
    del pb

    raw, cen, var = {}, {}, {}
    for cell, path in ACT.items():
        d = ptload.load(path)
        A = np.array(d["acts"][:, L, :], dtype=np.float64)
        out["inputs"][cell] = {"path": os.path.relpath(path, ROOT), "sha256": sha256(path),
                               "key": "acts", "layer": L, "shape": list(A.shape),
                               "src": d.get("src"), "src_sha256": d.get("src_sha256"),
                               "analysis_layer_declared": d["config"]["analysis_layer"],
                               "d_model_declared": d["config"]["d_model"]}
        raw[cell] = A
        cen[cell] = A - A.mean(axis=0, keepdims=True)
        var[cell] = A.var(axis=0)
        del d

    # --- claim 1 -------------------------------------------------------------
    pu = {c: profile(P, raw[c]) for c in raw}
    pc = {c: profile(P, cen[c]) for c in cen}
    out["claims"]["1_uncentered_r"] = r(pu["base"], pu["it"])
    out["claims"]["1_centered_r"] = r(pc["base"], pc["it"])

    # --- claim 2 -------------------------------------------------------------
    top = {c: np.argsort(-var[c])[:20] for c in var}
    union = np.union1d(top["base"], top["it"])
    out["claims"]["2_n_union20"] = int(union.size)
    out["claims"]["2_overlap"] = int(np.intersect1d(top["base"], top["it"]).size)
    out["claims"]["2_union_dims"] = sorted(int(x) for x in union)

    # --- claim 3 -------------------------------------------------------------
    pcu = {c: profile(P, cen[c], union) for c in cen}
    out["claims"]["3_centered_r_drop_union"] = r(pcu["base"], pcu["it"])
    pct = {c: profile(P, cen[c], top["it"]) for c in cen}
    out["claims"]["3_centered_r_drop_it_top20"] = r(pct["base"], pct["it"])

    # --- claim 4: odd/even dialogue split reliability, after union drop ------
    nd = raw["base"].shape[0]
    ev, od = np.arange(0, nd, 2), np.arange(1, nd, 2)
    for cell in raw:
        A = raw[cell]
        # (a) centering computed on the full 500, split afterwards
        C = cen[cell]
        ra = r(profile(P, C[ev], union), profile(P, C[od], union))
        # (b) centering recomputed within each half
        hb = [A[ix] - A[ix].mean(axis=0, keepdims=True) for ix in (ev, od)]
        rb = r(profile(P, hb[0], union), profile(P, hb[1], union))
        out["claims"]["4_reliability_%s_centered_on_full" % cell] = ra
        out["claims"]["4_reliability_%s_centered_within_half" % cell] = rb

    # --- claim 5 -------------------------------------------------------------
    out["claims"]["5_d_model"] = int(raw["base"].shape[1])
    out["claims"]["5_dim4140_exists_in_9B"] = bool(4140 < raw["base"].shape[1])
    out["claims"]["5_dim4140_in_union"] = bool(4140 in set(int(x) for x in union))

    # --- extras --------------------------------------------------------------
    rng = np.random.default_rng(0)
    trials = []
    for _ in range(200):
        drop = rng.choice(raw["base"].shape[1], size=union.size, replace=False)
        trials.append(r(profile(P, cen["base"], drop), profile(P, cen["it"], drop)))
    trials = np.array(trials)
    out["extras"]["random_drop_control"] = {
        "n_trials": 200, "k": int(union.size), "seed": 0,
        "mean_r": float(trials.mean()), "sd_r": float(trials.std(ddof=1)),
        "min_r": float(trials.min()), "max_r": float(trials.max()),
        "n_ge_observed": int((trials >= out["claims"]["3_centered_r_drop_union"]).sum())}
    for c in var:
        out["extras"]["variance_share_union_%s" % c] = float(var[c][union].sum() / var[c].sum())
        out["extras"]["variance_share_top20_%s" % c] = float(var[c][top[c]].sum() / var[c].sum())

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
