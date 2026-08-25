#!/usr/bin/env python3
"""P1.5: representational similarity of natural-text co-activation vs vector geometry.
RSA between (a) the cosine similarity matrix of the 171 cleaned L31 vectors and
(b) the correlation matrix of their per-token projections on external human text
(723 v2 per-token documents; winner-selected sample, disclosed).
Variants: raw; within-document centered (removes doc-topic covariance);
bulk-only (drops tokens where any emotion exceeds its own p99, removing the
shared extreme tail). Convention: Pearson (and Spearman) over the 14,535
upper-triangle pairs, matching the in-house layer-stability RSA.
Note the isotropy null: under isotropic residual covariance, co-activation
correlation equals cosine exactly; the finding is how closely and where the
empirical covariance preserves the geometry.
Writes data/heldout/maxact/rsa_coactivation.json.
Run from project root: python3 scripts/analyze_rsa_coactivation.py"""
import hashlib
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path("data/heldout/maxact")
VECTORS = Path("data/vectors/emotion_vectors_171.pt")
LAYER = 31


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- basis gate: refuse to run on non-canonical vectors. S2 landmine: the bare filename
# identify the vectors by their sha256, not the filename. ---
VECTORS_SHA256_EXPECTED = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"  # canonical
_vsha = sha256_file(VECTORS)
assert _vsha == VECTORS_SHA256_EXPECTED, (
    f"{VECTORS} basis mismatch: {_vsha[:12]}… != expected canonical "
    f"{VECTORS_SHA256_EXPECTED[:12]}… — refusing to run RSA on a non-canonical basis")


def quantile_from_counts(counts, edges, q):
    c = np.cumsum(counts)
    idx = int(np.searchsorted(c, q * c[-1]))
    return float(edges[min(idx, len(counts) - 1) + 1])


def corr_from_moments(S, m, n):
    cov = S / n - np.outer(m / n, m / n)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return cov / np.outer(sd, sd)


def spearman_ut(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = torch.load(VECTORS, map_location="cpu", weights_only=False)
    em = list(p["emotions"])
    V = p["vectors"][:, LAYER, :].float().numpy()
    Vn = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
    COS = Vn @ Vn.T

    r = json.loads((BASE / "pass1_results.json").read_text())
    assert list(r["by_emotion"].keys()) == em, "emotion order mismatch"
    doc_ids = sorted({w["doc_id"] for e in em for w in r["by_emotion"][e]})

    hz = np.load(BASE / "histograms.npz", allow_pickle=True)
    edges = np.linspace(float(hz["lo"]), float(hz["hi"]), int(hz["bins"]) + 1)
    p99 = np.array([quantile_from_counts(hz["counts"][i], edges, 0.99) for i in range(len(em))])

    pt_dir = BASE / "pertoken"
    # the released pool ships the text-stripped tar (proj arrays preserved, token strings
    # removed — RSA reads only "proj"); fall back to the full tar for a project-side run.
    _tar = BASE / "pertoken_stripped.tar"
    if not _tar.exists():
        _tar = BASE / "pertoken.tar"
    tar = tarfile.open(_tar) if _tar.exists() else None

    def load_npz(doc_id):
        pth = pt_dir / (doc_id + ".npz")
        if pth.exists():
            return np.load(pth, allow_pickle=True)
        return np.load(tar.extractfile("pertoken/%s.npz" % doc_id), allow_pickle=True)

    k = len(em)
    S_raw = np.zeros((k, k)); m_raw = np.zeros(k); n_raw = 0
    S_cen = np.zeros((k, k)); n_cen_docs = 0
    S_blk = np.zeros((k, k)); m_blk = np.zeros(k); n_blk = 0

    for i, d in enumerate(doc_ids):
        X = load_npz(d)["proj"][1:].astype(np.float64)  # drop BOS row
        if X.shape[0] < 2:
            continue
        S_raw += X.T @ X; m_raw += X.sum(0); n_raw += X.shape[0]
        Xc = X - X.mean(0, keepdims=True)
        S_cen += Xc.T @ Xc; n_cen_docs += 1
        keep = ~(X > p99[None, :]).any(axis=1)
        Xb = X[keep]
        if Xb.shape[0]:
            S_blk += Xb.T @ Xb; m_blk += Xb.sum(0); n_blk += Xb.shape[0]
        if (i + 1) % 200 == 0:
            print("  ...%d/%d docs" % (i + 1, len(doc_ids)))

    C_raw = corr_from_moments(S_raw, m_raw, n_raw)
    covc = S_cen / n_raw
    sdc = np.sqrt(np.clip(np.diag(covc), 1e-12, None))
    C_cen = covc / np.outer(sdc, sdc)
    C_blk = corr_from_moments(S_blk, m_blk, n_blk)

    iu = np.triu_indices(k, 1)
    cos_ut = COS[iu]
    variants = {}
    for name, C in (("raw", C_raw), ("within_doc_centered", C_cen), ("bulk_only_below_p99", C_blk)):
        ut = C[iu]
        pear = float(np.corrcoef(cos_ut, ut)[0, 1])
        spear = spearman_ut(cos_ut, ut)
        resid = ut - cos_ut
        order = np.argsort(-np.abs(resid))
        devs = [{"pair": "%s | %s" % (em[iu[0][j]], em[iu[1][j]]),
                 "cosine": round(float(cos_ut[j]), 3),
                 "coactivation": round(float(ut[j]), 3),
                 "delta": round(float(resid[j]), 3)} for j in order[:10]]
        variants[name] = {"rsa_pearson": round(pear, 4), "rsa_spearman": round(spear, 4),
                          "top_deviations": devs}
        print("%-22s RSA pearson %.4f  spearman %.4f" % (name, pear, spear))

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"vectors_sha256": sha256_file(VECTORS),
                   "pass1_results_sha256": sha256_file(BASE / "pass1_results.json"),
                   "histograms_sha256": sha256_file(BASE / "histograms.npz")},
        "layer": LAYER, "n_pairs": int(len(cos_ut)),
        "n_docs": n_cen_docs, "n_tokens_raw": int(n_raw), "n_tokens_bulk": int(n_blk),
        "sample_note": "723 winner-selected documents (high-activation-biased sample), disclosed",
        "isotropy_note": "under isotropic residual covariance co-activation correlation equals cosine; RSA measures preservation of geometry by the empirical covariance",
        "variants": variants,
    }
    jpath = BASE / "rsa_coactivation.json"
    jpath.write_text(json.dumps(out, indent=1))
    print("n_tokens raw %d | bulk %d | wrote %s" % (n_raw, n_blk, jpath))


if __name__ == "__main__":
    main()
