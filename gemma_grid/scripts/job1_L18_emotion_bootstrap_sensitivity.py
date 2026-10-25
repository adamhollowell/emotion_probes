#!/usr/bin/env python3
"""
Dependence-aware emotion bootstrap with the 2B cells read at layer 18 (paper section 5.3).

Producer of record for geometry_out/job1_L18_emotion_bootstrap_sensitivity.json, re-derived
on 2026-10-14 from the procedure that artifact serializes (dispatch
DISPATCH_LEDGER_COMPUTE_L18_AND_PERMUTATION_100T.md, run 2026-10-07). The original run retained no
script. The verify mode checks that every stored value is reproduced.

The template is gap_boot in phase4_c3_grade.py, unchanged except that the 2B analysis layer is 18
(the pre-registered 2B sensitivity layer) rather than 17. 27B stays at 31.

Per chain (base: 2b-base vs 27b-base; tuned: 2b-it vs 27b-it), four estimators: the analysis-layer
vectors and the static embedding-table vectors of each cell. PCA (center, SVD, scores = U * S) is fit
on all 171 emotions, PC2 is taken and restricted to the 45 Russell-Mehrabian overlap, and held fixed
outside the replicate loop. Per estimator r = |Pearson(PC2[sel], arousal[sel])|, clamped at 0.999999,
z = arctanh(r). The stored vectors are float32 and are cast to float64 before the PCA, which is the
reading that reproduces the stored values (verified 2026-10-14 against float32 SVD, which does not).
  Arousal rise: delta_z = z(r_an_27B) - z(r_an_2B).
  Depth gap per cell: z(r_an) - z(r_emb). delta = gap_27B - gap_2B.
B = 2000 rounds, one shared resample index per round applied to all four estimators, generator
numpy.random.default_rng(0) fresh per chain. SE = numpy.std(boots) with ddof 0. Two-sided normal
p = erfc(|delta / SE| / sqrt(2)). The layer-17 point estimates are reported as reference.

Label carried by the artifact: unregistered sensitivity analysis at a registered layer. It may
corroborate or complicate a registered verdict and never re-grades one.

Usage:
  python job1_L18_emotion_bootstrap_sensitivity.py            # recompute and verify
  python job1_L18_emotion_bootstrap_sensitivity.py --write    # recompute and overwrite the artifact
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import noise_floor as nf  # noqa: E402
import torch  # noqa: E402

B = 2000
SEED = 0
CLAMP = 0.999999
OUT = REPO / "gemma_grid" / "geometry_out"
ART = OUT / "job1_L18_emotion_bootstrap_sensitivity.json"
RATINGS = REPO / "data" / "ratings" / "russell_mehrabian_1977.csv"

VEC = {
    "gemma-2-2b-base": REPO / "gemma_grid/data/gemma-2-2b-base/vectors/canonical.pt",
    "gemma-2-2b-it": REPO / "gemma_grid/data/gemma-2-2b-it/vectors/canonical.pt",
    "gemma-2-27b-base": REPO / "data/vectors/emotion_vectors_171.pt",
    "gemma-2-27b-it": REPO / "gemma_grid/data/gemma-2-27b-it/vectors/canonical.pt",
}
LAYER = {"2B": 18, "27B": 31}
LAYER_REF = {"2B": 17, "27B": 31}
CHAINS = {"base": ("gemma-2-2b-base", "gemma-2-27b-base"), "tuned": ("gemma-2-2b-it", "gemma-2-27b-it")}

z = np.arctanh


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def p2(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2))


def pc2_sets(cell_small, cell_large, layer_small):
    """PC2 coordinates on the 45-emotion overlap for the four estimators, held fixed."""
    pc2 = {}
    em = None
    for c, size in ((cell_small, "2B"), (cell_large, "27B")):
        em_c, V = nf.load_vectors(VEC[c])
        em = em_c
        Vl = V[:, layer_small if size == "2B" else LAYER["27B"], :].astype(np.float64)
        emb = torch.load(OUT / c / "c3_vectors.pt", map_location="cpu", weights_only=False)
        Ve = emb["vectors"].float().numpy()[:, 0, :].astype(np.float64)
        for which, V_ in (("an", Vl), ("emb", Ve)):
            co, _ = nf.pca(V_, 5)
            pc2[(size, which)] = co[:, 1]
    idx45, _, ar = nf.load_ratings(RATINGS, em)
    idx45 = np.asarray(idx45)
    return {k: v[idx45] for k, v in pc2.items()}, ar, len(idx45)


def r_of(s2, ar, size, which, sel):
    rr = np.corrcoef(s2[(size, which)][sel], ar[sel])[0, 1]
    return min(abs(rr), CLAMP)


def measures(s2, ar, sel):
    r2, r27 = r_of(s2, ar, "2B", "an", sel), r_of(s2, ar, "27B", "an", sel)
    gap2 = z(r2) - z(r_of(s2, ar, "2B", "emb", sel))
    gap27 = z(r27) - z(r_of(s2, ar, "27B", "emb", sel))
    return r27 - r2, z(r27) - z(r2), gap2, gap27


def run_chain(cell_small, cell_large):
    s2, ar, n = pc2_sets(cell_small, cell_large, LAYER["2B"])
    full = np.arange(n)
    d_r, d_z, gap2, gap27 = measures(s2, ar, full)
    rng = np.random.default_rng(SEED)
    boot_rise, boot_gap = np.empty(B), np.empty(B)
    for b in range(B):
        sel = rng.integers(0, n, n)
        _, bz, g2, g27 = measures(s2, ar, sel)
        boot_rise[b] = bz
        boot_gap[b] = g27 - g2
    se_rise, se_gap = float(np.std(boot_rise)), float(np.std(boot_gap))
    s2r, arr, nr = pc2_sets(cell_small, cell_large, LAYER_REF["2B"])
    rd_r, rd_z, rgap2, rgap27 = measures(s2r, arr, np.arange(nr))
    return {
        "arousal_rise": {"delta_r": float(d_r), "delta_z": float(d_z), "boot_SE": se_rise,
                         "p_dep_aware": p2(d_z / se_rise)},
        "depth_gap": {"gap_z_2B": float(gap2), "gap_z_27B": float(gap27), "delta": float(gap27 - gap2),
                      "boot_SE": se_gap, "p_dep_aware": p2((gap27 - gap2) / se_gap)},
        "L17_reference": {"arousal_rise_delta_r": float(rd_r), "arousal_rise_delta_z": float(rd_z),
                          "depth_gap_2B_gapz": float(rgap2), "depth_gap_delta": float(rgap27 - rgap2)},
    }


def input_manifest() -> dict:
    m = {}
    for c, p in VEC.items():
        m[f"{c}/{p.name}"] = sha(p)
    for c in VEC:
        m[f"{c}/c3_vectors.pt"] = sha(OUT / c / "c3_vectors.pt")
    m[RATINGS.name] = sha(RATINGS)
    return m


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    stored = json.loads(ART.read_text())
    man = input_manifest()
    bad = [k for k, v in stored["sr5_input_manifest"].items() if man.get(k.replace("_100T", "")) != v]  # stored keys carry the as-run file names
    if bad:
        sys.exit(f"input hash mismatch against the artifact's manifest: {bad}")
    results = {chain: run_chain(*cells) for chain, cells in CHAINS.items()}
    if a.write:
        new = {k: v for k, v in stored.items() if k not in ("results", "sr5_input_manifest", "generated_utc")}
        new["results"] = results
        new["sr5_input_manifest"] = man
        new["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ART.write_text(json.dumps(new, indent=2))
        print("wrote", ART)
        sys.exit(0)
    ok, worst = True, 0.0
    for k, v in flatten(stored["results"]):
        n = dict(flatten(results))[k]
        if n != v:
            ok = False
            worst = max(worst, abs(n - v))
            print(f"MISMATCH {k}: stored {v!r} recomputed {n!r}")
    print("numpy", np.__version__, "torch", torch.__version__)
    print("VERIFIED: every stored value reproduced exactly" if ok else f"VERIFICATION FAILED (largest abs diff {worst:.3e})")
    sys.exit(0 if ok else 1)
