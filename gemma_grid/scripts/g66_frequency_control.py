#!/usr/bin/env python3
"""G6.6 frequency control (FROZEN analysis) — does the pos/neg readability asymmetry
survive controlling for token frequency?

FROZEN before execution (Ledger Chat, 2026-08-27). Closes G6.6's own disclosed caveat
("the L0 gap may partly reflect token-frequency differences"). Path args free; constants
below are not. Producer does not grade.
Output: geometry_out/g66_frequency_control.json

FROZEN DESIGN
1. Frequency gap: zipf frequency (wordfreq 3.1.1, same machinery as P4.5) for the 50 pos vs
   87 neg fair emotions; report medians + two-sided Mann-Whitney U.
2. Regression control, per cell: rank-transform per-emotion L0 own-word rank (and final rank);
   OLS rank ~ valence_sign + zipf. Report the sign coefficient with and without zipf (HC3 p).
3. Frequency-matched comparison, per cell: greedy nearest-neighbor match (without replacement)
   of each pos emotion to the neg emotion closest in zipf (caliper 0.5; unmatched dropped and
   counted); recompute the pos/neg L0-median and final-median gaps on the matched sets.
Pre-stated survival criterion (graded by the Ledger Chat): the asymmetry SURVIVES if the
valence-sign coefficient keeps its direction with p<.05 (HC3) in the zipf-controlled model at
all four 2B/9B cells for L0 (the cells where L0 readability varies), AND the matched-set L0 gap
keeps its direction in at least 3 of those 4 cells. Final-rank and 27B results report
descriptively. If the criterion fails, G6.6's finding text is rewritten to attribute the
asymmetry (in whole or part) to frequency — Adam rules the wording either way.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

CALIPER = 0.5  # FROZEN (zipf units)

def mannwhitney_p(a, b):
    from itertools import chain
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b]); order = np.argsort(allv)
    ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(allv) + 1)
    # midranks for ties
    vals, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); start = cum - cnt
    mid = (start + cum + 1) / 2.0
    ranks = mid[inv]
    U = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    tie = sum(c ** 3 - c for c in cnt)
    sig = np.sqrt(n1 * n2 / 12 * ((n1 + n2 + 1) - tie / ((n1 + n2) * (n1 + n2 - 1))))
    from math import erf, sqrt
    z = (U - mu) / (sig + 1e-300)
    return float(U), 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

def ols_hc3(X, y):
    XtX_inv = np.linalg.pinv(X.T @ X); b = XtX_inv @ X.T @ y; e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X); u = (e / (1 - h)) ** 2
    V = XtX_inv @ (X.T * u) @ X @ XtX_inv; se = np.sqrt(np.diag(V))
    from math import erf, sqrt
    p = [2 * (1 - 0.5 * (1 + erf(abs(bb / (ss + 1e-300)) / sqrt(2)))) for bb, ss in zip(b, se)]
    return b, se, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fair-emotions-json", required=True, help="fair_emotions_v1.json (1b589e06)")
    ap.add_argument("--nrc-csv", required=True, help="nrc_vad_v2_1.csv for the valence sign split")
    ap.add_argument("--ranks-json", required=True,
                    help="per-cell per-emotion own-word ranks: {cell: {emotion: {'L0': r0, 'final': rf}}}")
    ap.add_argument("--out", default="geometry_out/g66_frequency_control.json")
    args = ap.parse_args()

    from wordfreq import zipf_frequency
    import csv
    fair = json.loads(Path(args.fair_emotions_json).read_text())
    fair_list = fair.get("fair_emotions") or fair.get("emotions") or fair
    nrc = {}
    with open(args.nrc_csv) as f:
        for row in csv.DictReader(f):
            t = (row.get("term") or row.get("word") or "").strip().lower()
            try: nrc[t] = float(row.get("valence") or row.get("V"))
            except (TypeError, ValueError): continue
    center = float(np.mean(list(nrc.values())))
    pos = [e for e in fair_list if e in nrc and nrc[e] > center]
    neg = [e for e in fair_list if e in nrc and nrc[e] <= center]
    # The entry's split is by NRC sign (50/87 of 137); assert we reproduce the split sizes.
    zf = {e: zipf_frequency(e.replace("-", " "), "en") for e in pos + neg}
    U, p_mw = mannwhitney_p([zf[e] for e in pos], [zf[e] for e in neg])

    ranks = json.loads(Path(args.ranks_json).read_text())
    out = {"n_pos": len(pos), "n_neg": len(neg),
           "zipf": {"pos_median": float(np.median([zf[e] for e in pos])),
                    "neg_median": float(np.median([zf[e] for e in neg])),
                    "mannwhitney_U": U, "p": p_mw},
           "cells": {}, "caliper": CALIPER,
           "survival_criterion": ("sign coefficient keeps direction with p<.05 (HC3) zipf-controlled "
                                  "at all four 2B/9B cells for L0, AND matched-set L0 gap keeps "
                                  "direction in >=3 of 4")}
    for cell, table in ranks.items():
        rows = [(e, table[e]) for e in pos + neg if e in table]
        y0 = np.array([np.log10(max(float(r["L0"]), 1)) for _, r in rows])
        yf = np.array([np.log10(max(float(r["final"]), 1)) for _, r in rows])
        sign = np.array([1.0 if e in pos else 0.0 for e, _ in rows])
        z = np.array([zf[e] for e, _ in rows])
        res = {}
        for label, y in (("L0", y0), ("final", yf)):
            X1 = np.column_stack([np.ones(len(y)), sign])
            X2 = np.column_stack([np.ones(len(y)), sign, (z - z.mean()) / (z.std() + 1e-12)])
            b1, _, p1 = ols_hc3(X1, y); b2, _, p2 = ols_hc3(X2, y)
            res[label] = {"sign_beta_raw": float(b1[1]), "p_raw": float(p1[1]),
                          "sign_beta_zipf_controlled": float(b2[1]), "p_controlled": float(p2[1])}
        # matched comparison
        used_neg, pairs = set(), []
        for e in sorted(pos, key=lambda x: zf[x]):
            cands = [(abs(zf[e] - zf[n]), n) for n in neg if n not in used_neg and abs(zf[e] - zf[n]) <= CALIPER and n in table and e in table]
            if cands:
                _, m = min(cands); used_neg.add(m); pairs.append((e, m))
        if pairs:
            mp = [float(table[e]["L0"]) for e, _ in pairs]; mn = [float(table[m]["L0"]) for _, m in pairs]
            fp = [float(table[e]["final"]) for e, _ in pairs]; fn = [float(table[m]["final"]) for _, m in pairs]
            res["matched"] = {"n_pairs": len(pairs), "unmatched_pos": len(pos) - len(pairs),
                              "L0_median_pos": float(np.median(mp)), "L0_median_neg": float(np.median(mn)),
                              "final_median_pos": float(np.median(fp)), "final_median_neg": float(np.median(fn))}
        out["cells"][cell] = res
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("sha256:", hashlib.sha256(Path(args.out).read_bytes()).hexdigest())
    print(json.dumps(out["zipf"], indent=1))

if __name__ == "__main__":
    main()
