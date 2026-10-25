#!/usr/bin/env python3
"""G5.4a displacement model (FROZEN analysis) — does "moral/self-conscious moves most"
survive controlling for valence extremity and nuisance covariates?

FROZEN before execution (Ledger Chat, 2026-08-27). Path args free; constants below are not.
Producer does not grade. Output: geometry_out/g54a_displacement_model.json

FROZEN DESIGN
- Outcome: per-emotion post-Procrustes displacement, per tuning pair (three pairs), from the
  stageB displacement CSVs; pooled model = mean displacement across the three tuning pairs.
- Covariates (standardized): |valence| (NRC-VAD v2.1, 161/171 coverage; unrated emotions
  dropped and listed), arousal (NRC-VAD), 10-NN purity (per-emotion, cross-cell mean),
  L0 own-word median rank (log10, per-emotion, cross-cell mean over the four readable
  2B/9B cells), zipf frequency (wordfreq 3.1.1, ' word' lemma).
- Indicator (the tested variable): SELF-CONSCIOUS/MORAL membership, frozen list from
  Tangney et al. 2007 / Haidt 2003 as already cited at G5.4a, intersected with the 171:
  {ashamed, guilty, embarrassed, humiliated, proud, remorseful, regretful, self-critical}.
  The intersection actually used is reported. The vindictive-trio is NOT in the indicator
  (outside the canonical families, per the ledger's own labeling).
- Model: OLS displacement ~ indicator + |valence| + arousal + purity + log10(L0 rank) + zipf.
  Reported: standardized betas, p-values (HC3 robust), R2, and the indicator's beta/p WITH
  and WITHOUT the covariates.
- Pre-stated survival criterion (graded by the Ledger Chat): the indicator retains positive
  sign and p < .05 (HC3) in the pooled model with all covariates present. Per-pair models
  reported descriptively.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

INDICATOR = ["ashamed","guilty","embarrassed","humiliated","proud",
             "remorseful","regretful","self-critical"]  # FROZEN

def zscore(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def ols_hc3(X, y):
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    u = (e / (1 - h)) ** 2
    V = XtX_inv @ (X.T * u) @ X @ XtX_inv
    se = np.sqrt(np.diag(V))
    from math import erf, sqrt
    z = b / (se + 1e-300)
    p = [2 * (1 - 0.5 * (1 + erf(abs(zz) / sqrt(2)))) for zz in z]
    ss_res = float(e @ e); ss_tot = float(((y - y.mean()) ** 2).sum())
    return b, se, p, 1 - ss_res / ss_tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--displacement-csvs", nargs=3, required=True,
                    help="stageB displacement CSVs for the THREE tuning pairs (2B, 9B, 27B)")
    ap.add_argument("--nrc-csv", required=True, help="nrc_vad_v2_1.csv (emotion,valence,arousal columns)")
    ap.add_argument("--purity-json", required=True, help="per-emotion 10-NN purity per cell (G6.5 artifact)")
    ap.add_argument("--l0-ranks-json", required=True, help="per-emotion L0 own-word rank per cell (lens artifact)")
    ap.add_argument("--out", default="geometry_out/g54a_displacement_model.json")
    args = ap.parse_args()

    import csv
    disp = {}
    for path in args.displacement_csvs:
        with open(path) as f:
            for row in csv.DictReader(f):
                emo = row.get("emotion") or row.get("emo") or list(row.values())[0]
                val = [v for k, v in row.items() if k not in ("emotion", "emo")][ -1 ]
                disp.setdefault(emo, []).append(float(val))
    emos = sorted(e for e, v in disp.items() if len(v) == 3)
    assert len(emos) >= 160, f"unexpected emotion coverage: {len(emos)} with 3 pairs"

    nrc = {}
    with open(args.nrc_csv) as f:
        for row in csv.DictReader(f):
            term = (row.get("term") or row.get("word") or row.get("emotion") or "").strip().lower()
            try:
                nrc[term] = (float(row.get("valence") or row.get("V")), float(row.get("arousal") or row.get("A")))
            except (TypeError, ValueError):
                continue

    purity = json.loads(Path(args.purity_json).read_text())
    l0 = json.loads(Path(args.l0_ranks_json).read_text())
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        raise SystemExit("wordfreq required (3.1.1 per P4.5 precedent)")

    def per_emotion_mean(d, emo):
        vals = [d[c][emo] for c in d if emo in d[c]]
        return float(np.mean(vals)) if vals else np.nan

    rows, dropped = [], []
    for e in emos:
        if e not in nrc:
            dropped.append(e); continue
        v, a = nrc[e]
        pur = per_emotion_mean(purity, e)
        # L0 rank averaged over the four readable 2B/9B cells only (frozen)
        l0_cells = {c: r for c, r in l0.items() if ("2b" in c or "9b" in c)}
        l0r = per_emotion_mean(l0_cells, e)
        rows.append({"emotion": e, "disp_pooled": float(np.mean(disp[e])),
                     "disp_per_pair": disp[e], "abs_val": abs(v - np.nanmean([x[0] for x in nrc.values()])),
                     "arousal": a, "purity": pur, "log_l0": np.log10(max(l0r, 1.0)) if np.isfinite(l0r) else np.nan,
                     "zipf": zipf_frequency(e.replace("-", " "), "en"),
                     "ind": 1.0 if e in INDICATOR else 0.0})
    used_indicator = sorted(set(INDICATOR) & set(r["emotion"] for r in rows))

    def build(rows, keys):
        ok = [r for r in rows if all(np.isfinite(r[k]) for k in keys)]
        y = zscore(np.array([r["disp_pooled"] for r in ok]))
        cols = [np.ones(len(ok))] + [np.array([r["ind"] for r in ok])] + \
               [zscore(np.array([r[k] for r in ok])) for k in keys if k != "ind"]
        return np.column_stack(cols), y, len(ok)

    covs = ["abs_val", "arousal", "purity", "log_l0", "zipf"]
    Xf, yf, nf = build(rows, covs)
    bf, sef, pf, r2f = ols_hc3(Xf, yf)
    Xn, yn, nn = build(rows, [])
    bn, sen, pn, r2n = ols_hc3(Xn, yn)

    out = {"frozen_indicator": INDICATOR, "indicator_used": used_indicator,
           "dropped_unrated": dropped, "n_full_model": nf,
           "full_model": {"terms": ["intercept", "indicator"] + covs,
                          "beta": [float(x) for x in bf], "se": [float(x) for x in sef],
                          "p_hc3": [float(x) for x in pf], "r2": r2f},
           "indicator_only": {"beta": float(bn[1]), "p_hc3": float(pn[1]), "r2": r2n},
           "survival_criterion": "indicator beta > 0 and p<.05 (HC3) in the full pooled model",
           "per_pair_descriptive": {}}
    for i, name in enumerate(["tuning_2B", "tuning_9B", "tuning_27B"]):
        for r in rows: r["disp_pooled"] = r["disp_per_pair"][i]
        Xp, yp, npair = build(rows, covs)
        bp, sep, pp, r2p = ols_hc3(Xp, yp)
        out["per_pair_descriptive"][name] = {"indicator_beta": float(bp[1]),
                                             "indicator_p_hc3": float(pp[1]), "r2": r2p, "n": npair}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("sha256:", hashlib.sha256(Path(args.out).read_bytes()).hexdigest())
    print("full-model indicator beta/p:", out["full_model"]["beta"][1], out["full_model"]["p_hc3"][1])

if __name__ == "__main__":
    main()
