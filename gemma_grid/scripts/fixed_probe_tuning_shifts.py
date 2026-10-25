#!/usr/bin/env python3
"""Fixed-probe tuning-shift analogs of Anthropic's base-vs-post-training section.

UNREGISTERED EXPLORATORY (Adam-ordered, 2026-08-10). Canonical vectors only. M.9: every
input hashed into a run-time manifest.

Three analyses, per size pair (2B, 9B, 27B):
  A1 (Table-16 analog): per-emotion probe-activation shift [IT - base] on the
     500 shared neutral dialogues, probes held FIXED (primary: IT-cell raw
     mean vectors, matching Anthropic's post-trained-probe convention;
     sensitivity: base-cell probes). Activation = mean over dialogues of
     cosine(probe_e, pooled activation) at the analysis layer. Primary
     convention uncentered (faithful to Anthropic's cosine-on-activations);
     sensitivity: per-cell mean-centered activations.
  A2 (Fig-84 analog + two-block): per-layer |shift| profile and the
     layer-by-layer correlation matrix of shift vectors; block-split search.
  A3 (structure preservation): r(a_base, a_IT) per size (their r=0.83 analog);
     cross-size shift-consistency r's (grid version of their r=0.90).
Comparison column: Claude Table 16 diffs parsed from
  gemma_grid/anthropic_originals (join on emotion name).
"""
import json, math, re, hashlib, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptload

REPO = Path(__file__).resolve().parents[2]
GG = REPO / "gemma_grid"

PAIRS = {
    "2B": {"base": ("gemma_grid/data/gemma-2-2b-base/vectors/canonical.pt",
                    "gemma_grid/data/gemma-2-2b-base/vectors/gemma-2-2b-base__neutral.pt", 17),
           "it":   ("gemma_grid/data/gemma-2-2b-it/vectors/canonical.pt",
                    "gemma_grid/data/gemma-2-2b-it/vectors/gemma-2-2b-it__neutral.pt", 17)},
    "9B": {"base": ("gemma_grid/data/gemma-2-9b-base/vectors/canonical.pt",
                    "gemma_grid/data/gemma-2-9b-base/vectors/gemma-2-9b-base__neutral.pt", 28),
           "it":   ("gemma_grid/data/gemma-2-9b-it/vectors/canonical.pt",
                    "gemma_grid/data/gemma-2-9b-it/vectors/gemma-2-9b-it__neutral.pt", 28)},
    "27B": {"base": ("data/vectors/emotion_vectors_171.pt", "gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__neutral.pt", 31),
            "it":  ("gemma_grid/data/gemma-2-27b-it/vectors/canonical.pt",
                    "gemma_grid/data/gemma-2-27b-it/vectors/gemma-2-27b-it__neutral.pt", 31)},
}

MANIFEST = {}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    MANIFEST[str(Path(p).relative_to(REPO))] = h.hexdigest()

def normr(M):
    return M / np.linalg.norm(M, axis=-1, keepdims=True)

def parse_table16():
    f = GG / "anthropic_originals" / "Anthropic Emotion Vectors - original essay Base and IT sections.txt"
    if not f.exists():  # third-party essay text, not redistributed; only the Claude comparison reads it
        print("[table16] skipped: needs Table 16 of Sofroniew et al. (2026); see the README")
        return None
    sha(f)
    txt = open(f, encoding="utf-8-sig").read()
    seg = txt[txt.index("EmotionDiffBasePost-Trained"): txt.index("Table 16:")]
    rows = {}
    for line in seg.splitlines():
        m = re.match(r"^([a-z][a-z\- ]*?)([+-]\d\.\d{4})(-?\d\.\d{4})(-?\d\.\d{4})$", line.strip())
        if m:
            rows[m.group(1).strip()] = float(m.group(2))
    return rows

def load_cell(vec_rel, neu_rel):
    vp, np_ = REPO / vec_rel, REPO / neu_rel
    sha(vp); sha(np_)
    v = ptload.load(vp)
    em = list(v["emotions"]); raw = np.asarray(v["vectors_raw"], dtype=np.float32)
    n = ptload.load(np_)
    acts = np.asarray(n["acts"], dtype=np.float32)
    return em, raw, acts

def activation_profile(probes_l, acts_l, centered):
    A = acts_l - acts_l.mean(0, keepdims=True) if centered else acts_l
    P = normr(probes_l); Q = normr(A)
    return (P @ Q.T).mean(axis=1)          # (171,) mean cosine over dialogues

def block_split(C):
    """Best single split point maximizing within-block minus cross-block mean corr."""
    L = C.shape[0]; best = None
    iu = np.triu_indices(L, 1)
    for s in range(2, L - 2):
        w = np.concatenate([C[:s, :s][np.triu_indices(s, 1)], C[s:, s:][np.triu_indices(L - s, 1)]])
        x = C[:s, s:].ravel()
        gap = w.mean() - x.mean()
        if best is None or gap > best[1]:
            best = (s, gap, float(w.mean()), float(x.mean()))
    return {"split_layer": best[0], "gap": round(best[1], 4),
            "within_mean": round(best[2], 4), "cross_mean": round(best[3], 4),
            "overall_mean_offdiag": round(float(C[iu].mean()), 4)}

def main():
    t16 = parse_table16()
    if t16 is not None: print(f"Claude Table 16 parsed: {len(t16)} emotions")
    out = {"convention": "cosine(fixed probe, pooled neutral-dialogue activation); probes = raw per-emotion mean-diff vectors, L2-normalized; primary probes IT-cell, uncentered acts"}
    diffs_by_size = {}
    for size, cells in PAIRS.items():
        em_b, raw_b, acts_b = load_cell(*cells["base"][:2])
        em_i, raw_i, acts_i = load_cell(*cells["it"][:2])
        assert em_b == em_i, size
        L = cells["base"][2]
        em = em_b
        res = {}
        # ---- A1 at analysis layer, 2 probe conventions x 2 centering ----
        for pname, praw in (("ITprobes", raw_i), ("BASEprobes", raw_b)):
            for cname, cent in (("uncentered", False), ("centered", True)):
                a_b = activation_profile(praw[:, L], acts_b[:, L], cent)
                a_i = activation_profile(praw[:, L], acts_i[:, L], cent)
                d = a_i - a_b
                key = f"{pname}_{cname}"
                r_struct = float(np.corrcoef(a_b, a_i)[0, 1])
                ov = [k for k, e in enumerate(em) if e.lower() in t16] if t16 is not None else []
                r_claude = float(np.corrcoef(d[ov], [t16[em[k].lower()] for k in ov])[0, 1]) if t16 is not None else None
                res[key] = {"r_structure_base_vs_it": round(r_struct, 4),
                            "r_diff_vs_claude_table16": round(r_claude, 4) if r_claude is not None else "skipped: needs Table 16 of Sofroniew et al. (2026)",
                            "n_overlap_claude": len(ov)}
                if pname == "ITprobes" and cname == "uncentered":
                    order = np.argsort(d)
                    res["top10_up"] = [(em[k], round(float(d[k]), 4)) for k in order[::-1][:10]]
                    res["top10_down"] = [(em[k], round(float(d[k]), 4)) for k in order[:10]]
                    res["diff_full"] = {em[k]: round(float(d[k]), 5) for k in range(len(em))}
                    diffs_by_size[size] = d
                    # spearman vs claude
                    if t16 is not None:
                        dv = d[ov]; cv = np.array([t16[em[k].lower()] for k in ov])
                        rs = np.corrcoef(np.argsort(np.argsort(dv)), np.argsort(np.argsort(cv)))[0, 1]
                        res["spearman_diff_vs_claude"] = round(float(rs), 4)
                    else:
                        res["spearman_diff_vs_claude"] = "skipped: needs Table 16 of Sofroniew et al. (2026)"
        # ---- A2 per-layer profile + two-block ----
        nL = raw_b.shape[1]
        layer_diffs = np.zeros((nL, len(em)), dtype=np.float32)
        for l in range(nL):
            a_b = activation_profile(raw_i[:, l], acts_b[:, l], False)
            a_i = activation_profile(raw_i[:, l], acts_i[:, l], False)
            layer_diffs[l] = a_i - a_b
        mags = np.abs(layer_diffs).mean(axis=1)
        third = nL // 3
        res["fig84_analog"] = {
            "mean_abs_shift_by_third": {"early": round(float(mags[:third].mean()), 5),
                                        "mid": round(float(mags[third:2*third].mean()), 5),
                                        "late": round(float(mags[2*third:].mean()), 5)},
            "per_layer_mean_abs_shift": [round(float(x), 5) for x in mags]}
        C = np.corrcoef(layer_diffs)
        res["two_block"] = block_split(C)
        res["layer_corr_matrix"] = [[round(float(x), 3) for x in row] for row in C]
        out[size] = res
        print(f"{size}: structure r={res['ITprobes_uncentered']['r_structure_base_vs_it']}, "
              f"r vs Claude T16={res['ITprobes_uncentered']['r_diff_vs_claude_table16']}, "
              f"two-block split L{res['two_block']['split_layer']} gap {res['two_block']['gap']}")
    # ---- A3 cross-size consistency ----
    ks = list(diffs_by_size)
    cons = {}
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            cons[f"{ks[i]}_vs_{ks[j]}"] = round(float(np.corrcoef(diffs_by_size[ks[i]], diffs_by_size[ks[j]])[0, 1]), 4)
    out["cross_size_shift_consistency_r"] = cons
    print("cross-size shift consistency:", cons)

    (GG / "geometry_out" / "fixed_probe_tuning_shifts_results.json").write_text(json.dumps(out, indent=1))
    (GG / "geometry_out" / "fixed_probe_tuning_shifts_manifest.json").write_text(json.dumps(
        {"rule": "M.9 — every input hashed at run time", "inputs_sha256": MANIFEST}, indent=1))
    print("results + manifest written")

if __name__ == "__main__":
    main()
