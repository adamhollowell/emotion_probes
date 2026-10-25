"""GGE1 draft v2 — Tables 1-3 (100T basis).

T1 (§2.1): cross-cell geometry summary. Values from stageA CSV; bootstrap
   SDs from per-cell bootstrap_B1000.npz (ratios -> variance shares,
   rs[:,0]/rs[:,1] -> valence/arousal, mapping gated below); self-ARI SD
   from the CSV seed sweep. Cohesion ships WITHOUT an uncertainty
   (coordinator ruling 3) and the header note says so.
T2 (§5.1): registered tuning battery. Pooled/floor/p from stageC_grading
   (tuning family); per-size triples transcribed from Ledger G2.1's
   per-size serialization block (source of record, ruling 5), gated here:
   each triple's mean must reproduce the pooled artifact delta.
   The binding neighbor-cosine footnote travels verbatim.
T3 (§6.2): verdict-level norm robustness. Labels read from Ledger
   Appendix R rows R.1-R.6 (ruling 2); per-arm numbers stay in Appendix E.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gge1_style as G

OUT = G.FIG_DIR
gates = {}

# ------------------------------------------------------------------ T1
# BUILD RULE (MEMO_TO_FIGURES_TABLE1_ROUNDING_RULE_100T.md, Ledger ruling
# 2026-09-06): Table 1 values come from the FULL-PRECISION per-cell
# stageA_summary.json artifacts (source of record), never from the 4dp
# CSV aggregate — re-rounding the CSV's rounded values produced two
# double-rounding artifacts (2B tuned cohesion .748→.749, self-ARI SD
# .074→.075). The CSV is kept only as a cross-check gate.
csv_rows = {r["cell"]: r for r in csv.DictReader(
    open(os.path.join(G.T100, "stageA_cross_cell_table.csv")))}
vals, sd = {}, {}
for c in G.CELLS:
    s = json.load(open(os.path.join(G.T100, c, "stageA_summary.json")))
    det = s["deterministic"]
    vals[c] = dict(
        var_pc1=det["var_pc1"], var_pc2=det["var_pc2"], var_pc3=det["var_pc3"],
        valence_r=det["valence_r"], arousal_r=det["arousal_r"],
        cohesion=s["cohesion"]["mean_purity_top10"],
        self_ari_mean=s["kmeans_seed_sweep"]["self_ari_mean"],
        self_ari_std=s["kmeans_seed_sweep"]["self_ari_std"],
    )
    # cross-check gate: full-precision values must agree with the CSV
    # aggregate at the CSV's own per-column precision (variance shares are
    # 2dp in the CSV; correlations/cohesion/ARI are 4dp) — same objects,
    # no drift, without ever using the CSV's rounded digits in the table
    for k, tol in (("var_pc1", 5.1e-3), ("var_pc2", 5.1e-3),
                   ("var_pc3", 5.1e-3), ("valence_r", 5.1e-5),
                   ("arousal_r", 5.1e-5), ("cohesion", 5.1e-5),
                   ("self_ari_mean", 5.1e-5), ("self_ari_std", 5.1e-5)):
        assert abs(vals[c][k] - float(csv_rows[c][k])) < tol, (c, k)
    z = np.load(os.path.join(G.T100, c, "bootstrap_B1000.npz"))
    # gate the rs column mapping: bootstrap mean must sit near the point est.
    mv, ma = float(z["rs"][:, 0].mean()), float(z["rs"][:, 1].mean())
    assert abs(mv - vals[c]["valence_r"]) < 0.02, (c, "rs0", mv)
    assert abs(ma - vals[c]["arousal_r"]) < 0.03, (c, "rs1", ma)
    sd[c] = dict(
        v1=float(np.std(z["ratios"][:, 0]) * 100),
        v2=float(np.std(z["ratios"][:, 1]) * 100),
        v3=float(np.std(z["ratios"][:, 2]) * 100),
        val=float(np.std(z["rs"][:, 0])),
        aro=float(np.std(z["rs"][:, 1])),
    )
gates["T1_rs_mapping"] = "PASS (bootstrap means match point estimates)"
gates["T1_fullprecision_vs_csv"] = ("PASS (per-cell summaries agree with the "
                                    "CSV aggregate at its per-column precision)")

# "the anchor" is a captions-only convention (REVIEW_100T_ROUND1 fix):
# the 27B base row is not marked in the table body; the caption carries it.
# "tuned", not "IT": tables match the paper's term (2026-10-20, as gge1_style.cell_label)
NAME = {"gemma-2-2b-base": "2B base", "gemma-2-2b-it": "2B tuned",
        "gemma-2-9b-base": "9B base", "gemma-2-9b-it": "9B tuned",
        "gemma-2-27b-base": "27B base", "gemma-2-27b-it": "27B tuned"}

t1 = ["# Table 1 (§2.1) — Cross-cell geometry summary (100T-canonical)",
      "",
      "<!-- Built by build_tables.py per ASSET_BINDING_MAP_100T.md."
      " Uncertainties are bootstrap SDs (B=1000, story resample);"
      " self-ARI ± is the 20-seed clustering sweep SD. Cohesion ships"
      " without an uncertainty: no bootstrap object exists, and the"
      " pair_cos derivation is REFUSED on object identity (coordinator"
      " ruling 3 v2, Ledger 2026-09-06 — pair_cos is the anchor's nine"
      " neighbor-pair cosines, a different statistic). Caption must state"
      " that no cohesion uncertainty was computed. -->", "",
      "| cell | PC1 var (%) | PC2 var (%) | PC3 var (%) | valence r | arousal r | cohesion | self-ARI |",
      "|---|---|---|---|---|---|---|---|"]
for c in G.CELLS:
    r = vals[c]  # full-precision per-cell values; rounded exactly ONCE here
    s = sd[c]
    t1.append(
        f"| {NAME[c]} | {r['var_pc1']:.2f} ± {s['v1']:.2f} "
        f"| {r['var_pc2']:.2f} ± {s['v2']:.2f} "
        f"| {r['var_pc3']:.2f} ± {s['v3']:.2f} "
        f"| .{round(r['valence_r']*1000):03d} ± .{round(s['val']*1000):03d} "
        f"| .{round(r['arousal_r']*1000):03d} ± .{round(s['aro']*1000):03d} "
        f"| .{round(r['cohesion']*1000):03d} "
        f"| .{round(r['self_ari_mean']*1000):03d} ± .{round(r['self_ari_std']*1000):03d} |")
open(os.path.join(OUT, "gge1_v3_table1_crosscell.md"), "w").write("\n".join(t1) + "\n")

# ------------------------------------------------------------------ T2
grading = json.load(open(os.path.join(G.T100, "stageC_grading.json")))
pooled = {t["signal"]: t for t in grading["tests"]
          if t["family"] == "tuning" and "POOLED" in str(t.get("contrast", ""))}

# Ledger G2.1 per-size serialization block (source of record, ruling 5).
# Transcribed 2026-09-05; gate below reproduces each pooled delta.
G21 = {  # signal -> (2b, 9b, 27b)
    "valence_r": (+0.0535, +0.0620, +0.0520),
    "arousal_r": (-0.1362, -0.0233, -0.0892),
    "var_pc1": (+0.2706, -0.6908, +0.5362),
    "var_pc2": (-0.0865, -0.3576, -0.5258),
    "var_pc3": (-0.2049, +0.3829, +0.2856),
    "offdiag_pc1_arousal": (+0.0316, +0.0334, +0.0579),
    "offdiag_pc2_pleasure": (-0.0807, -0.0364, -0.1471),
    "neighbor_cos(9 anchor pairs)": (+0.0012, +0.0001, +0.0059),
    "self_ARI(k=10)": (-0.0058, -0.0399, -0.0041),
}
for sig, triple in G21.items():
    got = float(np.mean(triple))
    want = pooled[sig]["delta"]
    assert abs(got - want) < 2.5e-4, (sig, got, want)
gates["T2_mean_reproduces_pooled"] = "PASS (all 9 rows, tol 2.5e-4)"

LABEL = {
    "valence_r": "valence r (Δ, tuned − base)",
    "arousal_r": "arousal r (Δ)",
    "var_pc1": "PC1 variance share (Δ pp)",
    "var_pc2": "PC2 variance share (Δ pp)",
    "var_pc3": "PC3 variance share (Δ pp)",
    "offdiag_pc1_arousal": "off-diagonal PC1·arousal (Δ)",
    "offdiag_pc2_pleasure": "off-diagonal PC2·pleasure (Δ)",
    "neighbor_cos(9 anchor pairs)": "neighbor cosine, 9 anchor pairs (Δ)",
    "self_ARI(k=10)": "partition self-agreement (Δ ARI)",
}


def fp(p):
    if p < 1e-6:
        return f"{p:.1e}"
    return f".{round(p*1000):03d}" if p < 1 else "1.0"


t2 = ["# Table 2 (§5.1) — The registered tuning battery (100T-canonical)",
      "",
      "<!-- Built by build_tables.py. Per-size deltas transcribed from"
      " Ledger G2.1 'Per-size serialization' (closed 2026-09-05; source of"
      " record per coordinator ruling 5), gated here against the pooled"
      " stageC deltas. Pooled/floor/p read from stageC_grading.json."
      " Pairwise Fisher p for the two correlation rows are in G2.1. -->",
      "",
      "| registered measure | Δ 2B | Δ 9B | Δ 27B | pooled Δ | floor | p |",
      "|---|---|---|---|---|---|---|"]
for sig, triple in G21.items():
    t = pooled[sig]
    fl = "—" if t["floor"] is None else (f"{t['floor']:.2f}" if t["floor"] < 1 else f"{t['floor']:.1f} pp")
    t2.append(f"| {LABEL[sig]} | {triple[0]:+.4f} | {triple[1]:+.4f} | "
              f"{triple[2]:+.4f} | {t['delta']:+.4f} | {fl} | {fp(t['p'])} |")
t2 += ["",
       "**Binding footnote (travels with this table, verbatim).** The"
       " neighbor-cosine per-size values come from the producer's registered"
       " rule, a bootstrap-mean cosine at the anchor's nine official top-1"
       " pairs. The per-cell point estimates in the Stage A artifacts are a"
       " different, non-graded object and must never be substituted."]
open(os.path.join(OUT, "gge1_v3_table2_tuning_battery.md"), "w").write("\n".join(t2) + "\n")

# ------------------------------------------------------------------ T3
# Verdict labels read from Ledger FINAL Appendix R rows R.1-R.6
# (coordinator ruling 2). Transcribed 2026-09-05.
t3 = ["# Table 3 (§6.2) — Verdict-level norm robustness (100T-canonical)",
      "",
      "<!-- Built by build_tables.py. Verdict labels transcribed from"
      " Ledger Appendix R rows R.1-R.6 (coordinator ruling 2); ledger IDs"
      " map to rows in order (row 1 = R.1, ... row 6 = R.6) and are kept"
      " out of the publication-facing text per REVIEW_100T_ROUND1. Per-arm"
      " numbers live in Appendix E and are NOT duplicated here. Caption"
      " obligations: define the five settings and the natural/matched-45"
      " variants; carry the operative asymmetry finding (ledger R.4) and"
      " the note that the scale-lead floor clearance is norm-arm-sensitive"
      " (registered follow-up design datum). -->", "",
      "| analysis (registered verdict) | R&M [registered] | NRC-VAD [natural] | NRC-VAD [matched-45] | Warriner [natural] | Warriner [matched-45] |",
      "|---|---|---|---|---|---|",
      "| tuning main effect, valence / arousal (null / null) | null / null | null (sub-floor at n = 161) / null | null / null | null / null | null / null |",
      "| arousal-scale lead (lead, not confirmed) | + direction, both chains | + direction, both chains | + direction, both chains | + direction, both chains | + direction, both chains |",
      "| arousal interaction, tuning × scale (null) | null, + direction | null, + direction | null, + direction | null, + direction | null, + direction |",
      "| embedding asymmetry: valence present, arousal absent (confirmed) | holds | holds | holds | holds | holds |",
      "| arousal depth-gap grows with scale (direction observed; null at the registered bar) | + both chains | + both chains | + both chains | + both chains | + both chains |",
      "| negative control: arousal off PC2 on rated-subset refit (0 of 6) | 0 of 6 | 5 of 6† | 0 of 6 | 2 of 6† | 2 of 6† |",
      "",
      "† Integer survive-counts are bootstrap-unstable (fragility"
      " disclosure travels as a footnote); the directional dose-response —"
      " arousal leaves PC2 as the rated subset shrinks — is the claim, and"
      " the registered control is stably 0 of 6."]
open(os.path.join(OUT, "gge1_v3_table3_norm_robustness.md"), "w").write("\n".join(t3) + "\n")

json.dump({"gates": gates,
           "sources": {
               "table1": ["geometry_out/stageA_cross_cell_table.csv",
                          "geometry_out/<cell>/bootstrap_B1000.npz"],
               "table2": ["geometry_out/stageC_grading.json",
                          "Ledger FINAL — G2.1 per-size serialization block"],
               "table3": ["Ledger FINAL — Appendix R rows R.1–R.6",
                          "norm_robustness_full_results.json (6b51146e, numbers stay in Appendix E)"]}},
          open(os.path.join(OUT, "gge1_v3_tables_data.json"), "w"), indent=1)
print("gates:", gates)
