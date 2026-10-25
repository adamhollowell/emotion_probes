# Phase 5 — pre-paper audit batch, 100T (prompt, ready to dispatch)

Dispatch to a FRESH Claude Code context with repo access and **no access to any producing
session's reasoning**. This is the pre-paper Phase-5 batch: five owed audits run in one pass so
the whole pre-paper record is checked together. **Your job is to refute each computation.**
Recompute every number with independently-written code from the artifacts; do not trust any
script's printed output. **Nothing in the ledger, the repo, or any artifact may be modified.**

Two basis regimes appear below and the distinction is load-bearing:
- **CLEANED-vector tasks must be 100T all the way down** — any PA or HA cleaned vector reachable
  from a canonical 100T quantity is a finding.
- **RAW / diff-of-means / residual-norm tasks are basis-immune** (byte-identical under PA/HA/100T)
  and are exempt from the 100T-basis check — but their gates and arithmetic are still audited.
Each task states its regime.

## Global inputs / gates (shared)
- Anchor cleaned vectors `data/vectors/emotion_vectors_171_100T.pt` — **must be sha256
  `6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b`**; its `variant` field
  still reads `neutral_HA` and **must stay untouched** (item-10 stale label; 32 producers gate
  on this hash). Read-only; re-hash intact at the end.
- Per-cell `gemma_grid/data/<cell>/vectors/canonical_100T.pt` + `<cell>__neutral_100T.pt`;
  per-cell lens W_U `<cell>__lens_ingredients.pt` (anchor = repo-root `lens_ingredients.pt`).
- Analysis layers: 2b → 17, 9b → 28, 27b → 31. Helpers: `noise_floor.py`, torch-free `ptload.py`.
- PA/HA trees are provenance only — never a source for a 100T cleaned quantity.

---

## Task WS — welfare-paper base↔IT scan, Column 1 (CLEANED 100T)
Artifacts in `gemma_grid/geometry_out_100T/baseit_welfarepaper_scan/`:
`p_antipar_baseit_100T.json` (46486654…), `p_emoscatter_baseit_100T.json` (0f1d4e65…),
`p_emoscatter_tuningaxis_baseit_100T.json` (**6d9426df…**, the both-objects artifact of record;
superseded fe2f4805 is retained in INPUT_MANIFEST), `p_lens_axis_baseit_100T.json` (d2c8f854…).

**SR.7 firewall (binding):** readout (B)'s per-cell valence/arousal shifts ARE the registered
G2.1 quantities; the registered verdict stands (pooled NULL_subfloor/NULL). Your recompute of
(B) must NOT be presented as a re-grade — confirm the numbers, but flag, do not adjudicate.

Refute: **(A)** valence-pole antiparallelism (opposite-valence mean cosine, PC1 pos/neg
separation) per cell, IT vs base. **(B)** own-PC1 valence_r base→IT (2b .711→.764, 9b .715→.777,
27b .720→.772) and arousal_r(PC2) fall; RM-45 overlap. **(B-ii)** the ratified Δ-PC1 = top PC of
the shift matrix Δ_e = v_it(e)−v_base(e); confirm the **pooled-mean axis is degenerate**
(mean-centered vectors → half-split cosine −1.000, quarantined `do-not-interpret`); confirm both
projected objects are valence-ordered on Δ-PC1 — v(e) per arm |r|≈0.80 and Δ_e shift-loading
+0.763/+0.735/+0.810 (naive), with LOO de-circularization moving ≤0.001; arousal is the null
column (~0); Δ-PC1 var-frac 0.353/0.359/0.293. **(C)** lens promoted/suppressed per cell, full
6-cell coverage (per-cell W_U present). Confirm the disclosures are in-file (mild circularity,
var-frac, sign convention) and the **AX firewall** (`loving` in JSON rows only). Any cleaned
path without `100T` is a finding.

## Task OP3 — band-sweep, L42–L45 base-27B (RAW / basis-immune)
Artifact `gemma_grid/geometry_out_100T/op3_band_sweep_analysis.json`; producers in
`gemma_grid/probe_band/` (seeds 0/1 manifests); spec `OP3_BAND_SWEEP_SPEC.md`. Basis-immune
(diff-of-means RAW structures) — **verify that claim** (the reference `vectors_raw` sha is
stamped in `inputs_sha256`; nothing rides on the filename), then it is exempt from the 100T
check. Gates to reproduce: fingerprint gate anchor revision `938270f5`; L31 control vs locked
structure (point prediction 0.9849, seed-0 draw); seed-0 sampler indices reproduce the
`probe_l45/` provenance exactly.
Refute: **(1) ceilings** R_ff(ℓ)=RSA(seed0-ℓ,seed1-ℓ), P(ℓ)=√R_ff — P(45)=0.9462 and
P(31)=0.9854 must reproduce within ±0.005 (internal-consistency gate); **(2) band shape** per
ℓ∈{42,43,44,45}: RSA(fresh-ℓ,locked-ℓ) vs P(ℓ)−0.010, the three-branch GENUINE / vintage-flavored
/ INDETERMINATE grade; is the divergence a cliff at L44–45 or a graded slope; **(3) mechanism**
top-k dim attribution of the L44→L45 drop (k=1,10,100; dim 4140 reported individually);
**(4) IT comparison** same statistics on 27b-it `canonical_100T.pt::vectors_raw`. No registered
row is touched (OP.3 is unregistered characterization).

## Task PC — P-colon (CLEANED 100T; E1 probes both sides)
Artifact `gemma_grid/geometry_out_100T/g4_pcolon_grade.json` (e6c108eb…); producer
`gemma_grid/colon_it/grade_pcolon_100T.py`. Inputs (stamped): `scenario_acts.pt` 94244b23…
(27b-base bare-final), `colon_it/colon_analysis_it_100T.pt` 769a917f… (27b-it, 3 arms),
`emotion_vectors_171_100T.pt` 6de2e293… (E1 100T probes, both sides). Convention (frozen fig2 §1):
bare-prompt final token, 12 scenarios, L31 resid_post f32, unit-normalized dot on E1 100T probes,
one probe set both models.
Refute: the primary-triad (regretful/remorseful/guilty) IT−base diff / SEM / z at L31 (all
IT_lower); the aggregate triad z = −1.59 (one-sided p ≈ 0.056); the band means (mid IT_higher,
late strongly IT_lower); the rank-overlap vs Appendix T.1 (2/10; moral triad ranks 34/55/57).
**Grade of record = NULL** (predicted IT_lower sign uniform → NOT falsified; effect within the
across-scenario noise at the conventional bound → not confirmed). Confirm the numbers and
whether the NULL determination is supported; flag, do not re-adjudicate. AX firewall: `loving`
out of any prose. The §4.6 raw-colon arm is the demoted tertiary, excluded from the grade.

## Task FI — Tier-1 E1 fold-ins (mixed regime)
Artifacts in `gemma_grid/geometry_out_100T/`: `p32_resid_norm_growth_family_100T.json`
(43f67346…, **RAW / basis-exempt**), `p34b_arousal_pc23_family_100T.json` (2bd3c1fc…, **CLEANED
100T**), `p37_velocity_family_100T.json` (cd33539e…, **CLEANED 100T**). Producers
`scripts/phase4_p32_resid_norm_100T.py` / `_p34b_arousal_pc23_100T.py` / `_p37_velocity_100T.py`.
Refute: **P3.2** per-cell residual-norm growth (L0→final ratio, largest-drop transition) — confirm
it loads only `neutral_100T` activations (no cleaned/emotion vector) and **disclose the
per-dialogue-mean PROXY** (E1's per-token files were pod-lost; this recovers growth-ratio +
final-layer transition only). **P3.4b** arousal PC2/PC3: `arousal_pc2_at_L` must equal Stage-A
`arousal_r` per cell (cross-consistency), R(PC2,PC3), fixed global PC2 sign. **P3.7** dimensional
velocity (adjacent-layer cosine); **E2 firewall — do NOT interpret loving's velocity minimum**
(output-space artifact). `assert "100T"` on every cleaned path (P3.4b/P3.7); anchor remap intact.

## Task G66 — valence-asymmetry (CLEANED 100T; norm-lens own-ranks)
Artifact `gemma_grid/geometry_out_100T/valence_asymmetry_onset.json` (994772cd…). Recipe: norm
lens `logits = ((v/rms(v))·(1+ln_w)) @ W_U`, eps 1e-6, cleaned 100T vectors, all layers;
`rank_e = 1 + #(logits_row > own_logit)`; per-cell fair set = 140. Valence split from
`geometry_out_100T/fair_emotions_v1.json → g6_6_split` (positive_50 / negative_87).
Refute: per-cell pos/neg `{onset, L0, final}` (onset = first layer group-median own-rank < 10000);
the combined pos+neg median onset must reproduce each cell's `lens_summary.per_layer_median_own_rank`
onset (2b/9b → 0, 27b → 23); fair count 140 at every cell; anchor sha-gated. Surprise-gate shape
(for the record, not a re-grade): positive more readable at L0 for 2B/9B (pos L0 < neg L0);
27B positive onset 1–2 layers before negative; negative reaches lower final ranks at all six cells.

---

## Report + return path
Verdict per task (**SUSTAINED / REFUTED / UPHELD-WITH-CAVEAT**), every recomputed number that
differs (with the artifact and the delta), and required corrections if any. Confirm the anchor
sha is untouched at the end. **Return this report to the GGE1 100T Data Coordinator** — NOT to
the Ledger or Grid directly. The Coordinator is the verification step: it re-checks the report,
then routes the verified result to the Grid/ledger session for integration. Producing sessions
do not grade this report.
