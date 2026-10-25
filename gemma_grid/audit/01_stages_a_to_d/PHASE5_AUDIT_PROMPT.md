# Phase 5 — Stage A–D (+ CPU tail) adversarial audit, 100T (prompt, ready to dispatch)

Dispatch to a FRESH Claude Code context with repo access and **no access to any
producing session's reasoning**. Gate: the 100T canonical flip's process-integrity
condition (`coordination_100T/GGE1_100T_PRECOMMITMENTS.md` item 1) requires Phase-5
to sustain the computation. Unpaused at LC-20. Scope includes the c3grade incident
(LC-22/LC-24) and the relabel/hash anomaly (item 10 / LC-19–20).

---

You are auditing the **100T Stage A–D core + CPU tail** in
`gemma_grid/geometry_out_100T/`, produced by
`gemma_grid/scripts/run_grid_phase4_100T.sh`. **Your job is to refute it.**
Recompute every number with independently written code from the artifacts; do not
trust any script's output without reproducing it. **Nothing in the ledger, the repo,
or any artifact may be modified.** This is a CLEANED-vector audit: everything must be
**100T all the way down** — any PA or HA cleaned vector reachable by a canonical 100T
quantity is a finding. (RAW vectors are basis-invariant and exempt.)

## Inputs (100T-canonical)
- Anchor cleaned vectors `data/vectors/emotion_vectors_171_100T.pt` — **must be
  sha256 `6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b`**; its
  `variant` field still reads `neutral_HA` and **MUST stay untouched** (item 10).
- Per-cell `gemma_grid/data/<cell>/vectors/canonical_100T.pt` +
  `<cell>__cleanup_basis_100T.pt`; neutral bases `neutral_100T.pt` /
  `<cell>__neutral_100T.pt`.
- Outputs in `gemma_grid/geometry_out_100T/`: `stageA_cross_cell_table.csv`,
  per-cell `stageA_summary.json`, `stageB_contrasts.json` + 9
  `stageB_displacement__*.csv`, per-cell `bootstrap_B1000.npz`,
  `stageC_grading.json`/`.md`, `c3_summary.json`, per-cell `c3_embedding.json` +
  `c3_vectors.pt`, `c3_grading.json`, `lens_cross_cell.json` + per-cell
  `lens_summary.json`, `extracts_summary.json`, `layerstab_ci_100T.json`,
  `seam_gap_metric_100T_v1.json`, `op1_spectrum_measures_100T_v1.json`, App R/T
  results, `zhang_parallels_100T/`.
- Producing scripts: `noise_floor.py`, `phase4_*_100T.py`, `_c3_100T_savevecs.py`,
  `phase4_stageA_crosstable.py`, `phase4_lens_crosstable.py`,
  `phase4_lens_preflight.py`, the orchestrator, `ptload.py`.
- SR.5: per-step manifests, `neutrals_100T/run_phase4/100T_LOCAL_SHA256.txt`,
  `preflight_sha256.txt`.
- PA/HA trees (`geometry_out/`, `geometry_out_HA/`) are retained as provenance
  ONLY — **never a source for a 100T cleaned quantity**.

## Tasks

**(a) Anchor identity + gate.** Confirm `emotion_vectors_171_100T.pt` sha256 ==
`6de2e293…`. Recompute the anchor re-clean cross-validation independently (rel ≤
1e-6; producer reported 7.11e-07); confirm k@L31 = 28.

**(b) Rebase / cleanup basis.** For each of the six cells, recompute the per-layer
SVD cleanup basis (k = smallest cumulative-variance ≥ 0.50) from the cell's
`neutral_100T` activations and re-clean the raw vectors; confirm bit-repro of
`canonical_100T.pt` (anchor: `emotion_vectors_171_100T.pt`). Any cell whose cleaned
vectors match the PA or HA basis is a canonicality finding.

**(c) Stage A.** Recompute var_pc1/2/3, valence_r, arousal_r, loving_pc1/2,
cohesion (mean_purity_top10, mutual_top1_pairs), self-ARI (20-seed) per cell from
cleaned vectors + `data/ratings/russell_mehrabian_1977.csv`; confirm the 13-column
`stageA_cross_cell_table.csv` to table precision. Verify the SR.4 aggregator's
column→key map (`phase4_stageA_crosstable.py`).

**(d) Stage B + displacement.** Recompute the cross-cell contrasts and the 9
displacement CSVs.

**(e) Bootstrap gate.** Reproduce the B=1000 bootstrap reproduction gate per cell
(rel ≤ 1e-4); confirm the 100T means are recomputed locally from raw per-story
slices with **no PA-scratch reuse** (`phase4_bootstrap100T_all.py`).

**(f) Stage C grading.** Recompute pooled [IT−base] deltas + floors + permutation
p's (G2.1/G2.2), var_pc1 endpoint contrasts both chains (G3.1) with Holm, arousal_r
scale contrasts (G3.2), DiD (G4.1), refit-45 best-PC (G1.2). Verify the
verdict-flip column against the ledger's current verdicts — **flag** flips; do not
adjudicate.

**(g) G1.3 embedding baseline + THE c3grade incident.** Recompute the c3 embedding
signals (valence present / arousal absent at embedding) from the 100T-cleaned c3
embeddings. Then verify the incident remediation (LC-22): (i) the quarantined bad
artifact `geometry_out_100T/_quarantine/c3_grading.json.BAD_pa_cleaned_redirect`
stayed quarantined and is read by nothing; (ii) `c3_grading.json` (sha256
`4cc8584e5526c835a07080a761f03264c3ffb38f80bb0aeec99a0d3fabbaaa12`) was regenerated
from the step-8b vectors; (iii) each `geometry_out_100T/<cell>/c3_vectors.pt`
`["vectors"]` **DIFFERS** from the PA tree's and `["vectors_raw"]` **MATCHES** PA
(basis-invariant control). Reference 100T `vectors` sha256 per cell:
- gemma-2-2b-base `93554af8580bca4b0ef6dd4c8c3045240259f833ee5c074c21746f46342bd50f`
- gemma-2-2b-it `b4bd25e41489289d9cbbeef2b4e51f298822d10180dd32dfa43e0b34723edb89`
- gemma-2-9b-base `3f83d141a47a7e1c23cbf4d83359b704ae71a88d7c8910265db9018c68024863`
- gemma-2-9b-it `41020e8a1e5ef62287764b3ea3d20dd35d6d80211cb5e9d3b5637edc19b0175d`
- gemma-2-27b-base `5303cc8773928d994122e4f3b3c4b8e9079ab3b8d554fa0cb9fc7f31232244a7`
- gemma-2-27b-it `1f69b58826412e7a8e9995d1e2667f8b48e6bc8fbd2aaff390e83cda611e786b`

Any PA-cleaned vector feeding c3grade is a canonicality violation.

**(h) SR.5 completeness.** Confirm every input each script reads is hashed in a
manifest, hashes verify on disk today, and no input escapes the manifest (imports,
hardcoded data). Confirm the SR.5 tripwire fired (100T `stageA_cross_cell_table.csv`
differs from the HA table, as it must).

**(i) CPU tail (spot-verify).** Lens: anchor 27b-base onset L23 and median-rank
collapse ~23k→4.8k→3k, reproduced from the tokenizer-free preflight assets (vocab +
emotion_token_ids + W_U memmap recovered from `data/vectors/lens_ingredients.pt` —
verify they reconstruct byte-exactly and that the anchor gate holds). Layerstab; op1
(var_pc1 cross-check vs stageA); seam_gap (anchor 27b-base seam L23 / onset L23
hard-pass; the `ledger_reference_mismatch` record for 27b-it seam 23-vs-24 under the
anchor-hard / non-anchor-warn convention); Zhang z1/z2/z3.

**(j) Item 10 — relabel/hash anomaly (record, do NOT act).** The working anchor
`emotion_vectors_171_100T.pt` MUST remain sha256 `6de2e293` with `variant:
neutral_HA` untouched (32 `repo/github_repo` producers + `LOCK_RECORD.json` + S1/S2/S3
manifests gate on it). The stale-label fix is **export-only** (LC-20). Confirm the
working anchor is untouched; record the label anomaly; the export copies
(`repo/github_repo/…`, `repo/hf_model_repo/…`) are out of Stage A–D scope.

## Report
Verdict per task (**SUSTAINED / REFUTED / UPHELD-WITH-CAVEAT**), every recomputed
number that differs (with the artifact and the delta), and required corrections if
any. The producing sessions do not grade this report; it returns to the GGE1 100T
grid/ledger session for integration.
