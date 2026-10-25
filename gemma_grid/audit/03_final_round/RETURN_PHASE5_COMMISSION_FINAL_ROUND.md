# Return: Phase-5 COMMISSION — final consolidated adversarial audit — to the GGE1 100T Data Coordinator

**From:** Claude Code, fresh-context adversarial auditor (per `COORDINATOR_PHASE5_COMMISSION_FINAL_ROUND_100T.md`, pre-statement `09eaa4fb…2eef`). Independent of Coordinator / producer / Ledger. Every item recomputed with my **own hand-written code** (own loaders + own PCA/SVD/Procrustes/OLS-HC3/Mann-Whitney/bootstrap/norm-lens) — **no producer helper or reported number trusted for the arithmetic**. Producer scripts were read only to recover the frozen *spec*; none was executed. Read-only throughout; all scratch code in the session scratchpad. Return path: **Coordinator** (not Ledger/Grid directly).

## Bottom line — NINE of NINE SUSTAINED

| # | Item | Artifact · sha256(12) | Verdict | Max Δ (recompute vs artifact) |
|---|---|---|---|---|
| 1 | P-map + floor | `pmap_transport_100T` 10c2836fa34f · `…floor` 09e3c4b8366b | **SUSTAINED** | 0.0 (primary median_rank + both median_cosines) |
| 2 | g66 frequency control | `g66_frequency_control_100T` f19e12feb82b | **SUSTAINED** | 0.0 (all 6 cells) |
| 3 | g54a displacement model | `g54a_displacement_model_100T` f729614a2274 | **SUSTAINED** | 0.0 |
| 4 | WS-P bipolarity cosines | `bipolarity_pairs_100T` 6c1cbcf8b5fe | **SUSTAINED** | 0.0 (first independent recompute) |
| 5 | A1 displacement rank-stability | `analysisA_displacement_bootstrap_100T` 612056765797 | **SUSTAINED** | 0 (incl. all 513 per-emotion CI95) |
| 6 | A2 parts 1–3 | `a2_sr7…` 8a0e3928d028 · `a2_scalepair…` 2af39db3f6f6 | **SUSTAINED** | 0 (incl. CSV sha12) |
| 7 | C dep-aware bootstrap | `analysisC_g32_depaware_100T` 196c59b03ccb | **SUSTAINED** | ≤6e-9 (f64 SVD last-digit) |
| 8 | Spectrum PR + TwoNN | `op1_spectrum_measures_100T_v1` ff483ee9f858 | **SUSTAINED** | 0.0 (3-dp) |
| 9 | B seam/onset (+realization-of-intent) | `analysisB_seam_onset_bootstrap_100T` ca3f961459fa | **SUSTAINED** | integer-exact seam/onset; 9.1e-12 gesdd↔gesvd (no output effect) |

No refutation on any item. No number differed beyond the float precision of the producer's own dtype: items 1/2/3/4/5/6/8 bit-exact (Δ 0.0); item 7 within f64 SVD BLAS last-digit (≤6e-9); item 9 integer-exact on every sampled seam/onset draw. **On the strength of this the evidence base closes — routing to the Ledger for the closing changelog.**

## Integrity anchors — re-verified UNCHANGED (start and end)
- Anchor `emotion_vectors_171_100T.pt` sha256 = `6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b` ✓ (`6de2e293…971b`), re-hashed intact at start and end by the item-9 auditor.
- SR.5 manifest `neutrals_100T/run_phase4/100T_LOCAL_SHA256.txt`: `shasum -c` = **110/110 OK** (every scope artifact byte-stable).
- All 14 audited artifacts match the commission's stated sha256(12) prefixes exactly: pmap 10c2836fa34f, floor 09e3c4b8366b, g66 f19e12feb82b, g54a f729614a2274, bipolarity 6c1cbcf8b5fe, A1 612056765797, a2_sr7 8a0e3928d028, a2_scalepair 2af39db3f6f6, C 196c59b03ccb, spectrum ff483ee9f858, spectrum-manifest 9df67f8b5540, B-v2 ca3f961459fa, B-drawlevel 23b056696470, B-v1headline c6d8546e471a.
- Item-9 code files: frozen estimator `11a12790`, gesvd successor `a9e5e376`, gesvdfix parallel wrapper `61564d28`, v2fullscope collector `1451c557`, unmodified parallel `da1c4f36` — all confirmed.
- Frozen P-map spec `P_MAP_TRANSPORT_SPEC_FROZEN_100T.md` sha256 = `09b4dacdafa6…` = the `spec_sha256` embedded in the pmap artifact ✓.

## Firewalls honored
- **SR.7**: items 2/3/6 confirmed the *computation* only; the registered survival/converse verdicts were **not** re-graded (numbers + grade-side reported, adjudication left to the Ledger).
- **SR.13**: no PA/HA basis comparison entered any recompute; every cleaned path carried "100T" or was the anchor.
- **AX**: `loving` handled only as one of the 171 (a per-emotion row in items 1/4/5/6, never singled out interpretively).

---

## Per-item detail

### 1 — P-map transport + permutation floor — SUSTAINED
Independent 5-fold Procrustes/ridge transport reimplemented from the frozen spec (fold_seed 0, `RandomState(0).permutation(arange(171)%5)`; per-cell analysis-layer slices 2b17/9b28/27b31 of the cleaned 100T vectors).
- **Primary pair `2b-base→27b-base`** (the graded pair): median held-out rank **1.0**, top1 1.0, top5 1.0, **median_cosine 0.9716237565124259** — Δ **0.0**. Ridge secondary median_rank 1.0, **median_cosine 0.9856777830953231** — Δ **0.0**. Cross-BLAS bit-exact (my Accelerate result equals the OpenBLAS artifact to full f64).
- **Floor p**: recomputed add-one arithmetic (1 + count)/(1 + N_perm) = **(1+0)/10001 = 9.999000099990002e-05** = the artifact `p_add_one` exactly. The full 10 000-perm floor is the documented DEFERRED_PROHIBITIVE run (~13.2 h on the Linux host), so I checked the **direction** with an independent 5-perm probe under PERM_SEED=0: permuted top-1 ∈ {0.00585, 0, 0, 0, 0.00585}, **count ≥ obs(1.0) = 0** — and these five values reproduce the floor JSON's `first30_thishost[:5]` exactly. No shuffle approaches perfect top-1 → consistent with `floor_count = 0`.
- **Branch-B confirmed**: `ceiling_branch="B"`, no `ceiling_replicate` key present, `ceiling_note` documents that no independent split-half 171×d replicate matrices exist on disk → graded against the permutation floor alone, **no split-half ceiling constructed** (constructing one would be the forbidden ceiling-construction).

### 2 — g66 frequency control — SUSTAINED (max |Δ| = 0.0, all 6 cells)
Own zipf (wordfreq 3.1.1), own tie-corrected Mann-Whitney, own OLS-HC3 (pinv + hat-matrix sandwich). Used the **`fair_140`** key of `fair_emotions_v1.json`; NRC-VAD-sign split reproduced **50 pos / 87 neg** (= the pre-serialized `g6_6_split`). Consumed the ranks-json **L0-scalar view** (`entry["L0"]`/`entry["final"]`, per its `_source_note`), over the six model cells.
- zipf pos-median 3.855 / neg-median 3.800, Mann-Whitney **U = 2351.0, p = 0.4313145200609503** — exact.
- All 6 cells × {L0, final} × {sign β raw, HC3 p raw, sign β zipf-controlled, HC3 p controlled} + matched-set (n_pairs, L0/final pos/neg medians) reproduce with **Δ = 0.0**.
- Survival criterion (Ledger-graded; computation only): all four 2B/9B L0 zipf-controlled sign β negative with p<.05 (−0.625/−0.661/−0.715/−0.696; p 1.5e-3…3.5e-4), matched-set L0 gap directional 4/4 → asymmetry survives.

### 3 — g54a displacement model — SUSTAINED (max |Δ| = 0.0)
Own OLS-HC3. Pooled outcome = mean post-Procrustes displacement over the three tuning-pair `stageB_displacement__*.csv`; covariates |val|/arousal/purity/log10(L0)/zipf standardized; indicator = self-conscious/moral membership; L0 covariate built as the documented **L0-scalar view** over the 2B/9B cells.
- `indicator_used` (7), `dropped_unrated` (10), **n = 137** — match.
- Full-model β/se/HC3-p/R² for all 7 terms, `indicator_only`, and the three per-pair descriptives reproduce with **Δ = 0.0**.
- **Indicator sign + collapse confirmed**: indicator-only β = **+1.060 (p = 0.115)** already non-significant; full model β = **+0.248 (p = 0.809)** — positive sign, collapses to null under covariates → does not survive p<.05 (adjudication the Ledger's).

### 4 — WS-P descriptive cosines (bipolarity) — SUSTAINED (max |Δ| = 0.0) — FIRST independent recompute
**Coordinator intent confirmed:** the audited object is `bipolarity_pairs_100T.json` (6c1cbcf8b5fe) — the LC-54 bipolarity mapping, not the welfare-scan B(ii) projection (`p_emoscatter_tuningaxis_baseit_100T.json` 6d9426df, which already SUSTAINED in the 08-26 pre-paper batch). I recomputed the bipolarity object.
- Loaded the six 100T vector sets (cleaned `canonical_100T.pt`; 27b-base = anchor), sliced each at its analysis layer, mean-centered, unit-normalized (producer's `+1e-12`), full cosine matrix.
- 8 frozen Plutchik/axis pairs → **2 not-in-vocabulary drops** (`anticipating`, `trusting`) → **6 computed** per cell.
- Every per-pair cosine, both percentiles (`pctl_from_a/b`), and the `_all_pairs_context` (min/p05/median) across all 6 cells reproduce with **Δ = 0.0** (full f64).

### 5 — A1 displacement rank-stability — SUSTAINED (Δ = 0)
Recomputed from the three draw-level `analysisA_ranks__*.csv` (1000 draws × 171).
- `trio_each_top10_frac` (rank ≤ 10), `trio_mean_rank_top_decile_frac` (trio-mean ≤ 17.1), `loving_rank_ci95` (int-truncated 2.5/97.5 pctl), and **all 513 `rank_ci95_per_emotion` intervals** (3 pairs × 171) reproduce exactly.
- B = 1000 confirmed (1000 rows/CSV); seed 20260827 / shared-sel are producer-declared params (corroborated by the a2_scalepair producer's shared-stream reuse; per the commission the check is from the CSVs).

### 6 — A2 parts 1–3 — SUSTAINED
- **Part 2** (`a2_sr7_stability_reproduction_100T`): from the three tuning-pair CSVs, `loving` frac(rank≤85) = **0.007 / 0.854 / 0.093**, frac(≤86), draws-at-86, CI95 [98,153]/[22,118]/[75,158] all match — plus the recorded **CSV sha12 (db2a096cec56 / baedc81f4168 / e7f8b5d1a615) reproduce**.
- **Part 3** (`a2_scalepair_9bit_27bit_100T`): from the scale-pair CSV, frac = **0.761**, frac(≤86) 0.766, at-86 5, CI [10,119], trio rank≤10 (vindictive 0.888 / vengeful 0.889 / spiteful 0.747) — all match. **Knife-edge confirmed: 0.761 < 0.95 → NULL-side.**
- **Part 1 producer-reproduction flag RESOLVES**: the `COORDINATOR_COMPUTE_A_SR7_STABILITY` doc was a Coordinator candidate whose independent producer reproduction was *owed*; the a2_sr7 artifact is that reproduction, and my third-party recompute confirms it byte-for-byte (incl. the ≤85 vs ≤86 convention leaving all grade-sides unchanged).

### 7 — C dependence-aware bootstrap — SUSTAINED (max |Δ| ≈ 6e-9)
Own PCA (column-centered SVD, PC2 = U[:,1]·S[1]; |corr| sign/scale-invariant), own RM-1977 arousal reader (45-emotion overlap), fresh `default_rng(0)` reset per chain, B = 2000.
- SE_registered = √(2/42) exact. base: arousal_r 0.49924/0.66752, Δz 0.25798, **p_registered 0.23712**, SE_boot 0.058568, **p_dep 1.0589e-05**. it: 0.36304/0.57835, Δz 0.27959, **p_registered 0.20011**, SE_boot 0.057707, **p_dep 1.2658e-06**.
- Every value within ≤6e-9 of the artifact (f64 SVD BLAS last-digit); registered p reproduces (0.237/0.200), dep-aware p ~1e-5/1e-6 as recorded.

### 8 — Spectrum (PR + TwoNN) — SUSTAINED (max |Δ| = 0.0 at 3-dp)
Own PR = (Σλ)²/Σλ² from the SVD² spectrum of mean-centered vectors; own TwoNN MLE (ID = N/Σln(r2/r1) on unit-normalized rows); raw = `vectors_raw`, cleaned = `vectors`, at analysis layer.
- All 6 cells × {cleaned, raw} × {participation_ratio, twonn_id, var_pc1_check} reproduce with **Δ = 0.0** (3-dp).
- var_pc1 gate vs stageA reproduces PASS in all cells; the numbers underlying the LC-54 correction-of-record (Phase-5-coverage correction; promotion to G3.1/OP.1) are confirmed. UNREGISTERED-exploratory status noted.

### 9 — B seam/onset (+ realization-of-intent) — SUSTAINED
Recomputed two ways: (i) my own CSV→summary aggregation of the 500-row draw-level CSV, and (ii) a dedicated fresh-context auditor that reimplemented the estimator from spec and recomputed the per-draw seam/onset **from the vectors** (SVD re-clean + norm-lens W_U streaming), on the pod-matched env (numpy 2.0.2 / scipy 1.13.1, single-thread BLAS).
- **Seam**: vectors→per-draw recompute matches the draw-level CSV integer-exact on every sampled draw (2B/9B 40/40 each, 27B 12/12 each); CSV→JSON aggregation reproduces the frozen ci95 exactly ([7,7],[7,7],[8,10],[10,10],[23,24],[23,24]).
- **Onset (27B)**: 27b-it full-neutral validate = layer **23** (median 3709.5), draws 0–4 all 23; 27b-base validate = **23** (median 6282.0); frozen `onset_27B` p50/min/max = 23 for both confirmed. (27b-base *per-draw* onset via the 4.4 GB zipfile-lens stream is intractably slow off-pod → scope-limited to the validate; the validate + 27b-it draws + CSV aggregation establish it.)
- **gesdd↔gesvd**: the successor's only change is the `_svd_thin` gesdd→gesvd fallback in `clean_layer`. On a convergent draw, `max|clean_gesdd − clean_gesvd|` over all 32 layers = **9.10e-12** with identical k → zero effect on integer seam/onset. Reproducing the RNG stream to the 5 named fallback draws: **draw 158 layer 7 raised `np.linalg.LinAlgError` under gesdd even on this macOS/Accelerate host**, and the gesvd fallback gave seam = **23** (both 27B), matching the CSV; draws 262/314/367/447 converged under gesdd here (host-specific non-convergence, as noted in the frozen record).
- **Realization-of-intent (independently code-diffed by me and re-confirmed by the auditor)**: frozen `11a12790` byte-stable; successor `a9e5e376` = *only* the gesvd fallback (no change to seam/onset/gap math, RNG, thresholds, window); gesvdfix parallel `61564d28` = *only* the import redirect; v2 collector `1451c557` reuses `Bmod.seam_of/onset_of/clean_layer/make_wu_getter/ln_of` with `default_rng(20260828)` untouched (sels pre-generated serially, one `integers(0,500,500)` per draw, reassembled by index → bit-identical to serial), adding only the draw-level CSV, the full onset curve over WIN, the lt100/lt1000 sensitivity, and 2B/9B L0-readability, behind a built-in faithfulness assert that curve-`<10k` == `onset_of`.
- **curve == onset_of & L0-readable**: `onset_at(meds,10000)` = the first WIN layer with median<10000 = `onset_of`'s early return by construction; L0-readable (clean L0 → norm-lens → fair-set median own-rank <10000) at full-neutral = 4488.5/5071.0/3690.5/3827.0 (all <10000) → `l0_readable_frac_2B9B = 1.0` reproduced.
- **CSV↔JSON internal consistency** (my aggregation): gap frac_zero 0.896/0.756, gap ci95 [0.0,0.0217], lt1000 onset 27b-it p50=26 ci [26,27], lt1000 gap 27b-it ci [−0.087,−0.0435] frac_zero 0.0, lt100 empty both, l0 frac 1.0×4 — all exactly derivable from the 500-row CSV.

---

## Items for the Coordinator to route / rule
1. **No corrections required** to any artifact — all nine sustained, numbers exact to producer dtype precision (item 7 within f64 SVD last-digit; item 9 integer-exact).
2. **Non-defect provenance notes** (offered for the record; no artifact effect):
   - g66/g54a loaders: the two frozen loader schemas are honored via the documented adapters — g66 keys on `fair_140`; both consume the ranks-json **L0-scalar view** (`entry["L0"]`). Independent recompute confirms the numbers under those adapters. (Consistent with last round's item-4 path-wording fix.)
   - The `per_emotion_ownword_ranks_100T.json` carries a top-level `_source_note` key; iterating only the six `gemma-*` cells (as the recompute does) reproduces the artifact, which contains no `_source_note` "cell". Cosmetic only.
   - The `a2_scalepair_ranks__…csv` (item-6 part-3 draw source) is not itself SR.5-manifest-listed; its parent JSON `a2_scalepair_9bit_27bit_100T.json` (2af39db3f6f6) is manifest-gated, and the two are mutually consistent by construction (single-run producer) — confirmed by the CSV recompute reproducing the JSON.
   - Item-9 27b-base per-draw onset is scope-limited to the full-neutral validate off-pod (4.4 GB zipfile-lens stream); the validate + 27b-it per-draw + CSV aggregation fully establish the onset. Flagged as a coverage note, not a discrepancy.
3. **Accelerate-BLAS note** (for whoever reproduces off-pod on macOS): numpy 2.x on Apple Accelerate emits spurious `divide by zero / overflow / invalid value encountered in matmul` RuntimeWarnings that do **not** affect results (every bit-exact item above ran with them present). Independently, draw-158/layer-7 gesdd non-convergence *does* reproduce on Accelerate, so the gesvd remedy is not merely a Linux-openblas artifact.

Anchor `6de2e293…971b` intact start-to-end. All-SUSTAINED → returns to the **Ledger** for the closing changelog; the evidence base closes. This session did not grade verdicts or edit the ledger.
