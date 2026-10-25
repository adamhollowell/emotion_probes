# Gemma Grid — Pre-registration (GATE 1)

Locked BEFORE any GPU extraction (PLAN §2.3). Nothing here may be changed
after extraction begins without logging the change and the reason. Items fill
in as Adam approves them; a signal/prediction is not "registered" until it
appears here with a value.

Status: **ASSEMBLED — (a)(b)(c)(d)(e) ALL LOCKED, no open sub-decisions.** Ready
for Adam's formal GATE 1 sign-off (the gate itself). Anchor bootstrap resolved:
full L31-truncated rerun as a 6th Phase-3 job. Nothing touches a GPU until GATE 1
is signed off.

---

## (a) Cells and depth-fraction rule — ✅ LOCKED 2026-07-20

Six cells, Gemma-2 family, corpus + tokenizer held fixed:

| Cell | model_id | tuning | n_layers | analysis layer |
|---|---|---|---|---|
| gemma-2-2b base   | google/gemma-2-2b     | base | 26 | **L17** |
| gemma-2-2b-it     | google/gemma-2-2b-it  | it   | 26 | **L17** |
| gemma-2-9b base   | google/gemma-2-9b     | base | 42 | **L28** |
| gemma-2-9b-it     | google/gemma-2-9b-it  | it   | 42 | **L28** |
| gemma-2-27b base  | google/gemma-2-27b    | base | 46 | **L31** (anchor, done) |
| gemma-2-27b-it    | google/gemma-2-27b-it | it   | 46 | **L31** |

**Depth-fraction rule (LOCKED):** strict 2/3 — `analysis_layer = round(2/3 * n_layers)`,
read from each model's own config at load and asserted. Uniform across the grid.

- Rejected alternative: `31/46 = 0.6739`. It is a back-computed artifact of the
  anchor's rounding, not an independently chosen fraction; it agrees with strict
  2/3 everywhere except 2B, where it would pick L18 — the layer *farther* from
  two-thirds depth (2/3·26 = 17.33 → L17).
- **2B sensitivity check (PRE-REGISTERED):** L17 is the headline layer for both
  2B cells. L18 (and a small ±band) is reported as a sensitivity check. This is
  free — extraction captures all layers. Reading: if 2B geometry is stable
  across L17/L18, the choice was immaterial; if knife-edge, that instability is
  itself a reported finding. Neither outcome is decided post hoc.

## (b) Resolved analysis layers — ✅ see table above

## (c) Signal set — ✅ LOCKED 2026-07-20 (C1–C4 confirmed for all six cells; loving PC1/PC2 added as #9b; C5 include/defer locked)

Three groups. Section refs are to E1 Draft v7. Every signal in C1–C3 is
computed from a cell's own all-layer vector artifact with **no GPU forward
pass**; the between-cell set C4 is the grid's actual output.

### C1 — Per-cell geometry @ the analysis layer (free from the vectors)
| # | Signal | Def / anchor value at 27b-base | § |
|---|---|---|---|
| 1 | PC1/PC2/PC3 variance % | 26.20 / 13.32 / 11.86 | 5.1 |
| 2 | Valence corr (PC1↔pleasure) | r = 0.709 on 45-overlap (R&M 1977) | 5.2 |
| 3 | Arousal corr (PC2↔arousal) | r = 0.622 | 5.2 |
| 4 | Discriminant off-diagonals + permutation p | PC1↔arousal −0.14, PC2↔pleasure −0.09; 5,000-iter label-shuffle, p≈0.0002 both | 5.2 |
| 5 | **Arousal scale-refit test (NEGATIVE control)** | refit PCA on the 45-overlap ALONE; arousal rotates OFF PC2 → axis is scope-conditioned, not basis-independent | 5.2 / 9.3 |
| 6 | Vector norms | norm↔valence r≈+0.38, ↔signed-arousal null, ↔\|arousal\| r≈+0.29; norm CV across depth | 5.4 |
| 7 | Neighbor cosines | afraid→scared 0.987, joyful→happy 0.977, etc. | 5.1 / 6.6 |
| 8 | k=10 families + cross-seed self-ARI | self-ARI 0.684 (min 0.426) | 4.1 / 4.2 |
| 9 | Cluster cohesion / boundary | 10-NN in-cluster fraction (mean 0.735, 47 mutual-NN pairs); per-region spread; least-stable boundary emotions | 4.3 |
| 9b | **`loving` PC1/PC2 position** | anchor: PC1 +381.11, PC2 −223.66 (post sign-fix) — tracks whether tuning/scale moves the warm-affiliative boundary case | 5.3 |

### C2 — Per-cell depth validation (confirms the derived ⅔ layer sits in a stable band — the per-cell analog of §6.5's "convergent case for L31")
| # | Signal | Def | § |
|---|---|---|---|
| 10 | Cross-layer RSA to analysis layer | cosine of each layer's 171×171 cosine matrix vs the analysis layer's | 6.1 |
| 11 | Cross-layer neighbor Jaccard | top-5 retention + top-10 overlap vs analysis-layer neighborhood, all layers | 6.5 |

### C3 — Per-cell embedding baseline (NO forward pass — reads the static embedding table directly, per §6.6 "without loading the full model")
| # | Signal | Def | § |
|---|---|---|---|
| 12 | Embedding-layer baseline vs analysis layer | bag-of-embeddings null (mean from 50th token, same diff-of-means + cleanup, cleanup basis refit on neutral embeddings): valence/arousal corr, PC variance, RSA, ARI vs locked partition, neighbor cosines. Anchor: valence already 0.509 at embedding, arousal only 0.108 → depth *builds* arousal, merely *sharpens* valence | 6.6 |

### C4 — BETWEEN-CELL comparison signals (the grid's actual output)
The grid's findings are contrasts, not per-cell numbers. Comparison structure:
base↔IT within each size (3 pairs) and size↔size within each tuning (2 chains),
all against the fixed 27b-base anchor.

| # | Signal | Def | Localizes |
|---|---|---|---|
| i   | Cross-cell RSA | cosine of two cells' 171×171 cosine matrices (§6.1 method, across cells) | global geometry drift |
| ii  | Procrustes axis-alignment | orthogonal Procrustes on per-emotion PC-coordinate matrices; rotation + residual | whether valence/arousal are the SAME axes across cells |
| iii | Cross-cell ARI | ARI between two cells' k=10 partitions of the 171 emotions | cluster reorganization |
| iv  | Per-emotion displacement | post-Procrustes per-emotion distance between cells | WHICH emotions move (e.g. does `loving` move most under IT) |

**Registered constraint (dimension safety):** cells differ in d_model
(2304 / 3584 / 4608). Between-cell signals therefore operate ONLY on
dimension-free objects — the 171×171 emotion-similarity matrices (RSA, ARI)
or aligned per-emotion PC-coordinate matrices (Procrustes, displacement).
Raw activation-space vectors of different width are NEVER directly compared.

### C5 — Include / defer decisions — ✅ LOCKED 2026-07-20
| Signal | Decision | Cost | Note |
|---|---|---|---|
| **Logit-lens readout** (§7.1 unembed table, §7.2 raw-weights (1+w), §7.3 onset depth) | ✅ **INCLUDE (cheap)** | near-zero: capture at extraction | Each cell's extraction MUST persist `W_U` + the final RMSNorm weight (see "Persist per cell" below); readout is then an offline laptop analysis. §7.2 is the (1+w) application (P4.2/P4.3) that compat=False requires. |
| **Held-out-text activation** (§5.5 co-activation RSA on ~940k tokens; §8 maxact + magnitude calibration) | ✅ **DEFER** (first pass) | — | Deferred to a follow-up. Revisit only if a base↔IT geometry difference needs external-text corroboration. |

**Persist per cell (extraction-run output contract, set by C5 + (e) bootstrap):**
1. the cleaned + raw emotion vectors, ALL layers (existing `canonical.pt`);
2. `W_U` (unembedding) and the final-layer RMSNorm weight — for the offline
   logit lens. Near-zero marginal cost while the model is loaded; skipping it
   would force a second GPU model-load per cell later.
3. **per-story pooled activations at the analysis layer (± sensitivity band)**
   for all 171 emotions + the 500 neutral dialogues (~4 GB/cell) — the
   bootstrap-friendly intermediate for the (e) story-level bootstrap. Full
   per-emotion all-layer files are ~520 GB across the 5 new cells (infeasible
   to retain); the layer-band slice keeps the bootstrap laptop-only at Phase 4.
   The anchor's per-story activations are NOT retained locally
   (`data/activations/` empty), so the anchor gets a dedicated **L31-truncated
   full rerun** as a 6th Phase-3 job (see (e)) producing this same per-story L31
   intermediate — not a subsample.
→ The `--cell` driver wiring (Phase 0) must support (2)+(3) and TWO extraction
   modes: (A) new cells = full-depth, persist all-layer vectors + unembed +
   per-story analysis-layer slice; (B) anchor rerun = `stop_at_layer=32`,
   per-story L31 slice + 500 neutrals at L31 only (the all-layer anchor already
   exists as locked `vectors_171.pt`; this rerun is bootstrap-only).

## (d) Directional predictions — ✅ LOCKED 2026-07-20 (edited per Adam)

Anti-storytelling guard: locked BEFORE the five new cells run. Directions only
(sign of the effect), not point values. Confidence: **H** strong hypothesis +
prior evidence · **M** plausible · **expl** EXPLORATORY (registered but OUTSIDE
the Holm family — not counted as a confirmatory test). Adam owns these; further
edits are logged in the decision log.

Reference (27B-base anchor, L31): var 26.2/13.3/11.9 · valence r 0.709 ·
arousal r 0.622 · off-diag −0.14/−0.09 · self-ARI 0.656 · cohesion 0.735
(warm-affiliative 0.43) · loving +381/−223 · embedding valence 0.509 / arousal 0.108.

### Per-cell signals — predicted SCALE trend (2B→9B→27B) and TUNING effect (base→IT)
| # | Signal | Scale (2B→27B) | Tuning (base→IT) | Conf | Basis |
|---|---|---|---|---|---|
| 1 | PC1/PC2/PC3 variance % | PC1↑ modest; PC2/PC3 gap widens (arousal resolves onto a cleaner PC2) | PC1,PC2 ↑ slightly | M | circumplex sharpens with capability (Q2) |
| 2 | Valence r (PC1↔pleasure) | ↑ with scale | ↑ (DIRECTION ONLY — 0.81 is Claude's, not a Gemma target) | **H** | Q1; main effect = pooled avg [IT−base] |
| 3 | **Arousal r (PC2↔arousal)** | ↑ strongly; **may be weak/absent at 2B**, resolves by 9B–27B | ↑ (DIRECTION ONLY, no target) | **H** | Q1 + embedding shows depth *builds* arousal; main effect pooled (see Test definitions) |
| 4 | Off-diagonals | shrink toward 0 (cleaner separation) | shrink | M | sharper axes with scale/tuning |
| 5 | **Arousal scale-refit (neg. control)** | fails refit at all sizes (arousal stays scope-conditioned) UNLESS scale strengthens it | IT more likely to SURVIVE the 45-refit | M | §5.2/§9.3 scope-conditioning; interaction candidate |
| 6 | Vector norms (norm↔valence r) | ~stable, slight ↑ | ~stable | **expl** | weak coupling in anchor — EXPLORATORY, outside Holm family |
| 7 | Neighbor cosines | high at all scales (already 0.96 at embedding), slight ↑ | minor | M | local structure is largely inherited |
| 8 | k=10 self-ARI | ↑ (partition more stable) | ↑ (IT tightens) | M | Q1: tighter partition |
| 9 | Cluster cohesion | ↑; warm-affiliative tightens most | ↑, esp. warm-affiliative (0.43→?) | M | Q1: soft partition tightens |
| 9b | **loving PC1/PC2 position** | — | **moves** (toward higher valence / deeper into warm cluster) | **H** | Q1 names loving explicitly; measured via signal iv |
| 10 | Cross-layer RSA to ⅔ layer | stable high-RSA band at each cell's derived layer; band may be shorter at 2B | minor | M | validates per-cell layer choice (§6.5 analog) |
| 11 | Cross-layer neighbor Jaccard | derived layer sits in stable neighborhood band | minor | M | as above |
| 12 | Embedding baseline split | valence present at embedding everywhere; arousal built by depth, **gap larger in bigger models** | — | M | §6.6: depth builds arousal |

### Between-cell signals (C4) — the grid's output
| # | Signal | Prediction | Conf |
|---|---|---|---|
| i | Cross-cell RSA | base↔IT HIGH (tuning perturbs modestly); size↔size LOWER (scale changes geometry more) | M — coin-flip: could reverse if IT reshapes more than a scale step |
| ii | Procrustes | **valence axis aligns well across all cells; arousal axis aligns worse** (weak/scope-conditioned); residual larger for scale than tuning | M |
| iii | Cross-cell ARI | base↔IT near/above self-ARI floor (modest reorganization); size↔size below floor | M |
| iv | Per-emotion displacement | **loving + warm-affiliative + boundary emotions (bored, restless, infatuated) move most; prototype cores move least**; IT tightens warm-affiliative | **H** |

### Test definitions (locks how main effect vs interaction is graded)
- **Tuning main effect** for a signal = the POOLED average of `[IT−base]` across
  the three size-pairs (2B, 9B, 27B), tested as one contrast. A single cell
  reversing (e.g. 2B-it not gaining arousal) does NOT disconfirm the main effect
  — it feeds the interaction below.
- **Interaction** = difference-in-differences across scale, e.g. arousal:
  `([IT−base])@27B − ([IT−base])@2B`. Isolates "tuning effect depends on scale"
  from the scale main effect.
- **Scale main effect** = trend across {2B, 9B, 27B} within a fixed tuning level.

### Interaction predictions (Q3 — do tuning effects depend on scale?)
- **Arousal sharpening under IT is LARGER at bigger scale** — registered as the
  DiD `([IT−base] arousal r)@27B − ([IT−base] arousal r)@2B > 0`. Rationale: IT
  can only sharpen an arousal axis scale has already partly built. Conf **M** —
  the primary interaction claim.
- Cluster-tightening under IT roughly scale-independent. **EXPLORATORY** (outside Holm family).
- `loving` displacement under IT largest where the warm cluster is already
  resolved (9B/27B). **EXPLORATORY** (outside Holm family).

_Edited 2026-07-20 per Adam: targets removed (direction-only for 2,3); tuning
main-effect defined pooled; interaction defined as DiD; signal 6 + two
interaction sub-claims → EXPLORATORY (outside Holm family); signals 3, 5 and
cross-cell RSA (i) kept as predictions._

## (e) Noise-floor tolerance — ✅ LOCKED 2026-07-20 (tier-2 package + Holm–Bonferroni + persist analysis-layer slice approved; anchor bootstrap decision pending)

**Measurement (2026-07-20):** `gemma_grid/scripts/noise_floor.py` run twice as
independent processes on the anchor (27b-base, L31, vectors_171.pt); diffed.
Artifacts: `gemma_grid/geometry_out/noise_floor/run{1,2}.json`.

- **Reproducibility floor = 0.** Every deterministic C1–C4 signal: run-to-run
  `|Δ| = 0.00e+00` (bit-exact). The analysis pipeline is deterministic.
- **k-means** reproduces exactly at fixed seed; its intrinsic floor is seed
  spread: cross-seed self-ARI **0.656 ± 0.075** (min 0.426, 20 seeds);
  ARI-to-locked 0.692 ± 0.093.

**Consequence:** reproducibility floor 0 confirms "did it reproduce" but cannot
set "is a cross-cell difference real." That second tier is set from SAMPLING
error, below. (Reproducibility-tier deltas above ~1e-6 still flag a broken
pipeline — that role is unchanged.)

### Proposed cross-cell "real vs null" thresholds (a difference below → logged NULL)
| Signal | Repro floor (measured) | Proposed effect-size floor | Basis |
|---|---|---|---|
| PC1/PC2/PC3 variance % | 0.00 | **Δ ≥ 1.0 pp** | the original's own figures disagree ~1pp (26↔27, 14↔15); sub-1pp is not meaningful |
| valence_r, arousal_r, off-diagonals (n=45) | 0.00 | **Δr ≥ 0.15** | Fisher-z: SE(Δz)=√(2/42)=0.22 → ≈0.14 in r near 0.6 |
| norm↔valence / norm↔\|arousal\| (n=171) | 0.00 | **Δr ≥ 0.10** | Fisher-z: SE(Δz)=√(2/168)=0.11 → ≈0.09 in r near 0.38 |
| neighbor cosines | 0.00 | **Δcos ≥ 0.03** | bounded metric; sub-0.03 within resampling wobble |
| layer stability (RSA-to-analysis) | 0.00 | **Δ ≥ 0.01** | reproducible to 0; small guard |
| `loving` PC1/PC2 position | 0.00 | **compare POST-Procrustes only** | raw coords are in each cell's own PCA units — not comparable; movement is captured by between-cell signal iv (displacement) |
| k=10 ARI (cross-cell, signal iii) | exact @ seed | **cross-cell ARI < 0.51 ⇒ real reorg; ≥ 0.51 ⇒ null** | 0.51 = self-ARI mean 0.656 − 2σ(0.075) |

**Between-cell signals i–ii–iv (RSA, Procrustes residual, displacement):** these
do not exist for a single cell, so their floor cannot be measured from the anchor
alone. Pre-registered rule: **measure their floor at the first contrast** (e.g.
re-extract the anchor under a second seed, or the first base↔IT pair) and apply
it before reading any i/ii/iv result. Flagged, not silently skipped.

**Open option for Adam:** the deterministic effect-size floors above are grounded
in analytic sampling SE. If you want them empirically tightened, I can add a
bootstrap (resample stories, recompute) to produce per-signal CIs — a laptop job
IF the per-story intermediate is persisted (see item 3 of the extraction contract).

### Tier-2 test procedure per signal (reuses existing E1 machinery, one level up)
Confirmed against an independent second read (Downloads/gemma_grid_tier2_noise_floor_proposal.md);
methods converge. Each cross-cell contrast is tested as:
- **Correlations (valence/arousal r, off-diagonals, norm-r):** Fisher r-to-z
  TWO-SAMPLE test on the shared overlap (n=45 for R&M — constant across cells,
  since the overlap is a model-independent label intersection and the family
  shares one tokenizer; n=171 for norm-r). Closed form; extends the §5.2
  permutation null from "vs zero" to "cell A vs cell B."
- **PCA variance % and neighbor cosines:** no closed form → STORY-LEVEL
  bootstrap. Resample each emotion's stories with replacement (~1,000×),
  rebuild the 171×d matrix, recompute the statistic → per-cell SD. Cross-cell
  gap is real if it clears ~2× the pooled SD. Needs the persisted per-story
  intermediate (extraction contract item 3). **Basis convention (uniform across
  all six cells, incl. the anchor rerun):** each resample projects out that
  cell's OWN official cleaned basis, held fixed; only emotion-stories are
  resampled. This keeps anchor and new cells estimating the SAME quantity — the
  symmetry the anchor full-rerun exists to protect.
- **k=10 ARI:** cross-cell ARI read against the cell's self-ARI band (floor
  ~0.43); below → real reorganization. (§9.3 already uses this exact test.)
- **Layer-stability:** reuse the §6.4 bootstrap-CI code, cross-cell.

### Multiple-comparisons correction — 🟡 PROPOSED (new; not in my earlier draft)
6 cells × ~8 signals × several contrasts will throw false positives at flat
α=0.05. Proposed: **Holm–Bonferroni WITHIN each of the three pre-registered
contrast families** (tuning effect, scale effect, interaction) — controls
family-wise error without the over-conservatism of one pooled correction across
all families. Holm = step-down Bonferroni: sort m p-values, test the k-th
smallest against α/(m−k+1), stop at first failure; uniformly more powerful than
plain Bonferroni, no independence assumption. Locked at GATE 1 alongside the
thresholds. Sub-floor OR non-significant-after-correction differences → NULL.

### Bootstrap-retention decisions
- Per-story intermediate: **persist analysis-layer slice (~4 GB/cell)** —
  ✅ APPROVED 2026-07-20 (extraction contract item 3).
- Anchor variance%/cosine bootstrap: ✅ **FULL RERUN APPROVED 2026-07-20** (Adam,
  research-integrity call). Rationale: a *subsampled* anchor bootstrap would give
  the anchor a thinner, differently-shaped noise estimate than the 5 new cells
  get free from their own full extraction, asymmetrically weakening every
  anchor-involving contrast — 27B base↔IT and every scale step up to 27B, exactly
  the comparisons this study cares most about. Spec:
  1. **Full rerun** of ALL ~1,200 stories/emotion (the fixed corpus, NOT a slice)
     + the 500 neutrals through `google/gemma-2-27b`, forward pass **truncated at
     L31** (`stop_at_layer=32`; layers 32–46 and final logits not needed here).
  2. Persist per-story pooled L31 vectors. Bootstrap = resample emotion-stories,
     recompute diff-of-means, then **project out the SAME official neutral-corpus
     cleaned basis** (reconstructed from the recaptured 500 neutrals at L31). All
     reported numbers are on the CLEANED basis, so the bootstrap must be too — a
     raw-activation bootstrap estimates the noise floor of a *different quantity*
     than every cross-cell test actually compares.
  3. **REPRODUCTION GATE** before trusting anything: average the recaptured
     per-story vectors back to 171 emotion vectors, diff vs locked
     `vectors_171.pt` at L31 under `verify_reproduction.py`'s TOL bands. Scope:
     the L31-dependent anchors (variance, rankings, loving, neighbours, k-means,
     fig8). The 2 layer-stability anchors need all layers → out of scope for an
     L31-only recapture, and are unchanged (they come from the locked all-layer
     set). If the L31 anchors don't reproduce → **STOP, bootstrap void**: the
     rerun isn't measuring the same quantity as the original.
  4. Runs as its OWN **dry-run → pilot → full** inside Phase 3, same gate
     discipline as the 5 new cells — a **6th extraction job**, not a special case
     that skips to a full GPU run.

---
### Decision log
- 2026-07-20 — (a)/(b) locked: six cells; strict-2/3 layer rule; 2B headline L17
  with L18 pre-registered as sensitivity check.
- 2026-07-20 — (c) signal set expanded to C1–C4 (per-cell geometry, depth
  validation, embedding baseline, between-cell contrasts) + dimension-safety
  constraint. C1–C4 definitions PROPOSED (awaiting Adam's confirm/amend).
- 2026-07-20 — (c)/C5 locked: logit-lens INCLUDE (capture W_U + final RMSNorm at
  extraction); held-out-text DEFER. Sets the per-cell extraction output contract.
- 2026-07-20 — (c) LOCKED: C1–C4 confirmed for all six cells; loving PC1/PC2
  added (#9b).
- 2026-07-20 — (e) MEASURED: anchor analysis run twice → deterministic repro
  floor = 0 (bit-exact); k-means self-ARI 0.656±0.075. Two-tier thresholds
  proposed (repro floor + sampling-error effect-size floor). Awaiting approval.
- 2026-07-20 — (e) extended after independent second read: per-signal tier-2
  test procedure (Fisher r-to-z / story bootstrap / self-ARI band); Holm–
  Bonferroni within each contrast family (NEW). FINDING: anchor per-story
  activations not retained locally; extraction contract item 3 added to persist
  the per-story analysis-layer intermediate for the 5 new cells (~4 GB/cell).
  n=45 overlap confirmed constant across cells.
- 2026-07-20 — (e) LOCKED: Adam approved the tier-2 package (thresholds + tests
  + Holm–Bonferroni) and the 5-cell per-story analysis-layer intermediate.
  Anchor-activation availability still under consideration (analytic-only meanwhile).
- 2026-07-20 — (d) STRAWMAN drafted (per-cell scale/tuning + between-cell +
  interaction predictions, confidence-flagged).
- 2026-07-20 — (d) LOCKED (Adam's edits): direction-only for signals 2,3 (no
  0.81/0.66 targets); tuning main-effect = pooled [IT−base]; interaction = DiD;
  signal 6 + 2 interaction sub-claims → EXPLORATORY; signals 3,5,RSA(i) kept as
  predictions.
- 2026-07-20 — ANCHOR bootstrap RESOLVED (Adam, integrity call): full
  L31-truncated rerun of the anchor (all ~1,200 stories/emotion + 500 neutrals),
  cleaned-basis bootstrap, reproduction-gated vs locked vectors_171.pt, run as a
  6th Phase-3 job under dry→pilot→full. No open sub-decisions; GATE 1 fully
  assembled, awaiting formal sign-off.
- 2026-07-20 — PHASE 0 CLOSED: `gemma_grid/scripts/run_cell_extraction.py` built
  (config-driven, modes A/B, lens capture, per-story slice, reproduction gate).
  Validated no-GPU: reproduction gate identity→24/24 PASS & perturbed→void on
  real vectors_171.pt; lens capture + truncated stop_at_layer capture verified
  on the real TransformerBridge (gpt2/CPU); all 6 configs load+assert. The full
  Gemma extraction path (modes A/B end-to-end) is exercised at the Phase-2 pilot.
- 2026-07-20 — **GATE 1 SIGNED OFF (Adam).** Phase 2 begun: CPU shape dry-run
  (`dry_run_shapes.py`) PASSES at all 3 new shapes (2B/9B/27B → L17/L28/L31).
  Pilot harness added: `--pilot` corpus subset; outputs namespaced per cell under
  `gemma_grid/data/<cell_id>/` (fixed a latent write into first-round data/ dirs +
  cross-cell canonical.pt collision). Pilot runs on the GPU pod — see
  `gemma_grid/PHASE2_PILOT.md` → GATE 2 (Adam) before any full extraction.
- 2026-07-23 — **PHASE 3 COMPLETE: ALL SIX CELLS EXTRACTED + VALIDATED.** 2b-base,
  2b-it, 9b-base, 9b-it, 27b-it (mode A, disk-light) + 27b-base anchor rerun
  (mode B, truncated L31). **REPRODUCTION GATE: PASS — 24/24 anchors reproduced,
  |Δ|~1e-9** vs locked vectors_171.pt, so the anchor bootstrap is VALID and the
  disk-light/truncated extraction is proven faithful to the round-one pipeline.
  Basis integrity verified: all 5 mode-A cells distinct (hashes), correctly
  shaped, emotion-order aligned; and the one dimensionally-invisible risk
  (27B-IT cleaned with the anchor's basis, shared d=4608) decisively ruled out —
  IT's basis rank 16 vs anchor's 23, and applying the anchor basis to IT's raw
  gives 0.168 rel diff (≠0), control with IT's own basis gives 1e-6. All six
  cells' artifacts pulled to Adam's laptop. → **GATE 3** (Adam) before Phase 4.
- 2026-07-21 — **PHASE 2 PILOT PASSED** on the H100 pod (gemma-2-2b, real model).
  Mode A + mode B both EXIT=0; core assert fired correctly:
  `n_layers=26 d_model=2304 analysis L17`. Outputs: canonical (3,26,2304),
  per-story slice layers[17] (3,1,2304)+neutral, lens W_U (2304,256000).
  Fixes made during pilot: lens `W_U` stored native bf16 not float32 (2.36→1.18 GB;
  27B would've been 4.7 GB float32). Pod provisioning done: code+corpus+gate inputs
  synced, `hf auth login` persisted a token file (was env-only, invisible to
  non-interactive runs), gemma-2-2b cached. Awaiting Adam's GATE 2 sign-off →
  Phase 3 full extraction.
