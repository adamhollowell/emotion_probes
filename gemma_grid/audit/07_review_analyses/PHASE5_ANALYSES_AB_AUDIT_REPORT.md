# Phase-5 adversarial audit of Analyses A and B: report

**Auditor.** A fresh Claude Code session (Opus 5.5) with no history on this project, 2026-10-14.
**Governing documents.** The commission `gemma_grid/PHASE5_ANALYSES_AB_AUDIT_PROMPT_100T.md` (sha256 `eac0fbbe9be678c1…`, 146 lines, verified) and the dispatch `gemma_grid/PHASE5_ANALYSES_AB_AUDIT_DISPATCH_100T.md` (sha256 `80599e866df0c8a4…`, 216 lines, verified). I read the commission first and worked to it. The dispatch's charges S1 to S7 are reported separately below.
**Nothing was edited.** Every computation ran from a scratch directory outside the repository. The only file this audit writes in the repository is this report.

**Method.** Every numeric claim was derived first with code I wrote: a Procrustes and retrieval implementation written from the frozen spec's text, and a PCA, correlation and bootstrap implementation written from reading `noise_floor.compute`. I imported none of the producer scripts or the frozen module. I sealed my values before opening any producer output:

- **Analysis B values.** Sealed at 2026-10-14T21:04:13Z in `BLIND_B_values_sealed.json`, sha256 `4795b091…`.
- **Analysis A pair values.** Sealed at 2026-10-14T21:18:27Z in `BLIND_A_pairs_summary.json`, sha256 `8b4a36e3…`.

Both files are held in the session's scratchpad. In every table below, my value comes first and the producer's second. The only exceptions are the gate references, which were pinned in advance.

---

## 0. Bottom line

### Table 0.1. Pre-committed charges (written before either analysis ran, per the commission)

The first column gives each charge's status. The last column gives the largest absolute difference between my value and the producer's.

| Charge | Status | Largest abs. diff (mine vs producer) |
|---|---|---|
| Gate A (own code) | **PASSED**: top-1 1.0 (171/171), median rank 1, median held-out cosine 0.971624 (rounds to .972) | 2.6e-13 against the stored artifact |
| Gate B (own code) | **PASSED**: pooled Δ +.0558 / −.0829 / +.0409 / −.0881. Per-cell signals match the stage A `deterministic` blocks | 9.6e-9 (float64), 2.0e-7 (float32) |
| A1 procedure claim | **SUSTAINED**. Ten differences listed; none changes a number. See §4.1 for a defect that the frozen procedure and the wrapper share | 0 in ranks; 2.4e-10 in cosine |
| A2 frozen script unmodified | **SUSTAINED**. Hash `ee736f5d…5526` matches the 2026-08-28 pod bundle record | none |
| A3 which script ran | **SUSTAINED**, within limits on what the record can show | none |
| A4 retrieval numbers re-derived, all 30 pairs | **SUSTAINED** | rank 0; per-emotion cosine 2.4e-10; summaries 1.9e-10 |
| A5 all 30 pairs present | **SUSTAINED**: 30 distinct pairs, the complete ordered set | none |
| A6 null versus the bar | **The arm cannot be compared with the bar** (arithmetic below). **The claim that no comparison is invited is REFUTED in part**: the output reports `n_at_or_above_observed: 0` | null counts: 0 of 500 differ (element by element) |
| B1 shared draw | **SUSTAINED** | draw-matrix sha256 identical (`4fa383ec…`) |
| B2 fixed PCA, fit on 171 | **SUSTAINED** | none |
| B3 intervals re-derived | **SUSTAINED** on all 18 quantities | 4.3e-7 |
| B4 two error models | **SUSTAINED for the final output.** **UNTESTED for the superseded first output**, which no longer exists | none |
| B5 draws and seed | **SUSTAINED**: B = 2,000, seed 20261014 | none |
| X1 protected files untouched | **SUSTAINED** for `PREREGISTRATION.md`, the frozen module and `COMMITTED_PHRASES.tsv`. **UNTESTED** for the draft and the ledger, both changed after the runs | none |
| X2 input manifests | **SUSTAINED**: 39 of 39 hashes match the files on disk. No used input is unhashed | none |
| X3 no verdict language | **SUSTAINED**, except for the A6 count. The SE-ratio framing is discussed under S7 | none |
| X4 basis labels | **Three label disagreements reported, not resolved** | none |

### Table 0.2. Supplementary charges (written after the results existed, per the dispatch)

| Charge | Status | Largest abs. diff (mine vs producer) |
|---|---|---|
| S1 pooled SE | Stage C uses **√(2/(3·42))**. Every pooled p the draft prints is correct under that form and wrong under √(2/42). The single-pair √(2/42) is correct where the draft uses it | p values agree to the printed precision |
| S2 interaction forms | **SUSTAINED.** Each form sits in the slot assigned to it by the 2026-10-08 ruling. The claimed "agreement of the Fisher-z forms" is an identity by construction, not a check | 2.5e-7 |
| S3 per-draw flip | **Not registered.** It was pre-stated in the dated handoff, with precedent only for the on-diagonal rows. The tail argument **mostly holds** for the intervals (endpoints move by at most .0036) and **does not hold** for the SE, which doubles. The inflated SE propagates into the SE-ratio block | flip count 10/2000, same as the producer; SE 8e-8 |
| S4 the 14 new pairs | **SUSTAINED.** The "wrapper" is the new script itself: hashed, not frozen. It reproduces the stored 16 pairs | 2.6e-13 (mine vs stored) |
| S5 superseded runs | **UNTESTED**: no record of either first run survives. The birth time of B's output file confirms that a first B output existed | none |
| S6 which vectors entered | The mapping **is determined by hashed code**. For contextual inputs, a gate or reproduction confirms it; for static inputs, only path naming does | none |
| S7 amendments complete | **A: incomplete.** A count at or above the observed value survives as a field. **B: complete in wording.** The ratio block's arithmetic is exact, but it re-supplies the floor comparison by one multiplication | ratio rows: 1/√3 holds to 10 decimal places |

**A finding that fits no charge (§4.1), and one that bears on the commission itself (§4.2).**

- **§4.1.** The frozen Procrustes map is not unique when it is fit on 137 emotions. Held-out cosines depend on which LAPACK routine fills the map's unconstrained directions.
- **§4.2.** By the filesystem's own creation times, the commission file was created at 14:49:42. That is after B's first output (13:43:46) and after B's final output (14:48:36).

---

## 1. A6, B4 and S1: the three charges that can touch a printed number

### A6. The null, and whether it can be compared with the registered bar

- **Shuffle count used: 500.** This comes from the manifest (`n_null: 500`), the script (`N_NULL = 500`) and the output (`n_null: 500`, with 500 entries in `null_counts`). My own run used the same construction (`RandomState(0).permutation(171)`, drawn 500 times in sequence and applied to the target rows) and the same count.
- **Smallest attainable add-one p at 500 shuffles:** 1/501 = **0.001996**.
- **The bar** is median rank ≤ 2 **and** permutation p < .001.

**The embedding-table arm cannot be compared with the pre-set bar at all.** Even if no shuffle ever reached the observed statistic, the arm could not produce p < .001. This holds for every possible outcome, not just the one observed.

As a contrast, the contextual floor ran 10,000 shuffles and got add-one p = 9.999e-05 (`pmap_transport_floor_100T.MANIFEST.txt`).

**Does the output invite the comparison anyway?**

The label disclaims it. It reads: "SHAPE PROBE ONLY (amendment A, 2026-10-14): … Not the frozen 10,000-draw floor; not compared to the §4.3 bar; no p value is computed or reported; no pass/fail."

The same block nonetheless reports three things:

- `"observed_top1_count": 171`, next to `"counts": {"max": 4, …}`.
- `"n_at_or_above_observed": 0`, which is script line 202.
- The full 500-element `null_counts` list.

The count at or above the observed value is the numerator of the add-one estimator. From it, p = (1 + 0)/501 = .002 follows in one step. **That is the ingredient A6 names, stated as zero.** The handoff asked for it (step 4 of its report section). Amendment A removed the p value but not this ingredient.

**Null re-derivation (blind, my own code).** See §1.1.

### B4. Two error models, and whether the output conflates them

**The final output** (`tuning_dependence_aware_ci_100T.json`, sha256 `8aac55a8…`) contains no comparison of any interval with ±.15 or with any other floor. The word "floor" appears in one value only: `"amendment_B": "2026-10-14: floor flags removed; intervals are not compared to any floor"`. It also appears inside the module name `noise_floor`. The `"pass": true` field belongs to the reproduction gate. It is not a verdict.

The output does not say in words that the bootstrap SE and the Fisher SE rest on different error models. `se_note` gives formulas only: "stageC uses sqrt(2/42) per-size, sqrt(2/(3*42)) for POOLED, sqrt(4/42) for the DiD; the handoff states sqrt(2/42) for pooled. Both ratios are given." The analysis title does say "dependence-aware". On its own wording, I find nothing a reader would take for an equivalence test. The ratio block's framing is a separate matter, covered under S7.

**The superseded first output cannot be audited.** The handoff (line 79) specified "two flags, whether the 95 percent interval lies wholly inside (−.15, +.15) and whether the 90 percent interval does". Flags in that form are what B4 anticipated: a dependence-aware interval set against an independence-based floor. The first output existed (§S5). It was read and then overwritten in place, so whether its wording conflated the two models is **UNTESTED**.

### S1. The pooled standard error (supplementary)

**The form registered grading uses.** `scripts/phase4_stageC_grade.py` line 43 defines `SE_POOL_45 = np.sqrt(2 / (3 * (N45 - 3)))`. It is used for the pooled valence and arousal p values (lines 106–107) and for the pooled off-diagonal p values (line 113). `geometry_out_100T/stageC_grading.json` was written by `scripts/phase4_stageC_grade_100T.py`, which calls `G.main()` unchanged, so the 100T grading used **√(2/(3·42)) = 0.12599**. The independently produced `geometry_out_100T/tuning_fisher_pooled_verification_100T.json` (2026-10-08) agrees.

√(2/(3·42)) is the correct SE of the mean of three independent Δz values, each with variance 2/42. The handoff's √(2/42) for the pooled case is a misstatement. The producer caught it and reported both ratios.

**Every printed number that depends on a pooled Fisher SE, from my own recomputation.**

The pooled Δz values (mine) are: valence +.1253, arousal −.1161, PC1-with-arousal +.0413, PC2-with-pleasure −.0906. The n = 171 norm row is computed from the stage A deterministic blocks.

| Draft location | Printed | p under √(2/(3·42)) | p under √(2/42) | Correct under stage C? |
|---|---|---|---|---|
| Table 2 valence; §5 l.266 "(p = .32 …)"; Table E.2 R&M row | .320 | .3200 | .5659 | yes |
| Table 2 arousal; l.266 "… and .36"; Table E.2 R&M row | .357 | .3567 | .5946 | yes |
| Table 2 off-diagonal PC1 with arousal; l.283 "(p = .74 …)" | .743 | .7430 | .8499 | yes |
| Table 2 off-diagonal PC2 with pleasure; l.283 "… and .47" | .472 | .4721 | .6781 | yes |
| l.289 norm against PC1, "(p = .043)", n = 171 | .043 | .0426 under √(2/(3·168)) | .2417 under √(2/168) | yes |

Every printed value is right under the stage C form and wrong under the other. No printed value is right under √(2/42) alone.

**Untested.** The p values of the four non-registered norm arms in Table E.2 (NRC-VAD natural and matched-45, Warriner natural and matched-45) depend on the same pooled form, but I did not recompute them. That would need those arms' rating overlaps and their pooled Δz, which come from the norm-robustness producer.

**Single-pair form.** At line 67 the draft says "√(2/42) = 0.22 in z, about .14 in r near .6". That is correct as the SE of a difference of two independent Fisher z values at n = 45: 0.2182, and 0.2182 × (1 − .6²) = .140. No per-size p is printed in the draft. The per-size p values in the grading file (valence .590 / .520 / .589, arousal .442 / .876 / .503) are correct under √(2/42), which is the right form for a single pair.

**The interaction's p = .94** (l.287) is correct under √(4/42): Δz_DiD = .02161, z = .070, p = .9442.

**Could a verdict change?** I re-grade nothing. I checked one structural fact in the code: `phase4_stageC_grade.py` lines 221–223 assign `NULL_subfloor` whenever `above_floor` is False, before Holm is consulted. A smaller SE therefore cannot change a sub-floor verdict.

Several sentences in the draft nonetheless invite a reader to think that a sub-floor correlation verdict is weakened by the error model:

- **l.69:** "those two floors are roughly four times stricter than a dependence-aware floor would be, and the tests they back are about as much too conservative."
- **l.291:** "A sub-floor null on a correlation row is therefore a null against a floor about four times stricter than a dependence-aware floor would be. It reflects the inflated floor and is not evidence of no effect."
- **l.476:** "A sub-floor null on a correlation row is therefore weaker evidence of absence than the floor suggests, while above-floor results are unaffected."

None of the three sentences comes from Analysis B's output. All three are in the draft now. The draft changed after both runs (X1), so whether any of them postdates Analysis B cannot be established.

### 1.1 Null re-derivation (A6, S7)

My blind 500-draw null under the rank-restricted map, the same construction with the null-space completion dropped (see §4.1):

- max 4, p99 3, p90 2, median 0, mean 0.616
- count at or above 171: 0

My blind 500-draw null under the frozen call (gesdd), with the same permutations:

- max 4, p99 3, p90 2, median 0, mean 0.62
- count at or above 171: 0

The producer's null matches on every summary (max 4, p99 3.0, p90 2.0, median 0.0, mean 0.62, at-or-above 0). The full 500-element `null_counts` list is **identical element by element** to mine.

The rank-restricted null differs from gesdd's in 10 of 500 draws, each by one count. The null's shape does not depend on the completion issue in §4.1.

---

## 2. Per-charge detail, pre-committed

### Gates

**Gate A.** Implementation: my own 5-fold Procrustes plus global scale, using folds `RandomState(0).permutation(arange(171) % 5)`, centered float64 vectors at L17 and L31, retrieval against all 171 targets, and a strict-greater rank rule. My result: top-1 1.0 (171/171), median rank 1, top-5 1.0, median held-out cosine **0.9716237565**. Pinned: 1.0 and .972. **Passed.**

**Gate B.** Implementation: my own PCA as an eigendecomposition of the 171 × 171 Gram matrix (not an SVD of the data), the POS/NEG and HI/LO sign rules, the R&M overlap of 45, per-axis sign flips, and Pearson r.

- Per-cell signals against the stage A `deterministic` blocks: max |diff| = **9.6e-9** in float64 and **2.0e-7** in float32. Both are inside 1e-6.
- Pooled deltas (both precisions): **+.0558, −.0829, +.0409, −.0881**. All four match the pinned values.

**Passed.**

### A1. The procedure claim

I compared the frozen `pmap_transport_100T.py` with the script that ran, `pmap_embedding_table_transport_100T.py`:

1. **Loader.** The frozen `load_vectors` requires a 2-D (171 × d) matrix and a "100T" path. The wrapper loads the 3-D (171 × L × d) store and slices `[:, layer, :]`. The August run used 2-D adapters sliced at the same layers ("input_adapter_note" in `pmap_transport_100T.json`). **No numeric change**: the stored 16 pairs reproduce to 1.1e-16 (producer) and 2.6e-13 (mine).
2. **The "100T" path guard.** The wrapper asserts it for both the contextual and the static paths. No change.
3. **Static-vector loader.** This is new code for the (171, 1, d) schema; it squeezes the singleton axis and casts to float64 as the frozen loader does. No change to procedure.
4. **Centering, Procrustes fit, cosine, CV loop and ranking.** These are imported unmodified (`P.center`, `P.eval_transport`). No change.
5. **Fold construction.** It is re-expressed as `RandomState(P.FOLD_SEED).permutation(arange(171) % P.N_FOLDS)`, identical to frozen line 111. No change.
6. **Pair coverage.** Frozen runs the primary pair, its reverse, and the a < b pairs (16 in all). The wrapper runs all 30. Coverage changes; numbers do not.
7. **Ridge secondary on the primary pair.** Omitted by the wrapper. An output is omitted; no number changes.
8. **Null.** Frozen runs 10,000 permutations in-process and reports an add-one p on the top-1 rate. The wrapper pre-generates 500 permutations from the same `RandomState(0)` stream, so they are the first 500 of the frozen sequence. It records counts, not rates, and computes no p.
9. **Parallelism.** The wrapper uses spawn workers with `VECLIB_MAXIMUM_THREADS=1`. Per-pair results are independent of order. `pool.map` keeps the null in permutation order.
10. **Emotion-list assertion.** The wrapper also asserts that the static lists match. This is an added check.

**None of the ten changes a number.** The defect in §4.1 lives in `fit_procrustes`, so the frozen run and this one share it.

### A2. The frozen script is unmodified

The current sha256 is `ee736f5d36c9a04c8d6e4eaf043d932cbb979a5e4fba22a0ea0e8613b91e5526`. The same full hash is recorded earlier in `nogpu_additions_100T/POD_RUN_ARCHIVE_2026-08-28/POD_BUNDLE/SHA256SUMS_INPUTS.txt`, in `geometry_out_100T/pmap_transport_floor_100T.MANIFEST.txt` ("verified UNCHANGED on pod post-run"), and in `MEMO_LC49_PMAP_FLOOR_VERIFIED_100T.md`. The file's mtime is 2026-08-26 20:20:09.

### A3. Which script ran

The manifest's `script` and `script_sha256` (`d79ca223…`) match the file on disk.

- **File times.** Both born and modified at 14:47:01 local time.
- **Recorded run.** The manifest gives `generated_utc` 20:16:48Z and `wall_s` 5381.6, so the run started at about **14:47:06 local**, 5 s after the script's last save. The script was not modified during the recorded run.

Two things cannot be shown:

- **Earlier form of the script.** Whether it existed before 14:47:01, for instance for the stopped first run, cannot be established. A birth time of 14:47:01 is consistent with both a new file and an atomic-rename save.
- **Relation to the commission's sentence.** The commission's statement that the file "already existed before this commission was written" fits a birth time 2 min 41 s before the commission file's own (14:49:42).

### A4. Retrieval numbers re-derived, all 30 pairs, both vector sets

I computed all 30 pairs rather than sampling. For each pair below, the values are median rank / top-1 / top-5 / median cosine, mine first. All four agree with the producer to the precision shown and to within 1.9e-10 at full precision. Per-emotion static ranks are identical; per-emotion cosines agree to within 2.4e-10.

| Pair (source > target) | Static table (mine = producer) | Contextual (mine = producer) |
|---|---|---|
| 2b-base > 27b-base (primary) | 1 / 1.000 / 1.000 / .9435 | 1 / 1.000 / 1.000 / .9716 |
| 2b-base > 2b-it | 1 / 1.000 / 1.000 / .9551 | 1 / 1.000 / 1.000 / .9779 |
| 2b-base > 9b-base | 1 / 1.000 / 1.000 / .9506 | 1 / 1.000 / 1.000 / .9790 |
| 2b-base > 9b-it | 1 / 1.000 / 1.000 / .9511 | 1 / 1.000 / 1.000 / .9734 |
| 2b-base > 27b-it | 1 / 1.000 / 1.000 / .9435 | 1 / 1.000 / 1.000 / .9687 |
| 2b-it > 2b-base | 1 / 1.000 / 1.000 / .9560 | 1 / 1.000 / 1.000 / .9781 |
| 2b-it > 9b-base | 1 / 1.000 / 1.000 / .9511 | 1 / .994 / 1.000 / .9759 |
| 2b-it > 9b-it | 1 / 1.000 / 1.000 / .9501 | 1 / 1.000 / 1.000 / .9776 |
| 2b-it > 27b-base | 1 / 1.000 / 1.000 / .9439 | 1 / .994 / 1.000 / .9705 |
| 2b-it > 27b-it | 1 / 1.000 / 1.000 / .9432 | 1 / 1.000 / 1.000 / .9732 |
| 9b-base > 2b-base | 1 / 1.000 / 1.000 / .9585 | 1 / 1.000 / 1.000 / .9821 |
| 9b-base > 2b-it | 1 / 1.000 / 1.000 / .9582 | 1 / 1.000 / 1.000 / .9786 |
| 9b-base > 9b-it | 1 / 1.000 / 1.000 / .9539 | 1 / 1.000 / 1.000 / .9770 |
| 9b-base > 27b-base | 1 / 1.000 / 1.000 / .9436 | 1 / 1.000 / 1.000 / .9745 |
| 9b-base > 27b-it | 1 / 1.000 / 1.000 / .9433 | 1 / 1.000 / 1.000 / .9710 |
| 9b-it > 2b-base | 1 / 1.000 / 1.000 / .9582 | 1 / 1.000 / 1.000 / .9766 |
| 9b-it > 2b-it | 1 / 1.000 / 1.000 / .9582 | 1 / 1.000 / 1.000 / .9810 |
| 9b-it > 9b-base | 1 / 1.000 / 1.000 / .9540 | 1 / 1.000 / 1.000 / .9769 |
| 9b-it > 27b-base | 1 / 1.000 / 1.000 / .9441 | 1 / 1.000 / 1.000 / .9723 |
| 9b-it > 27b-it | 1 / 1.000 / 1.000 / .9433 | 1 / 1.000 / 1.000 / .9762 |
| 27b-base > 2b-base | 1 / 1.000 / 1.000 / .9548 | 1 / 1.000 / 1.000 / .9762 |
| 27b-base > 2b-it | 1 / 1.000 / 1.000 / .9561 | 1 / 1.000 / 1.000 / .9753 |
| 27b-base > 9b-base | 1 / 1.000 / 1.000 / .9485 | 1 / 1.000 / 1.000 / .9760 |
| 27b-base > 9b-it | 1 / 1.000 / 1.000 / .9485 | 1 / 1.000 / 1.000 / .9742 |
| 27b-base > 27b-it | 1 / 1.000 / 1.000 / .9497 | 1 / 1.000 / 1.000 / .9745 |
| 27b-it > 2b-base | 1 / 1.000 / 1.000 / .9553 | 1 / 1.000 / 1.000 / .9734 |
| 27b-it > 2b-it | 1 / 1.000 / 1.000 / .9556 | 1 / 1.000 / 1.000 / .9777 |
| 27b-it > 9b-base | 1 / 1.000 / 1.000 / .9483 | 1 / 1.000 / 1.000 / .9728 |
| 27b-it > 9b-it | 1 / 1.000 / 1.000 / .9483 | 1 / 1.000 / 1.000 / .9777 |
| 27b-it > 27b-base | 1 / 1.000 / 1.000 / .9499 | 1 / 1.000 / 1.000 / .9744 |

### A5. All thirty pairs are reported

The output has 30 keys, 30 of them distinct, and they equal the full ordered set over the six cells. There are no duplicates and none is missing. Of the 30, 16 contextual entries are copied from the stored artifact and 14 are recomputed and labelled "recomputed here (pair not in stored artifact; frozen script runs a<b pairs only)".

### B1. The shared draw

**From the code.** `draws = rng.integers(0, N45, size=(B, N45))` is drawn once (script line 162). Inside the loop, `ix = draws[b]` indexes every cell in one dict comprehension (line 168). Nothing is redrawn per cell.

**Independently.** I generated `default_rng(20261014).integers(0, 45, (2000, 45))` myself. Its int64 sha256, `4fa383ec5c8869fb…`, equals the output's `draw_indices_sha256`. My joint-resampling implementation reproduces every SE and interval to within 4.3e-7. A per-cell redraw would not.

### B2. The fixed PCA

`fit_cell` runs once per cell, before the bootstrap loop (lines 115–121). `nf.pca(Vl, 5)` is called on `Vl = V[:, layer, :]`, which has 171 rows, and the sign rules are applied on all 171 before restricting to the 45 rated emotions. Inside the loop only the 45-vectors are indexed. My implementation fits on 171 with a different decomposition, an eigendecomposition of the Gram matrix, and agrees.

### B3. Intervals re-derived

I recomputed all 18 quantities: four signals × (three per-size contrasts + pooled), plus the two interaction forms.

| Quantity | Point r | Boot SE r | 95% percentile | 90% percentile |
|---|---|---|---|---|
| valence pooled | +.0558 / +.0558 | .0158 / .0158 | [+.0303, +.0910] / same | [+.0331, +.0852] / same |
| arousal pooled | −.0829 / −.0829 | .0211 / .0211 | [−.1290, −.0446] / same | [−.1191, −.0497] / same |
| PC1-arousal pooled | +.0409 / +.0409 | .0141 / .0141 | [+.0139, +.0690] / same | [+.0177, +.0645] / same |
| PC2-pleasure pooled | −.0881 / −.0881 | .0333 / .0333 | [−.1390, −.0351] / same | [−.1319, −.0456] / same |
| arousal 2B (named per-size row) | −.1362 / −.1362 | .0312 / .0312 | [−.1981, −.0804] / same | [−.1868, −.0887] / same |
| arousal 27B (named per-size row) | −.0892 / −.0892 | .0261 / .0261 | [−.1477, −.0415] / same | [−.1352, −.0496] / same |
| interaction, raw r | +.0470 / +.0470 | .0288 / .0288 | [−.0051, +.0988] / same | [+.0043, +.0899] / same |
| interaction, stage C form | +.0211 / +.0211 | .0356 / .0356 | [−.0366, +.0995] / same | [−.0276, +.0836] / same |

The largest difference across all 18 rows is 4.3e-7, at an interval endpoint, because float32 PCA routes differ between my code and the producer's. The handoff fixes the seed but not the generator; the producer chose `default_rng`. A second generator (`RandomState(20261014).randint`) moves interval endpoints by at most **.0068**, which is the Monte Carlo spread at B = 2,000.

### B5. Draws and seed

The output, the manifest and the code all record B = 2000 and seed 20261014, matching the handoff. Nothing on disk records a change after the final run. However, the script was modified at 14:47:22, after its first run at 13:43:46 (S5), and the version that produced the first output does not survive.

### X1. Nothing protected was touched

| File | Evidence | Result |
|---|---|---|
| `PREREGISTRATION.md` | mtime 2026-07-23 16:40:28 (no prior hash found) | unchanged since July by mtime |
| `pmap_transport_100T.py` | hash matches August records | unchanged |
| `drafting/COMMITTED_PHRASES.tsv` | mtime 2026-10-12 16:01:56 | unchanged since before the handoff |
| `drafting/GGE1_DRAFT_v7_100T.md` | The handoff recorded sha1 `b5674e14…`; it is now `8060a5bc…`, mtime **16:30:07** | **changed after both runs** |
| `Gemma_Grid_Findings_Ledger FINAL.md` | mtime **16:31:27**, beside `_to_delete/_ledger_prev_lc62.bak` and `_lc62.txt` | **changed after both runs** (the Ledger's LC62 entry) |

Neither producer script writes anything except its own output and manifest; I confirmed this by reading the code. The draft's 16:30 change coincides with `drafting/GGE1_v7_CLOSEOUT_2026-10-12.md` (16:30:23), which says "This chat alone edits v7". No copy of the draft at `b5674e14` survives, so the artifacts cannot show whether the producer session hand-edited the draft or ledger. **UNTESTED** for those two files. Settling it would need the producer session's transcript, or the draft's Dropbox version history for the window from 13:39 to 16:17.

### X2. Input manifests

All 19 entries in A's manifest and all 20 in B's match the files on disk, including the outputs and scripts. Each script's import closure is hashed: `phase4_stageA_100T` → `phase4_stageA` → `noise_floor`.

I found no input that is used but unhashed. Three minor points:

- B hashes `phase4_stageC_grade.py`, which it does not import. It transcribes that file's SE constants.
- B does not hash `phase4_stageC_grade_100T.py`, the producer of `stageC_grading.json`. It does hash the JSON itself.
- Neither manifest records the Accelerate LAPACK build, which matters for §4.1.

### X3. No verdict language, and no interpretation

Neither output contains paper text or words such as establish, refute, confirm or significant. `"pass": true` in B and `"pass"` in A's `gate` are reproduction gates. The two exceptions to a clean result are already covered: A's `n_at_or_above_observed` (A6/S7) and the framing of B's ratio block (S7).

### X4. Basis labels (reported, not resolved)

1. `data/vectors/emotion_vectors_171_100T.pt` (sha `6de2e293…`) carries `variant: "neutral_HA"`, while its filename and every consumer treat it as 100T.
2. `gemma_grid/data/gemma-2-2b-base/vectors/canonical_100T.pt` carries `provenance: "re-cleaned 2026-07-31 against HA neutrals (M.4 re-lock)"` under a `canonical_100T` variant. The stage A wrapper that maps these paths names the mapping `HA_VEC` and describes its inputs as "HA-cleaned vectors". I checked only the 2b-base file's provenance string.
3. Each `c3_vectors.pt` carries `source: "100T neutral basis (k=3); raw reused from PA (story-derived, basis-invariant)"`.

I did not quote values from any superseded basis.

---

## 3. Supplementary charges, detail (written after the results existed)

### S2. Two definitions of the interaction contrast

My derivations: raw-r DiD (Δr27 − Δr2) = **+.04702**; tanh(Δz27) − tanh(Δz2) = **+.02109**; Δz27 − Δz2 = **+.02161**.

- **The registered family.** `PREREGISTRATION.md` l.167 defines the interaction in r units: "`([IT−base] arousal r)@27B − ([IT−base] arousal r)@2B`". That is the raw-r form.
- **`stageC_grading.json`.** It stores tanh(Δz27) − tanh(Δz2) as `delta` (`phase4_stageC_grade.py` l.163).
- **`interaction_did_constructions_100T.json`.** It holds all four constructions and records Adam's ruling of 2026-10-08: "paper_reports: A_raw_r_did", "stored_artifact_delta_stays: C_tanh_per_size_did", with the verdict unaffected.
- **The draft.** It prints the raw form: "+.047" and "about a third of the .15 floor" (l.287). .047/.15 = .31, whereas the tanh form would give .14.
- **Analysis B.** It holds the raw form as `interaction_arousal_r_raw` and the tanh form as `interaction_arousal_r_stageC_form`.

Each form therefore sits in the slot the 10-08 ruling assigns it: **SUSTAINED**.

The producer's statement that "their Fisher-z forms agree" carries no evidential weight. Both rows store the same object, `dz27 − dz2` (script lines 100 and 102 return the identical second element), so `point_z`, `boot_se_z` and both SE ratios are equal by construction.

The two r-scale forms are different quantities with different bootstrap distributions:

- raw form: SE .0288, 95% interval [−.0051, +.0988]
- tanh form: SE .0356, 95% interval [−.0366, +.0995]

### S3. The per-draw sign-flip rule

**Registration status.** The registered statistic (`noise_floor.compute`, l.94–95) flips each axis once, on the full 45. The **bootstrap** form, which flips on every draw, appears nowhere in `PREREGISTRATION.md`. It was pre-stated by the dated handoff (born 13:39:05, step 3: "applying `noise_floor.py`'s per-cell rule that flips each axis to agree with its own rating, exactly as the registered statistic does"). That predates both scripts. The on-diagonal form has an earlier precedent: `analysisC_g32_depaware_100T.py` (§5.3) takes |corr| on every draw. Carrying the per-draw flip into the **off-diagonal** rows, where it changes a row's sign rather than its magnitude, is new to this analysis. **Not registered; pre-stated by dated filing; partly precedented.**

**Counts (mine).** On the shared `default_rng` draws, the 2B-IT arousal axis (PC2) flips in **10 of 2,000** draws. No other cell's PC2 flips, and PC1 never flips in any cell. On an independent `RandomState` draw set, the count is 9 of 2,000. The full-sample arousal r for 2B-IT is .3630, the lowest of the six cells.

**The tail argument, tested.** The 10 flipping draws are exactly the 10 largest values of the 2B off-diagonal PC2-with-pleasure delta (+0.60 to +1.05; the next value is +.066). Ten is fewer than the 50 draws in a 2.5% tail, so those draws do not themselves form the endpoint. They do shift the upper order statistic by ten places.

| Row | Flip per draw (as run) | No per-draw flip (full-sample sign held) |
|---|---|---|
| PC2-pleasure 2B: boot SE r | .0760 | .0377 |
| PC2-pleasure 2B: boot SE z | .0813 | .0394 |
| PC2-pleasure 2B: 95% | [−.1574, −.0029] | [−.1574, −.0056] |
| PC2-pleasure 2B: 90% | [−.1424, −.0187] | [−.1424, −.0204] |
| PC2-pleasure pooled: boot SE r | .0333 | .0259 |
| PC2-pleasure pooled: 95% | [−.1390, −.0351] | [−.1390, −.0387] |
| PC2-pleasure pooled: 90% | [−.1319, −.0456] | [−.1319, −.0464] |
| SE ratio, 2B row (se_z / √(2/42)) | .3726 | .1806 |

What this shows:

- **The tail argument mostly defends the percentile intervals.** Lower endpoints do not move. Upper endpoints move by at most **.0036**, which is under the .0068 Monte Carlo spread.
- **It does not defend the SE.** The SE doubles on the 2B row and rises by 29% on the pooled row.
- **The SE propagates.** Quantities derived from that SE are reported in the output: `boot_se_r`, `boot_se_z`, `se_ratio_boot_over_registered_stageC` and `se_ratio_boot_over_handoff_stated`, on the 2B and pooled PC2-pleasure rows.

### S4. The 14 pairs the frozen script never ran

The "wrapper" is `pmap_embedding_table_transport_100T.py` itself:

- **New code, hashed.** It is hashed as `script_sha256` (`d79ca223…`), and that hash matches the file on disk.
- **Not frozen.** Nothing records a freeze, so it could be edited. Its last modification precedes the recorded run start by 5 s.
- **Frozen machinery unchanged.** It imports the frozen `eval_transport` unchanged.

**Does it reproduce the stored subset?** The producer reports a max |Δ| against the 16 stored pairs of 1.1e-16 in median cosine and 0 in rank, top-1 and top-5. My independent implementation against the stored artifact gives **2.6e-13**. Against the producer's recomputed values for all 30 pairs, including the 14 new ones, it also gives 2.6e-13. The 14 new pairs inherit that credibility. **SUSTAINED.**

### S5. Both runs were superseded, and only the later run left a record

**Analysis B.**

- The output file's **birth time is 13:43:46**, 16 s after the script's birth time (13:43:30). Its **modification time is 14:48:36**. The first output was written into this same file and later overwritten in place.
- The script was modified at 14:47:22, so the first run's code does not survive either.
- No copy of the first output exists anywhere in the project tree.

The claim that "only the flags differ" is therefore **UNTESTED: unverifiable from artifacts**. Settling it would need the Dropbox version history for `geometry_out_100T/tuning_dependence_aware_ci_100T.json` (the 13:43:46 revision) and `nogpu_additions_100T/tuning_dependence_aware_ci_100T.py` (the 13:43:30 revision).

**Analysis A.**

- The stopped run left no file I could find. Between 13:39 and 16:17 the project tree gained only the eight expected files: two scripts, two outputs, two manifests, the handoff and the commission. The output directory has no partial or temporary files.
- No producer process is alive now; the only Python workers running belonged to this audit.
- The manifest's wall time (5381.6 s) and generation time (16:16:48) put the start at 14:47:06, 5 s after the script's last save. That is consistent with one uninterrupted run of the recorded script.
- Whether stray workers from the stopped run were running alongside it cannot be established, and would affect wall time only.

The claim that the output comes "entirely from the rerun" is consistent with every artifact but cannot be **verified** from them: **UNTESTED**. Its content is right either way, because my independent recompute reproduces it (A4, S4).

### S6. Which vectors entered

**The mapping is not recorded as data, but hashed code determines it.**

- **Contextual vectors.** `ctx_paths = {c: REPO / SA.HA_VEC[c]}`. `HA_VEC` is built in `phase4_stageA_100T.py` (hashed, `001f59d3…`) from `phase4_stageA.CELLS` (hashed, `2c51e86e…`). For the five mode-A cells it gives `gemma_grid/data/<cell>/vectors/canonical_100T.pt`. For `gemma-2-27b-base` it gives `data/vectors/emotion_vectors_171_100T.pt`, whose hash the script asserts.
- **Static vectors.** These are `geometry_out_100T/<cell>/c3_vectors.pt`. The cell is identified by the directory name only.

My own code built the same mapping independently from the handoff and spec text. What else confirms it:

- Every file's d_model matches its size class.
- For contextual inputs, Analysis B's gate checks all six cells against their stage A blocks to 1e-6. A swap within a size class would have failed, since the cells' values differ (for example arousal .4992 for 2b-base against .3630 for 2b-it).
- For contextual inputs to A, Gate A covers 2b-base and 27b-base. The 16-pair reproduction covers the rest.
- **Static-table inputs: no gate or reproduction touches them.** Their base/IT assignment within a size rests on the directory name alone. The files carry no cell key: `vectors`, `vectors_raw`, `layer: embed_tokens`, `source`.

**Ruling on the input-manifest rule.** It is satisfied in letter: every input was hashed at run time beside the output. A file-to-cell mapping is not recorded as data, but it can be reconstructed from hashed code, so the mapping is recoverable from hashed artifacts.

The producer's argument that "the gate reproducing confirms those were the inputs" is weaker than this code route, for two reasons. Gate A covers two of the six contextual cells, and no gate covers the static vectors.

### S7. Whether the two amendments are complete

**A6 and B4 were run first (§1).**

**Amendment A is incomplete.** No p value, bar or pass/fail appears, except in the label's disclaimers and the script docstring (l.22–23), which mention "the §4.3 bar" only to deny comparing against it. But the output still states a count at or above the observed value, `"n_at_or_above_observed": 0` (output key `primary_static_null/n_at_or_above_observed`, script l.202). It places `"observed_top1_count": 171` beside a null maximum of 4 and ships all 500 null counts. Together these give p = 1/501 in one step. That is the comparison the amendment set out to remove, stated as zero.

**Amendment B is complete in wording.** No floor flag, bar or interval-against-floor field survives in the output, the manifest or the code. "Floor" survives only in the `amendment_B` disclaimer, in script comments (l.27–28, l.44) and in the module name `noise_floor`.

**The ratio block.**

- **Arithmetic.** On all four pooled rows, `se_ratio_boot_over_registered_stageC / se_ratio_boot_over_handoff_stated` = **1.7320508076** (and the inverse is 0.5773502692), equal to √3 to 10 decimal places. Per-size and interaction rows carry identical ratios by design, since both columns use the same SE there. Each ratio equals `boot_se_z / registered_se_z` exactly. **No arithmetic defect.**
- **Framing.** The draft states that the .15 floor *is* one independence-model Fisher SE, √(2/42) in z (l.67). `se_ratio_boot_over_handoff_stated` is the dependence-aware SE divided by exactly that SE. Multiplying .15 by the ratio produces a "dependence-aware floor", and putting the intervals beside it re-creates the comparison amendment B removed. Nothing in the output does this in words. The ingredients sit side by side, labelled "registered".
- **Flip inflation.** The 2B off-diagonal PC2 ratio (.3726) carries the flip-rule inflation from S3. Without the per-draw flip it is .1806.

---

## 4. Anything that struck me and fits no charge

### 4.1 The frozen Procrustes map is not unique, and held-out cosines depend on LAPACK

The frozen `fit_procrustes` takes `U @ Vt` from `np.linalg.svd(X_tr.T @ Y_tr, full_matrices=False)`. With 137 centered training emotions, the cross-covariance has **rank 136** (measured on fold 0 of the primary pair), against a minimum dimension of 2,304. The other 2,168 columns of U and rows of Vt are an arbitrary completion that LAPACK's routine chooses. Held-out source vectors lie outside the training span, so that arbitrary part enters every prediction.

The spec's words, "W … with orthonormal columns minimizing Frobenius error, closed-form via SVD", do not pin it down: every completion minimizes the training error equally.

I measured the effect on the primary pair:

- **Per-emotion cosines.** Switching the driver from gesdd (numpy's default, which the frozen code uses) to gesvd moves fold-0 held-out cosines at the third decimal, for example .97907 → .97899 and .92809 → .92717. The norm of the prediction difference is 9% of the prediction.
- **Median cosine.** The median is robust to the driver: .971624 under gesdd against .971671 under gesvd (contextual), and .9435 against .9437 (static).
- **Rank-restricted map.** Dropping the completion and using the rank-restricted map W_r = U_r V_rᵀ lifts every median cosine by about .01 to .02. The primary pair goes from .9716 to **.9804** (contextual) and from .9435 to **.9662** (static).
- **Ranks.** Median rank stays 1 and top-1 stays 1.000 under all three completions in every pair, with two exceptions: 2b-it > 27b-base contextual top-1 goes from .994 to .988, and 2b-it > 9b-base is unchanged at .994.

The printed §4.3 cosine (.972), the battery's cosines and these 30-pair tables are therefore properties of gesdd's completion as well as of the data. The producer's gate tolerance of 1e-9 was met across Accelerate and OpenBLAS. That shows the completion is reproducible between those two libraries, not that it is defined by the method. I am not proposing an analysis. The rank and top-1 endpoints that carry the §4.3 verdict are unaffected.

### 4.2 The commission's timing cannot be confirmed from artifacts

The commission says it "was written before either analysis had run". The filesystem records:

| Event | Time (local) |
|---|---|
| Handoff created | 13:39:05 |
| B script created | 13:43:30 |
| B first output created | **13:43:46** |
| A script created / modified | 14:47:01 |
| B script modified | 14:47:22 |
| A recorded run start | ≈14:47:06 |
| B final output written | 14:48:36 |
| **Commission file created** | **14:49:42** |
| A output and manifest | 16:16:48 |
| Draft v7 modified | 16:30:07 |
| Dispatch created | 16:30:53 |
| Ledger modified (LC62) | 16:31:27 |

Several of these times are relevant:

- A birth time resets when an editor saves by atomic rename, so this does not prove the commission's *text* was written after the results.
- The B output's birth time survived an in-place overwrite, which shows that at least that file was written in place.
- Both amendments are dated 2026-10-14. They appear in scripts saved at 14:47:01 and 14:47:22, before the commission file's birth.
- A6 and B4 target exactly the two things those amendments removed.

On the artifacts alone, then, the pre-commitment of A1 to X4 is **unverifiable**, and the separation into two tables in §0 rests on the commission's own statement. A record of when the commission was drafted would settle it, for instance the drafting chat's transcript or an earlier saved revision.

### 4.3 The amendments live only in the producer's files

The handoff was never amended. It still specifies the floor flags (l.79), the count at or above (step 4) and √(2/42) for the pooled SE. "Adam's amendment A" and "Adam's amendment B" are recorded only in the two scripts and the two outputs. The only other record is the Ledger entry written at 16:31, after both runs. No dated filing of the amendments predates the runs.

---

## 5. Supplementary charges I judge leading

None of the supplementary charges steered a finding. Two notes:

- **S1** tells the auditor not to conclude that a dependence-aware SE could move a registered tuning verdict, and gives reasons. I checked the reasons against the grading code rather than accepting them (§1, S1: `NULL_subfloor` is assigned before Holm). They hold. The instruction pre-states a conclusion, but it is a correct one and matches the commission's §7.
- **S5** predicts its own outcome ("S5 is the charge most likely to end that way"). That is mildly leading. The finding (UNTESTED) rests on the absence of any surviving first-run record, not on the prediction. The birth-time evidence that a first B output existed is my addition.

---

## 6. Environment

| Item | Value |
|---|---|
| Python | 3.12.13 (`.venv`), used for all computation. 3.9 (`lens_env`) was used only for JSON comparison and p arithmetic |
| numpy | 2.4.6, Apple Accelerate BLAS/LAPACK (macOS 26.6.2, arm64, 10 cores) |
| scipy | 1.17.1 (gesvd driver for the sensitivity arm) |
| torch | 2.12.0, CPU, used only to read `.pt` files |
| Transport precision | float64 throughout. The vectors are stored as float32 and cast. Procrustes uses `np.linalg.svd` (gesdd) for the procedure of record, plus gesvd and rank-restricted thin-QR arms |
| Analysis B precision | Vectors cast to float32 as the registered loader does. Gram matrix in float32, eigendecomposition in float64. A float64 arm also passes the gate (9.6e-9). Correlations in float64 |
| Parallelism | spawn worker processes, `VECLIB_MAXIMUM_THREADS=1` |
| Audit code | `aud_common.py`, `aud_A.py`, `aud_A_run.py`, `aud_B.py`, `aud_B_gate.py`, `aud_B_boot.py` and `aud_S1.py`, in the session scratchpad, with hashes recorded in `BLIND_*_sealed_at.txt` |
