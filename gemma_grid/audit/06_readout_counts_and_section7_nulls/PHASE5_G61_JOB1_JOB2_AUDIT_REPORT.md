# Phase-5 adversarial audit: the G6.1 regrade, job1 (§5.3 L18), job2 (§7.3 splits) (100T)

- **Date:** 2026-10-14
- **Auditor:** fresh-context Claude Code session (claude-opus-5-5). I had no access to the reasoning of the sessions that produced these numbers.
- **Prompt:** `gemma_grid/PHASE5_G61_JOB1_JOB2_AUDIT_PROMPT_100T.md` (sha256 `577b65a3b83148b36e9f71ad33d272890d057335e1f2cb47ec9625267dc18e0a`, 170 lines, verified before starting).
- **Overall verdict:** **Every number the paper prints from these three sources reproduces under independent code.** The G6.1 counts, the job1 L18 values (point estimates, bootstrap p values, registered-test p values) and the job2 splits, separations and permutation null all reproduce. The job1 values reproduce bit-for-bit and the job2 values to 1e-16. Two of the procedure fields are wrong in ways that matter:
  1. job1's `procedure` field gives the wrong formula for its p value (B4).
  2. job2's `procedure` field states a split-search range that does not reproduce the stored null. The code searches one position fewer (C4/C5).

  In both cases the stored values survive under what the code actually does. They do not survive under what the field says. A third finding fits no charge: the job1 artifact on disk was overwritten on 2026-10-14 by the reconstructed script, so its stored numbers are that script's output (§6.1).

## 0. Audit hygiene

- **Blind arm.** For each charge group I wrote my implementation from the artifact's stated inputs and procedure text, recorded the values in scratch, and only then read the stored results. The committed scripts and the helper routines they import (`gemma_grid/scripts/noise_floor.py`, `block_split` in `fixed_probe_tuning_shifts{,_100T}.py`) were opened only after that group's blind values were recorded.
- **Not run.** I did not run `job1_L18_emotion_bootstrap_sensitivity_100T.py`, `job2_layer_permutation_null_100T.py` or `g6_1_regrade_extract_100T.py`. I did not read the findings ledger, the paper draft, or any memo or dispatch about these jobs.
- **One protocol document read.** I read `coordination_100T/GGE1_100T_PRECOMMITMENTS.md` item 4, which is the pre-data rubric and protocol, for A5 and A6.
- **Values seen before derivation.** These are disclosed so that nothing read is reported as verified:
  - A structure dump of `G6_1_transcription_inputs.json` showed list lengths that equal the per-cell on-theme counts before I ran the A1 join. The per-emotion counts and their distribution were not exposed.
  - job1's `reproduction_gate` and `flag_base_depth_gap_sign_reversal` strings, which carry the L17 reference values and the B5 driver values to 7–8 digits, were read with the procedure field.
  - job2's `search_artifact_gap_note` gives "0.39-0.62" as the observed range.
  - The prompt supplies the paper's printed values by design.
  - The A1 and B/C derivations are deterministic code. The exposures above did not enter them.
- **Basis discipline (§3 of the prompt).** This report quotes no value, count, verdict or token from the superseded-basis rows. Where those rows had to be opened (A2, A5, A6), only the outcome of the check is stated.
- **Scratch.** All my code and outputs are under `/private/tmp/claude-501/…/scratchpad/phase5_g61_job1_job2/`:
  - `a1_blind.py`, `a2_a4.py`, `a6.py`, `a6_shuffle.py`
  - `b_job1_blind.py`, `b_compare.py`
  - `c_job2_blind.py`, `c_job2_range.py`
  - `x_cleanbasis.py`

## 1. Bottom line

| Charge | Verdict | Max abs diff (mine vs stored) |
|---|---|---|
| A1 counts §6.1 prints | SUSTAINED | 0 (integers) |
| A2 subset relation | SUSTAINED | 0 (72/72 rows identical) |
| A3 verdicts follow rubric | SUSTAINED | 0 violations; 9 rows at exactly 2/5 |
| A4 commitment | SUSTAINED | 4/4 hashes match |
| A5 rubric calibration | SUSTAINED (as attested; see caveat) | 0 (exact match) |
| A6 blinding | SUSTAINED, with one note | n/a |
| B1 point estimates | SUSTAINED | 0 (bit-identical, float64) |
| B2 p values (both families) | SUSTAINED | 0 (bootstrap); registered p not stored, matches printed rounding |
| B3 float64 question | SUSTAINED: float64 is needed | float64 0; float32 1.07e-7 (point estimates ≤1.19e-8, above the artifact's 5e-9) |
| B4 script vs procedure field | REFUTED: one undisclosed load-bearing step | n/a |
| B5 sign-reversal flag | SUSTAINED | ≤5e-8 vs the flag's 7-digit text |
| C1 splits and separations | SUSTAINED | 0 |
| C2 permutation null | SUSTAINED (under the code's range) | 1.1e-16 |
| C3 seeding convention | SUSTAINED: per-size reseeding reproduces; one shared generator does not | per-size 1.1e-16; shared 6.0e-3 |
| C4 script vs procedure field | REFUTED: one undisclosed load-bearing step | n/a |
| C5 split-search bounds | REFUTED as described; estimator and null are consistent with each other | literal-range null: 2.8e-2 on null stats, count 9 vs 1 |

## 2. Charge group A: the G6.1 regrade

**Inputs.** All files are in `gemma_grid/geometry_out_100T/g6_1_regrade/`. I joined `G6_1_grader_verdicts.json` (144 rows) to `G6_1_label_key_SEALED.json` (144 rows) on `row_id`. Cell and emotion agree on every row. The key has 72 rows per basis, and 72 rows carry the 100T basis.

### A1. Counts (blind, then stored)

My per-cell counts come from the join, restricted to the 100T rows:

| Cell | Mine | Stored (extract `emotions_on_theme_out_of_12_per_cell`) | Stored (unblind result, 100T) |
|---|---|---|---|
| gemma-2-2b-base | 10 | 10 | 10 |
| gemma-2-2b-it | 7 | 7 | 7 |
| gemma-2-9b-base | 9 | 9 | 9 |
| gemma-2-9b-it | 9 | 9 | 9 |
| gemma-2-27b-base | 10 | 10 | 10 |
| gemma-2-27b-it | 11 | 11 | 11 |
| **Total** | **56/72** | 56 | "56/72" |

My per-emotion counts, out of 6. Both the extract and `G6_1_transcription_inputs.json` store the same values.

| Emotion | Count | Emotion | Count |
|---|---|---|---|
| afraid | 5 | inspired | 5 |
| angry | 5 | loving | 5 |
| calm | 1 | nervous | 5 |
| desperate | 6 | proud | 6 |
| guilty | 2 | sad | 4 |
| happy | 6 | surprised | 6 |

The per-emotion total is also 56. Compared with what the paper prints:

- On-theme in all six cells: four emotions (desperate, happy, proud, surprised).
- Five of six: five emotions (afraid, angry, inspired, loving, nervous).
- Four of six: one emotion (sad).
- Two of six: one emotion (guilty).
- One of six: one emotion (calm).
- Per-cell range: 7 to 11.

Every printed count matches.

### A2. Subset relation

The extract has 72 rows with 72 unique (cell, emotion) keys. Its key set equals the key set of the 100T rows of the 144-row key exactly. For every row, `tokens` (each token string and its `related` flag), `related_count` and `on_theme` are identical to the grader's row. No row is added, dropped or substituted.

I also checked where each row's tokens came from. For all 144 rows, the packet's `top5` list, the grader's token list and `table1_raw_analysis_layer[emotion].top` in the matching basis's `lens_summary.json` are identical. So each extract row carries the token list of its own basis.

### A3. Verdicts follow the rubric

The rubric string in the grader file is "row ON-THEME iff >=2/5 related". I recomputed the related count from the five `related` flags of every row:

- Recomputed counts equal the stored `related_count` in 72/72 extract rows.
- `on_theme == (count ≥ 2)` holds in 72/72 extract rows. It also holds in all 144 grader rows.
- **Rows at exactly 2 of 5: 9.** By cell: 2b-base 2, 2b-it 1, 9b-it 2, 27b-base 2, 27b-it 2, 9b-base 0. By emotion: inspired 3, and afraid, calm, desperate, guilty, loving and surprised 1 each.
- For completeness, 11 rows sit at exactly 1 of 5, where one token judgement would add a row.

### A4. Commitment

All four `full_run_commitment_sha256` entries match the files on disk: the sealed key, the grader verdicts, the packet and the manifest. The cross-file hashes also hold:

- The packet's `sealed_key_sha256` matches the key.
- The manifest's packet and key hashes match.
- `G6_1_unblind_result.json`'s `sr5_consumed_sha256` (three files) match.
- The extract's six `inputs_sha256` and the manifest's and key's twelve input hashes match the `lens_summary.json` files on disk.

### A5. Rubric calibration (attestation)

Three things were checked:

1. The earlier count of the anchor cell (gemma-2-27b-base) is recorded in the pre-data protocol, item 4 of `GGE1_100T_PRECOMMITMENTS.md`, filed 2026-08-11. That record describes it as graded under an implicit rubric.
2. I counted the anchor cell's on-theme verdicts among the superseded-basis rows of the blind regrade under the frozen rubric. **That count equals the earlier recorded count exactly.**
3. `G6_1_unblind_result.json` records the same equality, and its reproduction flag is true.

The record's statement is therefore true.

**Caveat.** I could not check that the earlier count was made on the same token lists that the regrade showed the grader. No artifact I was permitted to read serializes the earlier grading's token lists. The match is between counts, not between row-level judgements.

### A6. Blinding

**Packet content.** Every row has exactly the fields `row_id, cell, emotion, top5`. There are no basis fields, no prior outcomes and no per-row metadata. JSON formatting is uniform: one indent level and identical layout on all 144 row lines. Token-level surface features are distributed comparably across the two bases: leading space, non-ASCII content, stray whitespace. Where a cell-emotion pair's two lists coincide, the pair carries no basis information either way. Token content can differ by basis by construction, and that is the object being graded.

**Ordering.** The file order equals `row_id` order (1–144), so `row_id` carries nothing beyond position. Three tests found no sign that position reveals basis:

- A runs test on basis by position gives z = −0.50.
- The first half of the packet holds 42 of the 72 100T rows, against 36 expected.
- Within each of the 72 cell-emotion pairs, the 100T row comes first in 37.

I reproduced the shuffle exactly. It is `random.Random(20260819).shuffle` applied to a canonical list:

- the superseded-basis block, then the 100T block;
- cells in grid order;
- emotions alphabetical.

So the order is a genuine seeded shuffle, as the key states.

**Rubric frozen pre-data.** Item 4 of the precommitments holds the rubric in full, dated 2026-08-11 and marked "pre-data". The file's mtime is 2026-08-11. The 100T `lens_summary.json` files date from 2026-08-19 11:51–11:53, the packet from 12:34 and the grader verdicts from 17:29. The grader file's rubric string matches item 4's threshold. These files are untracked in git, so the timestamps are filesystem mtimes only.

**Note (does not refute).** The packet says `seed_location: "sealed key only"`, but `G6_1_regrade_manifest.json` also carries the seed. The seed is also a calendar date, and the shuffle uses the Python standard library. A grader who had the manifest, or who guessed a date seed, could recover every row's basis. Nothing in the packet itself leaks. Whether the grader had access to the manifest cannot be tested from the artifacts.

## 3. Charge group B: job1 (§5.3 L18)

### Inputs

All nine `sr5_input_manifest` hashes match the files on disk. My code asserts them before computing.

| Manifest key | Resolved path | Tensor used |
|---|---|---|
| 2b-base / 2b-it `canonical_100T.pt` | `gemma_grid/data/<cell>/vectors/canonical_100T.pt` | `vectors` (171, 26, 2304) f32, index 18 (index 17 for the L17 reference) |
| 27b-base `emotion_vectors_171_100T.pt` | `data/vectors/emotion_vectors_171_100T.pt` (`6de2e293…`) | `vectors` (171, 46, 4608), index 31 |
| 27b-it `canonical_100T.pt` | `gemma_grid/data/gemma-2-27b-it/vectors/canonical_100T.pt` | `vectors`, index 31 |
| `<cell>/c3_vectors.pt` ×4 | `gemma_grid/geometry_out_100T/<cell>/c3_vectors.pt` | `vectors` (171, 1, d), cleaned embedding |
| `russell_mehrabian_1977.csv` | `data/ratings/russell_mehrabian_1977.csv` | `Arousal Mean`; 45/45 terms match emotion names (lower-cased) |

**My implementation, written from the procedure text only.** Center over the 171 rows, then SVD, then PC2 = column 1 of U·S, restricted to the 45 overlap rows. For each estimator, r = min(|corrcoef|, 0.999999) and z = arctanh. For bootstrap rounds I use `default_rng(0)` fresh per chain, with one `integers(0,45,45)` index per round shared by all four estimators. SE is `np.std` with ddof 0. The dependence-aware p is erfc(|delta/SE|/√2), and the registered p is erfc(|delta_z/√(2/42)|/√2).

### B1. Point estimates (blind, float64, then stored)

| Quantity | Mine | Stored | |diff| |
|---|---|---|---|
| base rise delta_r | 0.011727056525990931 | 0.011727056525990931 | 0 |
| base rise delta_z | 0.02086080936882495 | same | 0 |
| tuned rise delta_r | 0.1192069530752321 | same | 0 |
| tuned rise delta_z | 0.16375382777615877 | same | 0 |
| base gap_z 2B / 27B | 0.6598799722293158 / 0.6253957430408075 | same | 0 |
| base depth-gap delta | −0.03448422918850835 | same | 0 |
| tuned gap_z 2B / 27B | 0.3675077462673757 / 0.480226880703575 | same | 0 |
| tuned depth-gap delta | 0.1127191344361993 | same | 0 |
| 2b-base arousal r at L18 / L17 | 0.6557976582891021 / 0.4992351004 | not stored as fields (flag text: 0.6557977 / 0.4992351) | <5e-8 |
| L17 reference (8 values) | e.g. base 0.16828961444495638, 0.2026331998843453 | same | 0 |

Against the printed values: base rise +.012 ✓, tuned rise +.119 ✓, base gap −.034 ✓, tuned gap +.113 ✓, 2b-base arousal .656 at L18 against .499 at L17 ✓.

### B2. p values

| | Mine | Stored | Printed |
|---|---|---|---|
| base rise, bootstrap | 0.5020729441182125 | identical | .50 |
| tuned rise, bootstrap | 0.00041333398224753347 | identical | .0004 |
| base gap, bootstrap | 0.5476468752113544 | identical | .55 |
| tuned gap, bootstrap | 0.08086928512814104 | identical | .08 |
| base rise, registered z = Δz/√(2/42) | 0.923841 | not stored | .92 ✓ |
| tuned rise, registered | 0.453005 | not stored | .45 ✓ |

The bootstrap SEs also match bit-for-bit: 0.03107840363585239, 0.0573504156986378, 0.046370525569365474 and 0.06457090758002589.

### B3. Float64 versus float32

In my float32 arm the SVD runs in float32. PC2 is then cast to float64 for the Pearson step.

| Precision | Max abs diff vs stored |
|---|---|
| float64 (cast before PCA) | **0 on all 26 stored values** |
| float32 point estimates | 1.19e-8 (L17 tuned gap delta). Also above 5e-9: 8.2e-9 (tuned L18 gap delta), 7.0e-9, 6.1e-9 and 5.8e-9 |
| float32 SE | ≤3.7e-9 |
| float32 p values | ≤1.07e-7 |

The artifact's own tolerance (`reproduction_gate`) is ≤5e-9 on every quantity. **Float32 does not reproduce within that tolerance on five point estimates**, so the cast is load-bearing at the artifact's own standard. It is not load-bearing at any printed precision: every float32 value rounds to the same printed digits.

There is one piece of independent evidence, and it survives from before the overwrite described in §6.1. The `reproduction_gate` text gives the L17 base depth-gap delta as "0.20263320". Float64 rounds to 0.20263320. My float32 arm rounds to 0.20263319. So the original run's own prose favours float64 over this float32 convention.

### B5. Sign-reversal flag

The flag states four numbers, all checked against my blind values:

- Delta of −0.0344842 at L18. Mine is −0.03448423.
- +0.2026332 at L17. Mine is +0.20263320.
- 2b-base arousal r of 0.4992351 at L17. Mine is 0.49923510.
- 2b-base arousal r of 0.6557977 at L18. Mine is 0.65579766.

The 2B gap at L18 (0.6599) exceeds the 27B gap (0.6254), as stated. The named driver is forced by the design. Between the L17 and L18 runs, the only input that changes is the 2B analysis-layer PC2. The 27B r values and both embedding r values are bit-identical across the two, so the reversal can only come from r_an_2B. The flag's arithmetic and driver are both correct.

## 4. Charge group C: job2 (§7.3 splits)

**Input.** `fixed_probe_tuning_shifts_100T_results.json` (sha256 `4d7fe9ea…b318a`, matches the manifest) gives `[size].layer_corr_matrix`. The matrices are 26×26, 42×42 and 46×46, symmetric, with unit diagonal, and are used as stored (3 dp).

**My implementation, written from the procedure text.** Block A is layers [0, s) and block B is [s, L). "Within" is the mean of the pooled off-diagonal entries of both diagonal blocks, and "cross" is the mean of C[A, B]. I searched s ∈ [2, L−2] **inclusive**, which is the literal reading of the field. The other reading of "within" is the average of the two block means. It does not reproduce the observed values: it gives 2B .659, and splits at 40 (9B) and 12 (27B).

### C1. Splits and separations (blind, then stored)

| Size | My split | My gap (within, cross) | Stored | |diff| |
|---|---|---|---|---|
| 2B (L=26) | 23 | 0.619560348731884 (0.5286328125, −0.0909275362) | 23, identical | 0 |
| 9B (L=42) | 27 | 0.38632202729044846 (0.5250701754, 0.1387481481) | 27, identical | 0 |
| 27B (L=46) | 16 | 0.6052912725225226 (0.6543891892, 0.0490979167) | 16, identical | 0 |

Printed: .620 at layer 23 of 26, .386 at 27 of 42, .605 at 16 of 46. All match. The input file's own `two_block` fields (4 dp) agree as well.

The observed split is the same under every search range I tried that contains it: [2, L−2], [2, L−3], [1, L−2] and [1, L−3]. Under [2, L−4] the 2B split cannot be reached, because 23 = L−3 is the top of the code's range.

### C2. Permutation null

Under my literal inclusive range [2, L−2], the null **did not reproduce**. I got a 2B count of 9 against a stored 1, and the null mean, median and p95 were off by up to 2.8e-2 at all sizes. Only `null_max` matched at all three sizes. That is the cue for C5.

Under s ∈ [2, L−3], which is the code's range (see C5), with B = 2000, seed 20261007 and `rng.permutation(L)`:

| Size | Count ≥ obs | p (add-one) | null max | null mean | null median | null p95 | Max |diff| vs stored |
|---|---|---|---|---|---|---|---|
| 2B | 1 | 0.000999500 | 0.682915012 | 0.152411050 | 0.132515199 | 0.368390192 | 1.1e-16 |
| 9B | 0 | 0.000499750 | 0.321525352 | 0.070011506 | 0.061038864 | 0.159050469 | 5.6e-17 |
| 27B | 0 | 0.000499750 | 0.415638728 | 0.121580470 | 0.113298192 | 0.262584263 | 5.6e-17 |

These are per-size reseeded values (see C3). At 2B the null maximum exceeds the observed gap, which is consistent with the count of 1.

### C3. Seeding convention: positive finding

I ran both conventions, with sizes processed in the order 2B, 9B, 27B:

| Convention | 2B | 9B | 27B |
|---|---|---|---|
| **Reseed `default_rng(20261007)` per size** | reproduces (1.1e-16) | reproduces (5.6e-17) | reproduces (5.6e-17) |
| One generator across all three sizes | reproduces (2B is first, so the stream is identical) | **does not**: mean 0.070836, median 0.061559, p95 0.156243 (max diff 2.8e-3) | **does not**: mean 0.119003, median 0.107618, p95 0.256571 (max diff 6.0e-3) |

**Per-size reseeding is the convention that reproduces the stored values. A shared generator does not.**

Two limits on how far this evidence reaches. First, counts and `null_max` are the same under both conventions, so only mean, median and p95 discriminate. Second, the discrimination is clean because the job2 artifact on disk is the original 2026-10-07 output: its `generated_utc` and its mtime both date from 2026-10-07. This test is therefore a genuine test against the original run, which is not the case for job1 (§6.1).

### C5. Split-search bounds

**What each source says:**

- **The artifact says** "best contiguous split s in [2,L-2]".
- **The code does** `for s in range(2, L - 2)`. That means s ∈ {2, …, L−3}. It holds in the cited `block_split` (`fixed_probe_tuning_shifts.py` and `fixed_probe_tuning_shifts_100T.py`, identical) and in the job2 script. The smallest block allowed is 2 layers at the start and 3 layers at the end.
- **My implementation** first searched the literal range [2, L−2], which includes the 2-layer final block. That range does not reproduce the stored null. Only [2, L−3] does. The field's notation is either wrong or an unstated half-open interval, and the stored null depends on it.
- **The estimator and the null use the same routine and range**, so here the null is not wider than the production estimator. The known failure mode named in the prompt is absent.

**Widened by one position at each end** relative to the code, to s ∈ [1, L−2]:

| Size | null max | null mean | null p95 | count ≥ obs | p (add-one) |
|---|---|---|---|---|---|
| 2B | 0.682915 (unchanged) | 0.194376 (+0.042) | 0.545800 (+0.177) | **87** (stored 1) | 0.04398 (stored 0.00100) |
| 9B | 0.321525 (unchanged) | 0.086017 (+0.016) | 0.230933 (+0.072) | 0 | 0.00050 |
| 27B | 0.415639 (unchanged) | 0.151241 (+0.030) | 0.325559 (+0.063) | 0 | 0.00050 |

**The null maximum does not move at any size.** The rest of the 2B null is sensitive to the bounds: allowing 1-layer blocks moves the 2B count from 1 to 87. Widening only the top end (the literal field reading) gives a 2B count of 9 and p = 0.0050. Narrowing to [3, L−3] lowers the null max (2B 0.5614, 9B 0.2908, 27B 0.3884). The observed separations and splits do not depend on any of these choices.

## 5. Undisclosed steps (B4, C4, C5 together)

### job1: script compared with the artifact's `procedure` field

| # | Step in script | In `procedure`? | Stored values survive without it? |
|---|---|---|---|
| 1 | float64 cast before PCA | No (disclosed inference) | At printed precision, yes. At the artifact's own 5e-9 tolerance, **no** (B3). |
| 2 | **Dependence-aware p = erfc(\|delta / SE\| / √2)** | **No. The field says "p = erfc(\|delta_z\|/sqrt(2))", with no division by SE.** | **No.** Applied literally, the field gives p = 0.983 (base rise) and 0.870 (tuned rise), against stored 0.502 and 0.000413. Only the /SE form reproduces. The field misstates its own p formula, while the script docstring states it correctly. For the depth gap the script uses `delta`, not `delta_z`, which the field's wording also does not cover. |
| 3 | Resample index `rng.integers(0, n, n)` | Not stated ("one shared resample index per round") | Draw method is load-bearing for exact SEs. I verified that `rng.choice(45, 45, replace=True)` gives the identical stream, so the two most natural readings agree. Other draw schemes would change the SEs at Monte-Carlo scale. |
| 4 | Overlap rows ordered by the emotion list | Not stated | Not load-bearing here. The emotion list and the CSV are both alphabetical, so the two orders coincide. I checked this and got identical SEs. |
| 5 | Layer = tensor index (18, 31); `vectors` key (cleaned) for both analysis-layer and `c3` embedding vectors; `Arousal Mean` column | Implied by "L18", "cleaned" (`basis` field) and "arousal" | Natural readings, all of which my blind code took independently. |
| 6 | `nf.pca(V, 5)` truncation to 5 PCs | Not stated | Irrelevant, because only PC2 is used. |

### job2: script compared with the artifact's `procedure` field

| # | Step in script | In `procedure`? | Stored values survive without it? |
|---|---|---|---|
| 1 | Per-size reseeding | Not stated (disclosed inference) | **No** for 9B and 27B null mean, median and p95 (C3). Yes for the observed values, counts and null max. |
| 2 | **Split range `range(2, L-2)`, i.e. s ≤ L−3** | **Misstated as "[2,L-2]"**, and the job2 script's own docstring repeats "[2, L-2]" | **No** for the null under the literal reading: 2B count 9 against 1, p .0050 against .0010, null stats off by up to 2.8e-2. Yes for the observed values (C1) and the null max. |
| 3 | Pooled "within" (upper-triangle entries of both blocks concatenated, then a single mean) | Ambiguous in the field. Disclosed by citing `block_split` "verbatim". | No. The average-of-block-means reading changes the observed values (2B .659; 9B split 40). The cited routine settles it, and the stored values match it. |
| 4 | Permutation drawn as `rng.permutation(L)` | Not stated | Load-bearing for exact null stats. It is the natural reading and reproduces exactly. |
| 5 | `np.percentile(…, 95)` with default linear interpolation | Not stated | Load-bearing for `null_p95` only, at the 1e-3 scale. |
| 6 | Size order 2B → 9B → 27B | Not stated | Matters only under a shared generator, which does not reproduce anyway. |

## 6. Findings that fit no charge

1. **The job1 artifact was overwritten by the reconstructed script.**
   - **Timestamps.** `job1_L18_emotion_bootstrap_sensitivity_100T.json` has `generated_utc` 2026-10-14T14:31:47Z and mtime 2026-10-14 10:31 local. The script's mtime is 10:30 local, and the script has a `--write` mode that replaces `results`, `sr5_input_manifest` and `generated_utc` while keeping the prose fields.
   - **Consequence.** The run of record is 2026-10-07, so the numeric results on disk are the 2026-10-14 script's output. "The script reproduces the stored values" is partly circular for job1, and B3's "float64 reproduces exactly" is a statement about that script's output.
   - **What is independent.** My blind implementation, written only from the procedure text, matches that output bit-for-bit. The surviving 2026-10-07 prose (L17 values to 8 digits, flag values to 7) agrees with it, and on one value it favours float64 over float32 (B3).
   - **What would close it.** An original 2026-10-07 copy of the artifact from Dropbox version history or a backup, compared at full precision.
   - **Not affected.** job2's artifact is untouched (2026-10-07), and the G6.1 grader, key, packet and manifest are untouched (2026-08-19).
2. **The provenance labels on job1's inputs are cosmetic, verified for the four analysis-layer inputs.** `canonical_100T.pt` carries a `provenance` string, and `emotion_vectors_171_100T.pt` a `variant` label, that both name the superseded neutral set. job1 uses the *cleaned* `vectors`, so this matters for job1, unlike for raw-vector analyses. I tested each input by projecting `vectors_raw` out of each stored per-layer cleanup basis at the layer job1 reads:

   | Input and layer | Max \|cleaned − projection\| using the 100T basis | Using the superseded basis |
   |---|---|---|
   | 2b-base L18 | 2.1e-7 | 2.17 |
   | 2b-base L17 | 2.0e-7 | 2.13 |
   | 2b-it L18 | 1.9e-7 | 4.08 |
   | 27b-it L31 | 8.0e-6 (scale 211) | 102 |
   | 27b-base L31 | 2.5e-4 (scale 166) | 68 |

   The 100T bases are `gemma_grid/data/<cell>/vectors/<cell>__cleanup_basis_100T.pt`. The cleaned vectors are the 100T-basis projection to float32 precision. I did not test the `c3_vectors.pt` embedding cleaning, whose `source` field reads "100T neutral basis (k=3)".
3. **G6.1 seed placement.** Covered in A6: the manifest also carries the seed, and the shuffle is reproducible from a date seed with the Python standard library.
4. **Scope of the commitment.** The extract's `full_run_commitment_sha256` covers four files. `G6_1_unblind_result.json` and `G6_1_transcription_inputs.json` are not committed by it, although each carries its own hashes of its inputs, and those check out. None of the paper's §6.1 numbers depend on them, because I re-derived the numbers from the committed files.

## 7. Environment

- **Charge A** (JSON only, no numerics): Homebrew `python3` 3.14.7, standard library only. `random.Random` was used for the shuffle reproduction under `.venv`.
- **Charges B and C:** `.venv/bin/python`, Python 3.12.13, numpy 2.4.6, torch 2.12.0 (used only to `torch.load` the `.pt` inputs). BLAS and LAPACK are Apple Accelerate (`numpy.show_config`). `lens_env` was not used.
- **Precision:**
  - job1's primary arm is float64 from the PCA onward, after casting the stored float32 tensors. The float32 arm runs the SVD in float32 and casts PC2 to float64 for the Pearson step.
  - job2 is float64 throughout.
- **Accelerate warnings.** The known spurious matmul RuntimeWarnings were suppressed. Every result checked is finite.
