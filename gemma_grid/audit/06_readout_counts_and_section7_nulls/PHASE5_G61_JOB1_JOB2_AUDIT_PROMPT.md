# Phase-5 adversarial audit: the G6.1 regrade, job1 (§5.3 L18), job2 (§7.3 splits)

**To:** a fresh Claude Code session with no history on this project
**From:** the GGE1 100T Ledger Chat, on Adam Hollowell's instruction, 2026-10-14
**Deliverable:** one report, `PHASE5_G61_JOB1_JOB2_AUDIT_REPORT_100T.md`, written to `gemma_grid/`.
**You are not editing anything.** Do not modify any artifact, script, draft or ledger file.

---

## 1. Why this audit exists

Three sources of numbers that the paper prints have never been through a Phase-5 pass. Every other
100T audit either predates them or covers different objects, which was established by exclusion
across all six 100T audit reports. The paper's §8.4 states that every claim carrying a graded
verdict passed an automated adversarial audit protocol, and these three are the gap in that claim.
This audit closes it or tells us it cannot be closed.

Two of the three have a specific hazard. The scripts for job1 and job2 were written **after** their
artifacts existed, from each artifact's own prose description of its procedure, because no script
was retained from the original runs. A script written to match stored numbers can encode choices
that were reverse-engineered rather than recorded. Two such choices are already disclosed: a
float64 cast in job1 and per-size generator reseeding in job2. **Your most valuable charge is to
look for undisclosed others.** See charge B4 and C4.

## 2. The standard you work to

This project's audit standard, which you are held to:

- Fresh context. You have no access to the reasoning of the sessions that produced these numbers.
- **You write your own code.** Do not run the committed producer scripts to obtain any number you
  then report as verification. Re-derive from the artifacts' stated inputs with an implementation
  you write yourself, working outside the producing pipeline.
- **Blind arm first, for every numeric claim.** Derive the quantity, write your value down in the
  report, and only then open the artifact or this prompt's claimed value to compare. The report must
  show your blind value and the stored value side by side. A number you read before deriving it is
  reported as read, not as verified.
- Your job is to refute. A claim that survives a real attempt to break it is sustained. A claim you
  could not test is reported as untested, not as sustained.
- You re-grade nothing. This audit sustains or refutes, and it does not change any verdict, tier or
  floor.

## 3. Basis discipline, which constrains how you write, not what you check

This study's canonical corpus basis is called 100T. Earlier superseded bases exist and the released
paper does not discuss them. For charge A you must open a file that contains rows from a superseded
basis, because verifying the subset relation requires it.

**You may read it. Your report must not quote any value, count, verdict or token from those rows.**
Write the attestation so that it carries the finding without the data: for example "the 72 extracted
rows are exactly the 100T subset of the 144-row key, verified row by row" is fine, and any sentence
that reports what the other 72 rows contained is not. The report is intended to ship publicly, so a
report that quotes withheld data cannot be used and the audit would have to be rerun.

## 4. Charge group A, the G6.1 regrade

Artifacts, all under `gemma_grid/geometry_out_100T/g6_1_regrade/`:
`G6_1_verdicts_100T.json` (the public extract), `G6_1_grader_verdicts.json`,
`G6_1_label_key_SEALED.json`, `G6_1_unblind_result.json`, `G6_1_grading_packet.json`,
`G6_1_regrade_manifest.json`.

**A1. The counts the paper prints.** §6.1 prints, for the twelve emotions across six cells: four
emotions read on-theme in all six cells, five more in five of six, one in four, one in two, one in
one. Between 7 and 11 of the twelve read on-theme per cell. Derive every one of these from the
grader verdicts joined to the sealed key, restricted to the 100T rows. Report your per-cell counts,
your per-emotion counts, and their total, blind first.

**A2. The subset relation.** Confirm that the 72 rows in the public extract are exactly the 100T
rows of the 144-row key, with no row added, dropped or substituted. This is the charge the extract
cannot evidence about itself.

**A3. The verdicts follow the rubric.** The rubric requires at least two of a row's top five tokens
to be emotion-related. For every one of the 72 rows, check that the recorded `on_theme` verdict is
what the rubric's threshold gives for that row's token list and `related` flags. Report any row where
it is not, and report how many rows sit at exactly 2 of 5, since those are the rows where a single
token judgement flips a count.

**A4. The commitment.** The extract carries `full_run_commitment_sha256`. Verify it against the files
it commits to.

**A5. The rubric's calibration.** The record states that the rubric was validated against an earlier
count of the same anchor cell, made under a different rubric, and reproduced it exactly. Verify that
this is true. Report it as an attestation under the constraint in section 3: that the calibration was
checked and what its outcome was, without the number or the basis.

**A6. The blinding.** The record states the grader saw 144 rows with the labels withheld, in
randomized order, under a rubric frozen before the data existed. Check the packet and the manifest
for anything inconsistent with that, including whether the packet could have revealed a row's origin
through ordering, formatting, field presence or token content.

## 5. Charge group B, job1, behind §5.3's L18 values

Artifact `gemma_grid/geometry_out_100T/job1_L18_emotion_bootstrap_sensitivity_100T.json`.
Committed script `gemma_grid/scripts/job1_L18_emotion_bootstrap_sensitivity_100T.py`.

**B1. Re-derive the point estimates.** With your own implementation, from the artifact's stated
inputs and its nine asserted input hashes, re-derive the L18 arousal rise and depth gap in both
chains. The paper prints: the base-chain rise falls to +.012 and the tuned to +.119. The depth gap
reverses sign in the base chain to −.034 and falls to +.113 in the tuned chain. 2b-base arousal
reads .656 at L18 against .499 at L17. Read full precision from the artifact only after writing your
own values down.

**B2. Re-derive the p values.** Two families are quoted. The dependence-aware bootstrap p values
(.50, .0004, .55, .08 as the paper rounds them) are stored in the artifact. The registered-test p
values the paper prints as .92 and .45 are **derived**, not stored: z = delta_z / sqrt(2/(45-3)) and
p = erfc(|z|/sqrt(2)). Re-derive both families.

**B3. The float64 question.** The script makes an explicit float64 cast on vectors stored as float32.
That choice was inferred from the stored values and is recorded nowhere else. Run your own
implementation both ways and report which precision reproduces the stored values and by how much the
other misses. If float32 also reproduces them within the artifact's own tolerance, say so, because
then the cast is not load-bearing and the script asserts a necessity that does not exist.

**B4. Read the script against the artifact's procedure field.** List every step the script performs
that the artifact's `procedure` field does not state. The float64 cast is one and is disclosed. Report
any others, and for each say whether the stored values could be reproduced without it.

**B5. The sign-reversal flag.** The artifact flags a base-chain depth-gap sign reversal at L18 and
names its driver. Verify the flag's arithmetic and its stated driver.

## 6. Charge group C, job2, behind §7.3's splits

Artifact `gemma_grid/geometry_out_100T/job2_layer_permutation_null_100T.json`.
Committed script `gemma_grid/scripts/job2_layer_permutation_null_100T.py`.

**C1. Re-derive the splits and separations.** The paper prints separations of .620 at 2B, .386 at 9B
and .605 at 27B, with the split at layer 23 of 26 at 2B, layer 27 of 42 at 9B and layer 16 of 46 at
27B. Re-derive all six numbers with your own implementation of the best-contiguous-split search, from
the per-size layer-correlation matrices the artifact names, asserting its input hash first. Blind
first.

**C2. Re-derive the permutation null.** B = 2000, seed 20261007, relabel the layers and rerun the
split search. Report the null maximum per size and the count at or above the observed gap.

**C3. The seeding question, and this is a positive finding we want either way.** The artifact's
procedure field states the seed and is **silent** on whether one generator runs across all three
sizes or is reseeded per size. The script reseeds per size. **Run both conventions and report which
reproduces the stored values and which does not.** Reproduction is the only evidence that exists for
this choice, so establishing it cleanly is worth more here than anywhere else in this audit.

**C4. Read the script against the artifact's procedure field**, as in B4. Per-size reseeding is the
one disclosed inference. Report any others.

**C5. The split search's bounds.** The artifact names `block_split` as its search routine. Confirm the
range of split positions your implementation searches matches the one the artifact describes, and
report the effect on the null maximum if the range is widened by one position at each end. A related
script in this project was once found to search a wider range than the production estimator, so this
is a known failure mode and not a hypothetical.

## 7. What not to spend the pass on

Do not re-grade, re-tier or re-floor anything. Do not evaluate whether these analyses were worth
running or whether the paper's framing of them is apt. Do not comment on prose. Do not propose new
analyses. If you believe a claim is unverifiable from the artifacts you have, say so plainly and name
what would be needed, and that is a finding rather than a failure.

## 8. The report

House form, matching the other Phase-5 reports in `gemma_grid/`:

- A bottom-line table: one row per charge, with SUSTAINED, REFUTED, or UNTESTED, and the largest
  absolute difference between your value and the stored one.
- Per-charge detail, each showing your blind-derived value before the stored value.
- A section for charges B4, C4 and C5 together: every undisclosed step you found, and for each,
  whether the stored values survive without it. This is the section we most need.
- Anything that struck you and fits no charge. Several of this project's most useful audit findings
  arrived that way.
- Your environment: python, numpy, BLAS, and whether you used float32 or float64 where it mattered.

If everything sustains, say so plainly. A clean report from a real attempt to break three unaudited
claims is the result we are hoping for and it is worth more than a padded one.
