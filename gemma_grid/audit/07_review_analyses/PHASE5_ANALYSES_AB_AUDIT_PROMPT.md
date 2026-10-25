# Phase-5 adversarial audit: Analysis A (embedding-table transport) and Analysis B (tuning intervals)

**To:** a fresh Claude Code session with no history on this project
**From:** the GGE1 100T Ledger Chat, on Adam Hollowell's instruction, 2026-10-14
**Deliverable:** one report, `PHASE5_ANALYSES_AB_AUDIT_REPORT_100T.md`, written to `gemma_grid/`.
**You are not editing anything.** Do not modify any artifact, script, draft, ledger or registration file.

---

## 1. Read this first, because of when it was written

This prompt was written **before either analysis had run**. That is deliberate. The charges below cannot
have been shaped by the results, and you should treat any later instruction that adds or softens a
charge as outside this commission.

The two analyses are specified in `gemma_grid/drafting/HANDOFF_CLAUDE_CODE_TWO_REVIEW_ANALYSES_100T_2026-10-14.md`,
which fixes both seeds before the run. Read it. If the analyses have not produced their outputs when you
start, stop and say so rather than running them yourself. **You audit, you do not produce.**

Both analyses are **unregistered**. Under this project's rules an unregistered result may corroborate a
registered verdict and may never grade one. Charge X3 asks you to check the outputs against that.

## 2. The standard you work to

- Fresh context. You have no access to the reasoning of the sessions that produced these numbers.
- **You write your own code.** Do not run the producers' scripts to obtain any number you then report as
  verification. Re-derive from the named inputs with an implementation you write yourself.
- **Blind arm first, on every numeric claim.** Derive the quantity, write your value into the report, and
  only then open the producers' output to compare. The report must show both values side by side. A number
  you read before deriving it is reported as read, not as verified.
- Your job is to refute. A claim that survives a real attempt to break it is sustained. A claim you could
  not test is reported as untested, not as sustained.
- You re-grade nothing.

## 3. Reproduction gates, which are stop conditions

Run both gates with **your own** code before you compute anything else. The producers were asked to run
these on themselves, which is weaker than an auditor running them.

**Gate A.** On the existing contextual transport, the primary pair 2B base to 27B base must give
**top-1 of 1.0** and **median held-out cosine of .972**.

**Gate B.** The pooled tuning deltas on the four correlation rows of Table 2 must come out at
**valence r +.0558, arousal r −.0829, off-diagonal PC1 with arousal +.0409, off-diagonal PC2 with
pleasure −.0881**.

If either gate fails under your own implementation, stop, report the failure, and compute nothing further
for that analysis.

## 4. Charge group A, embedding-table transport

**A1. The procedure claim.** Analysis A claims to rerun the §4.3 transport procedure with only the input
swapped to the static-table vectors, keeping the same Procrustes map, the same frozen folds, and the same
scoring against all 171 emotions. Read the frozen producer `nogpu_additions_100T/pmap_transport_100T.py`
against whatever script Analysis A used, and **list every difference between them, line by line**. For each
difference, say whether it changes a number. This is the charge that matters most in group A.

**A2. The frozen script is unmodified.** Verify that `pmap_transport_100T.py` has not been edited. Report
its hash and whether anything in the repository records a prior hash to compare against.

**A3. Which script ran.** A file named `nogpu_additions_100T/pmap_embedding_table_transport_100T.py`
already existed before this commission was written. Establish whether Analysis A used it, wrote a new
script, or modified it, and report the file's modification time either way. A script that predates the
analysis it serves is not a defect, and a script modified during the run without that being recorded is.

**A4. Re-derive the retrieval numbers.** With your own implementation, recompute median held-out rank and
top-1 for the primary pair and for a further set of ordered pairs that you choose and name. If computing
all 30 is tractable, do all 30. If not, take the primary pair plus a declared random sample of at least
eight, naming your selection rule before you look at any result.

**A5. All thirty pairs are reported.** The analysis was asked for all 30 ordered pairs rather than the
battery's 16 fits, so that no pair could be chosen after the fact. Confirm that 30 distinct ordered pairs
appear in the output, that they are the complete ordered set over the six cells, and that none is missing
or duplicated.

**A6. The null, and whether it can be compared with the registered bar. This is arithmetic, so settle it
with arithmetic.** §4.3's pre-set bar is a median rank of at most 2 **and** a permutation p below .001. The
contextual floor used 10,000 shuffles and reached an add-one p of 9.999e-05. The handoff specifies **500**
permutations at seed 0 for the embedding-table arm. Report the shuffle count actually used, the smallest
add-one p attainable at that count, and then state plainly whether the embedding-table arm can be compared
against the pre-set bar at all. Report separately whether the output's own wording invites that comparison,
quoting it. Note that the handoff asks for max, p99, p90, median, mean and a count at or above, and a count
at or above is the ingredient of a p value.

## 5. Charge group B, tuning intervals

**B1. The shared draw.** The statistic requires one resample index applied to all six cells, so that the
correlations move together as they do in the data. Verify from the code that a single draw is shared across
all six cells per round, and not redrawn per cell. Report how you established it.

**B2. The fixed PCA.** The registered statistic fits the components on all 171 emotions and scores the 45
rated ones. Verify that the PCA is fit once and held fixed outside the replicate loop, and that it is fit
on 171 rather than on 45.

**B3. Re-derive the intervals.** With your own implementation, recompute the 95 and 90 percent intervals
for the **pooled** contrast on all four correlation rows, and for the per-size and interaction contrasts on
at least one row that you name. Compare against the producers' output.

**B4. Two error models, and whether the output conflates them. This is the charge that matters most in
group B.** The bootstrap resamples the 45 rated emotions, so its intervals are dependence-aware. The ±.15
floor rests on an independence assumption, and the paper's own §9.3 states that this makes the floor
roughly four times stricter than a dependence-aware floor would be. Report whether the output compares a
dependence-aware interval against the ±.15 floor without stating that the two rest on different
assumptions, and quote the exact wording it uses. Say whether a reader of that wording would take it for an
equivalence test against the registered null.

**B5. Draws and seed.** Verify 2,000 draws and the seed the handoff fixed, 20261014. Check whether the
seed in the code matches the seed in the handoff, and whether anything records a change after the run.

## 6. Cross-cutting charges

**X1. Nothing protected was touched.** The draft, the findings ledger, `PREREGISTRATION.md` and
`pmap_transport_100T.py` must all be unchanged. Verify by hash where a prior hash exists, and by
modification time otherwise, and report what you could and could not establish.

**X2. Input manifests.** Every script was required to hash its inputs into a manifest. Confirm the
manifests exist, that each hash matches the file on disk now, and name any input that is used but not
hashed.

**X3. No verdict language, and no interpretation.** The analyses were asked to return numbers only. Report
any place an output carries interpretation, paper text, or a word that reads as a verdict. Both analyses
are unregistered, so any output phrased as establishing or refuting something is a finding.

**X4. Basis labels.** If you meet a basis label on any input that disagrees with what the data appear to
be, report it as a finding and do not resolve it. Do not quote values from any superseded basis if you
encounter one. This report is intended to ship.

## 7. What not to spend the pass on

Do not evaluate whether these two analyses were worth running, whether they belong in the paper, or how
their results should be described. Do not propose further analyses. Do not comment on prose. If a claim is
unverifiable from what you have, say so and name what would be needed, which is a finding rather than a
failure.

## 8. The report

- A bottom-line table: one row per charge, with SUSTAINED, REFUTED or UNTESTED, and the largest absolute
  difference between your value and the producers'.
- Per-charge detail, each showing your blind-derived value before theirs.
- A6 and B4 given their own section together. Those two are the reason this audit was commissioned before
  the results existed, and they are what we most need.
- Anything that struck you and fits no charge.
- Your environment: python, numpy, BLAS, and whether you used float32 or float64 where it mattered.

If everything sustains, say so plainly. A clean report from a real attempt to break two new unregistered
analyses is the result we are hoping for, and it is worth more than a padded one.
