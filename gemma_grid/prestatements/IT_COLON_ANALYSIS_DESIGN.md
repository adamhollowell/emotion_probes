# IT Colon Analysis — design (new GGE1-owned analysis, 100T session)

*(**Amendment 1, 2026-08-11, PRE-DATA** — aligned to E1's banked base-side protocol,
`coordination_100T/E1_baseside_protocol_for_GGE1.md`, which supersedes this design's
original assumptions. Deltas, disclosed in full: (1) prompt set is E1's frozen fig2
**12-scenario** set (hashes `scenario_acts.pt` = `94244b23…345e12`,
`fig2_scenario_sha256.txt`), not the assumed ~5-prompt P7.1 set; (2) the **primary
measurement point is the bare-prompt final token** — E1's fig2 gate empirically
REJECTED template/colon positions (off 1.8–3.0×) for cross-side comparability, so the
matched pairing runs bare-prompt final-token, unit-normalized dot, L31 resid_post f32,
per the frozen convention; (3) the raw Human:/Assistant: colon arm is demoted to a
labeled TERTIARY IT-side regime probe (the §4.6 datum), excluded from the prediction;
(4) the base side requires **no E1 GPU job** — persisted `scenario_acts.pt`
re-projects onto `emotion_vectors_171_100T.pt` on CPU (protocol §3); (5) **P-colon is
restated at the frozen measurement point, still pre-data** (no scenario data exists
for 27b-it under any basis): at the bare-prompt final token on the 12 scenarios,
27b-it's unit-normalized activations of the moral/self-conscious probes (regretful,
remorseful, guilty; secondarily self-critical, reflective) are LOWER than 27b-base's
at L31, graded against across-scenario spread; the name "P-colon" is retained as a
label though the measurement point is no longer a colon token. The body below is
retained as filed for the record; where it conflicts with this amendment, the
amendment governs.)*

**Commissioned:** coordination chat → Grid Chat, 2026-08-10. **Owner:** GGE1.
**Object:** the base↔IT contrast at the **Assistant-colon token** — Anthropic's exact
measurement point ("the colon token after 'Assistant,' immediately prior to its
response") — pairing 27b-it against E1's base-side `colon_analysis_100T.pt` (E1
addendum item 3: their fresh ~5-prompt forward pass at the 27b-base load). This is the
regime Appendix T could only approximate with pooled neutral dialogues: single-token,
assistant-turn, response-imminent. Squarely §4.6-adjacent — "Human:/Assistant:" is the
IT models' near-native format, and this measures exactly the base-vs-IT tag-sensitivity
contrast that rationale named.

## Registration posture (Grid Chat call, flagged for Adam's veto)

**New ledger entry, unregistered exploratory — with one genuinely pre-registered
directional mini-prediction**, filed here BEFORE any colon data exists anywhere:

> **P-colon (conf M, from Appendix T.1's suppression direction):** at the Assistant-colon
> token on the shared prompt set, 27b-it's activation of the moral/self-conscious probes
> (regretful, remorseful, guilty; secondarily self-critical, reflective) is LOWER than
> 27b-base's, at the analysis layer (L31) and in the mid-to-late band mean. Falsified by
> the opposite sign beyond the run's measured probe-activation noise; NULL if within it.

That is registerable because the data does not exist — unlike the T.1 observation that
generated it (out-of-sample discipline, same as `OP2_norm_set_PREREGISTRATION_draft.md`).
Everything else below is descriptive/exploratory and labeled so.

## Job

- **Job ID:** `GGE1-G4` (add to manifest §B; batches into shared head B's **27b-it**
  load — zero extra model loads; the base side is E1's `colon_analysis_100T.pt` job at
  shared head A and is NOT duplicated here).
- **Prompt set:** EXACTLY E1's P7.1 prompt set, verbatim (coordination must ship the
  prompt list with E1's job spec; a hash of the prompt file goes in both jobs'
  manifests so the pairing is provable). No new prompts in v1 — a matched
  charged/neutral mini-battery à la Anthropic Figs 37–39 is a v2 decision AFTER the
  five-prompt pairing lands (extending the set would fork E1's job; out of scope now).
- **Format arms (both cheap; both extracted):**
  1. **Primary — matched-raw:** the identical raw "Human:/Assistant:" text E1's base-side
     job runs, ending at the "Assistant:" colon token. Same text, both models — the only
     clean base↔IT contrast (base cannot take a chat template).
  2. **Sensitivity — IT-native:** the same prompts in Gemma's official chat template.
     NOTE the template has no "Assistant:" colon; the measurement point becomes the
     final token of the model-turn header, and this arm is a disclosed regime-shift
     comparison, never pooled with arm 1.
- **What is saved: RESIDUALS, not projections** — full resid_post at the measurement
  token, all 46 layers, f32 (~5 prompts × 2 arms × 46 × 4608 ≈ 4 MB). The E1 P7.1
  lesson (projections-only made re-basing impossible) is the design's founding rule.
  Output: `gemma_grid/colon_it/colon_analysis_it_100T.pt` (+ SR.5 manifest, fingerprint
  gate vs `aaf20e6b…`, pod sha256).
- **Est. GPU:** minutes at the existing 27b-it load.

## Pre-stated analysis (CPU, post-session)

1. **Probe convention (primary):** E1's `emotion_vectors_171_100T.pt` cleaned vectors as
   the fixed probe set applied to BOTH sides — Anthropic's one-probe-set-two-models
   convention, and the only convention that makes the base file directly comparable.
   Sensitivity: 27b-it's own `canonical_100T.pt` probes; raw-vector variant of both.
2. **Headline quantities (pre-stated):** per-emotion cos(probe, colon residual), per
   layer; the base↔IT diff profile at L31 and band means (early/mid/late thirds);
   **the P-colon test** (grade the mini-prediction, sign + noise bound from
   across-prompt spread); rank overlap of top-suppressed emotions vs Appendix T.1's
   pooled-neutral list (does the T.1 direction survive the regime upgrade?).
3. **Comparative texture (exploratory, labeled):** Claude Fig 37–39 qualitative
   comparison (their charged-scenario shifts vs our five prompts' diffs — named
   emotions only, no correlation claimed at n=5); the two format arms contrasted as a
   §4.6 datum (how much the near-native format moves IT's colon-state readings).
4. **Grading:** Grid Chat grades P-colon and writes the ledger entry; producing session
   reports numbers only. Phase-5 check owed before any paper contact (T-package
   precedent). AX firewall: the loving row stays in the JSONs, out of the prose.

## Dependencies and coordination

Blocked on: E1's `colon_analysis_100T.pt` job being scheduled (shared head A) and the
prompt list + hash being published to both jobs. Nothing else. If E1's job slips, arm
extraction can still run (prompts are fixed text) and the base pairing waits.
