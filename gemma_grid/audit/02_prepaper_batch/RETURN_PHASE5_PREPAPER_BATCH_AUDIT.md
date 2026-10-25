# Return: Phase-5 pre-paper audit batch — to the GGE1 100T Data Coordinator

**From:** Claude Code (fresh audit dispatch, per `PHASE5_PREPAPER_BATCH_AUDIT_PROMPT_100T.md`,
sha `453a565f…`). Five owed adversarial audits, run in one pass. Each task was recomputed by an
**independent fresh-context auditor** with its own hand-written code (own loaders, own
PCA/SVD/correlation/RSA/norm-lens — no producer helper trusted for the numbers). **Nothing in
the repo, ledger, or any artifact was modified** — read-only throughout; all scratch code in the
session scratchpad. Return path: **Coordinator** (not Ledger/Grid directly).

## Bottom line — all five SUSTAINED

| task | component | regime | verdict | max Δ (recompute vs artifact) |
|---|---|---|---|---|
| **WS** | welfare-paper base↔IT scan, Col-1 (A/B/B-ii/C) | CLEANED 100T | **SUSTAINED** | ~2e-6 (f32 SVD last-digit) |
| **OP3** | band-sweep L42–L45 base-27B | RAW / basis-immune | **SUSTAINED** | <1e-9 |
| **PC** | P-colon | CLEANED 100T | **SUSTAINED** | ~9e-5 (f32 einsum) |
| **FI** | Tier-1 fold-ins P3.2 / P3.4b / P3.7 | mixed | **SUSTAINED** (all 3) | 0 |
| **G66** | valence-asymmetry onset | CLEANED 100T | **SUSTAINED** | 0 (all 36 numbers) |

No refutation on any task. No number differed beyond the float precision of the producer's own
dtype. **Anchor `emotion_vectors_171_100T.pt` sha `6de2e293…971b` re-hashed intact at start and
end by every auditor and by the coordinator sweep; all ten audited artifacts unchanged.**

---

## Per-task detail

### WS — SUSTAINED (A/B/B-ii/C)
Independent fresh context (no access to the B-ii producing session's reasoning).
- **(A)** antiparallelism gaps IT−base reproduce: 2b +0.016461, 9b +0.014472, 27b +0.031262;
  opposite-cosine and PC1-separation IT>base at all three; q=11, n_rm=45.
- **(B)** own-PC1 valence_r base→IT 2b .711→.764 / 9b .715→.777 / 27b .720→.772; arousal_r(PC2)
  falls (2b .499→.363). **SR.7 respected** — these are the registered G2.1 quantities; confirmed
  as numbers only, the pooled NULL_subfloor/NULL verdict was **not** re-graded.
- **(B-ii)** pooled-mean axis structurally **degenerate** (half-split cosine −1.000000,
  ‖s‖/median-Δ ~2–5e-6, do-not-interpret) confirmed; **both** projected objects valence-ordered
  on Δ-PC1 — v(e) per-arm |r|≈0.80 and Δ_e shift-loading naive **+0.763369 / +0.735461 /
  +0.810334**, LOO moving ≤6e-4 (≤0.001 as claimed); arousal null (~0 to −0.13); var-frac
  0.353/0.359/0.293. Disclosures + `commissioned: LC-40` + `projected_object: "both"` +
  per-scale `delta_e_projection_valence_ordering` all present.
- **(C)** full 6-cell lens coverage; independently rebuilt signed valence-PC1 and recomputed
  raw logits over the 256000 vocab — top-15 promoted / bottom suppressed IDs reproduce
  byte-for-byte on both W_U paths (zipfile anchor + torch non-anchor).
- Basis: all six cleaned paths contain "100T", all six file shas match INPUT_MANIFEST, no PA/HA
  reachable. AX firewall clean (`loving` in per-emotion rows only).
- **Corroboration:** the coordinator also independently reproduced the B-ii naive Δ_e
  correlations from the stored loadings + RM CSV (no vector reload, no producer code) to 2e-16 —
  concurring with the fresh auditor.

### OP3 — SUSTAINED (basis-immunity verified TRUE)
- Parts 1–4 reproduce to <1e-9: R_ff / P(ℓ) ceilings (P(45)=0.9462, P(31)=0.9854; L31 selfcheck
  absdev 0.000479); band-shape grades all four layers **GENUINE**; L44→L45 mechanism (top1_dim
  4140, change-share 13.8%, negative recovery_fraction — the massive-activation dim *stabilizes*
  rather than drives the drop); IT comparison.
- **Basis-immunity claim verified TRUE:** the producer reads the *plain*-filename structures
  (`vectors_171.pt`, `…27b-it/…/canonical.pt`), but their `vectors_raw` tensors are
  **byte-identical** to the `_100T`-labelled variants (torch.equal True, max-abs-diff 0.0) — RAW
  diff-of-means, nothing rides on a cleaned/100T-labelled quantity. Correctly exempt from the
  100T check. Input shas (seed0/seed1/locked/it), fingerprint gate rev `938270f5`, ceiling gate
  all pass. Unregistered characterization — no registered row touched.
- **Descriptive flag (not a numeric defect):** the L44–L45 "divergence" reads more accurately as
  a **graded slope with its steepest single step at L43→L44** (seed-avg consecutive 0.909),
  continuing through L45, rather than an isolated L44→L45 cliff. The artifact's own
  `consecutive_corr_seedavg` / `band_unlikeness_vs_lockedL42` already carry the data for that
  phrasing. Offered for author wording; changes no number.

### PC — SUSTAINED (NULL supported; flag only)
- Triad (regretful/remorseful/guilty) IT−base diff/SEM/z at L31 all **IT_lower**; aggregate
  z=**−1.589**, one-sided p=**0.056**; band means early −62.4 / mid **+182.6 (IT_higher)** /
  late **−881.3 (strongly IT_lower)**; rank-overlap vs Appendix T.1 = **2/10**, moral-triad
  suppression ranks **34/55/57** — all reproduce to ≤9e-5 (f32). Input shas
  (`94244b23`/`769a917f`/`6de2e293`) verified. NULL determination supported (uniform IT_lower
  sign → not falsified; effect within across-scenario noise at the 0.05 bound → not confirmed);
  flagged, not re-adjudicated. AX firewall clean. §4.6 raw-colon arm correctly excluded.
- **On the auditor's one caveat:** it flagged the anchor's internal `variant:"neutral_HA"` field
  and suggested re-stamping it. **This is the pre-declared item-10 stale label** — the audit
  prompt itself states it "must stay untouched (item-10 stale label; 32 producers gate on this
  hash)." Correctness is protected by the sha gate. **Recommendation: no action** (do NOT
  re-stamp; the auditor's suggestion is overridden by the item-10 directive). Known trap, not a
  new finding.

### FI — SUSTAINED (P3.2, P3.4b, P3.7)
- **P3.2** (RAW/exempt): L0/Lfinal/growth_ratio/largest_drop + full per-layer arrays reproduce
  with **max |Δ| = 0** across all 6 cells (e.g. 27b-it ratio 131.24). Loads **only** neutral_100T
  activations (keys `acts/src/src_sha256`, no `emotions`/`vectors`) — no cleaned vector touched;
  27b-base correctly sources repo-root `neutral_100T.pt`. **Per-dialogue-mean PROXY disclosure
  present** in the artifact (E1 per-token files pod-lost → growth curve + final-layer transition
  only). (Note: `largest_drop_transition` uses the 0-based from-layer convention — confirmed
  correct, independently verified.)
- **P3.4b** (CLEANED 100T): 228 per-layer rows reproduce, max |Δ|=0. **Cross-consistency
  `arousal_pc2_at_L` == Stage-A `arousal_r` holds for all 6 cells** (fixed global PC2 sign makes
  the analysis-layer value = |corr(PC2, RM arousal)| by construction, and it does). All paths
  100T, n_rm=45.
- **P3.7** (CLEANED 100T): median maxrot/moststable transitions, moststable cos, and the
  terminal/last3/seam counts reproduce, max |Δ|=0. **E2 firewall respected** — loving's velocity
  minimum reported as a bare number, uninterpreted. All paths 100T.

### G66 — SUSTAINED
- All **36** numbers (6 cells × {pos,neg} × {onset, L0, final}) reproduce with **zero delta**
  under an independent norm-lens `((v/rms(v))·(1+ln_w))@W_U`, eps 1e-6, own-rank streamed over
  256000 tokens, all layers. Combined onsets match `lens_summary` (2b/9b→0, 27b→23); fair count
  140 per cell (pos_50/neg_87 → 137 with NRC labels). Surprise-gate shape confirmed (pos more
  readable at L0 for 2B/9B; 27B pos onset 1–2 layers before neg; neg reaches lower final ranks at
  all six cells) — flagged, not re-graded. All cleaned paths 100T; anchor untouched.
- Minor: the audit prompt's *stated* anchor path had a one-directory slip
  (`gemma_grid/data/vectors/…` vs the real repo-root `data/vectors/…`); the producer's
  `common.py::CELLS` uses the correct paths — spec-wording only, no artifact effect.

---

## Firewalls & discipline — all honored
- **SR.7** (WS-B never re-grades registered G2.1): honored — confirmed numbers, flagged only.
- **E2** (P3.7 loving velocity-min uninterpreted): honored.
- **AX** (`loving` in JSON rows only, out of prose): honored on WS, PC, FI.
- **P3.2 per-dialogue-mean proxy**: disclosed in-artifact, verified.
- **Basis regimes**: CLEANED tasks (WS, PC, P3.4b, P3.7, G66) verified 100T all the way down, no
  PA/HA leak; RAW tasks (OP3, P3.2) verified genuinely basis-immune (OP3 by byte-identity of
  `vectors_raw` across variants), gates + arithmetic audited.

## Items for the Coordinator to route / rule
1. **No corrections required** to any artifact — all five sustained, numbers exact to producer
   dtype precision.
2. **PC variant-label caveat**: recommend **no action** (item-10; leave `variant:"neutral_HA"`
   untouched per the prompt; sha-gate protects correctness). Flagging so the Coordinator can
   confirm the override of the auditor's re-stamp suggestion.
3. **OP3 phrasing note** (optional, for authors): "graded slope, steepest at L43→L44" is the more
   precise description than "L44–45 cliff." No number changes.
4. **G66 spec path-wording slip** (prompt only, not the artifact): note for whoever maintains the
   audit-prompt template.

Anchor `6de2e293…971b` intact start-to-end. Producing/auditing sessions do not grade this report;
it returns to the Coordinator for the verification step and onward routing to Grid/ledger.

---

## Addendum — WS blind re-audit (independence hardening)

Because this session produced the WS B(ii) addendum, WS was re-audited a **third** time by a
**blind adversarial** auditor given **no expected values**: it reconstructed the recipe from
committed source only (`common.py`, `noise_floor.py`, the four producers), recomputed every WS
quantity from the raw cleaned vectors, **saved its results before opening any artifact**, and was
tasked to actively falsify the B-ii valence-alignment claim.

**Result: SUSTAINED — abs delta 0.0 (bitwise) on A / B / B-ii / C.** Same BLAS as the producers,
so even SVD/matmul ordering reproduced. Its adversarial B-ii probes each confirmed the finding:
- pooled-mean axis degeneracy is **real and structural** (‖mean Δ‖/median‖Δ‖ ~2–5e-6, half-split
  cosine −1.0000) — not hidden signal;
- the Δ_e→valence alignment **survives leave-one-out** (naive→LOO change ≤0.003, no collapse) —
  not a self-term artifact;
- arousal is a **genuine null column** (~−0.05 to −0.11) — valence is not proxying arousal;
- Δ-PC1 variance fraction 0.293–0.359 makes "top PC" **well-defined** (not near-isotropic);
- no sign/normalization inflation; basis + AX firewall pass; anchor sha unchanged (start = end).

**Independence status: resolved.** WS now has three concurring checks — the original briefed
auditor, a from-scratch coordinator cross-check of the naive Δ_e correlations (2e-16), and this
blind no-answer-key adversarial re-audit — none of which found any discrepancy. No correction.
