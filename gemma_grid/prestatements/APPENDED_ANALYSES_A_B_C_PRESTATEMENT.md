# Pod job — P-map pod, appended GGE1 analyses: pre-statement (CORRECTED — FROZEN 2026-08-27 on Adam's go)

*Corrected 2026-08-27 by the GGE1 100T Data Coordinator against LC-51 (Ledger ratifications) and the
four resolutions in `RESOLUTION_PMAP_APPENDED_ANALYSES_100T.md`. Status: FROZEN 2026-08-27 on Adam's one-word go; this file's hash is recorded in the
SR.5 manifest before pod start. The P-map floor job is governed by its own frozen spec
and is untouched by this document.*

*Scope rule for the pod: the P-map floor plus the three analyses below, nothing else. Each analysis
is pre-stated with its resampling unit, frozen draw count, exact reported quantities, and the label
its results carry. All three are robustness/calibration analyses attached to existing ledger entries.
No registered verdict changes. Executed by the 100T Data Coordinator. Outputs land in
`gemma_grid/geometry_out_100T/` with manifests and sha256, Phase-5 audited before any ledger or paper
contact.*

## Standing prohibitions on this pod

1. **No Qwen artifacts are mounted, read, or computed on.** Not "computed but unread." The OP.5 blind
   frontier does not distinguish computed-and-unread from read once results land in a file. Any Qwen
   touch on this pod is a frontier breach.
2. **No estimator changes.** Every analysis runs the untouched reference estimator from the existing
   100T pipeline. The only new code permitted is the parallelization wrapper (per-draw dispatch across
   cores) and the result collector. Wrapper code is captured at teardown with the env freeze.
3. **Mac-track items stay off the pod.** The displacement regression, the G6.6 frequency control, the
   op1 spectrum promotion, and the bipolarity cosines go through their normal local process. Named
   here so their absence from the pod manifest is a recorded decision, not an omission.
4. **Core scheduling (LC-51).** The P-map floor's ~3.5 h determinism run is the priority job. The
   three ride-alongs schedule *around* the floor, not against its cores — riders yield to the floor.

---

## Analysis A — Displacement rank-stability bootstrap (attaches to G5.4a / G5.4b)

**Why.** The ledger carries a standing caveat that G5.4b's threshold-adjacent grades "have been
unstable across the study's perturbation history," and G5.4a (vindictive trio, guilt cluster) is a
quotable exploratory ranking with no uncertainty attached. This analysis quantifies both.

**Perturbation channel (primary).** Cleanup-basis resampling, mirroring the actual historical source
of instability (basis changes across PA/HA/100T). Per draw, per cell: resample the 500 neutral
dialogues with replacement, re-derive the 50%-variance cleanup basis at the analysis layer with the
reference procedure, re-clean the saved raw 171 vectors, recompute the per-emotion displacement for
the three tuning pairs with the reference displacement estimator, record all 171 ranks per pair.

**Frozen parameters.** B = 1,000 draws. Seed = 20260827. Cells: all six. **Pair cells SHARE draw
indices (RATIFIED, LC-51):** the pipeline's data-generating process uses ONE shared neutral corpus
for every cell, so a resample draw is applied identically to both cells of a pair — mirroring the
real process and preserving the paired structure that displacement (a within-pair quantity) depends
on. Matches the house convention (registered G3.4 CI resamples paired across cells). Anchor cell
27b-base reads repo-root paths (`data/vectors/emotion_vectors_171_100T.pt`, `neutral_100T.pt`); the
five non-anchor cells read their per-cell 100T neutral/vector artifacts.

**SR.7 (registered-quantity firewall, LC-51).** G5.4b's three registered sub-verdicts (p = .012 /
.742 / .216, and the scale-pair CONFIRMED) are quantified for stability by this bootstrap and are
NEVER re-graded by it. Whatever fraction of draws crosses a threshold, the verdicts of record stand
and the crossing rate lands in the knife-edge caveat — which is exactly the caveat this analysis
exists to put a number on.

**Reported quantities, exactly these.**
- Per tuning pair, per emotion: bootstrap 95% percentile CI of displacement rank.
- Per tuning pair: fraction of draws in which each vindictive-trio member sits in the top 10, and in
  which the trio's mean rank sits in the top decile.
- Per tuning pair: loving's rank CI, and the fraction of draws in which each G5.4b converse
  sub-verdict would grade DISCONFIRMED / NULL under the registered rule (reported as a stability
  rate under SR.7, not a re-grade).
- One summary table per pair; full draw-level ranks saved as CSV for audit.

**Label.** Disclosed robustness analysis attached to the G5.4a/G5.4b caveat. Registered and
exploratory verdicts of record unchanged. If trio-top-decile stability is low, the paper's G5.4a
sentence weakens accordingly — that consequence is accepted in advance.

---

## Analysis B — Seam / onset measurement resolution (attaches to G6.2 / G6.4)

**Why.** G6.4 claims a gap of exactly 0.000 layers between the geometry consolidation seam and the
vocabulary readability onset in both 27B cells. An exact coincidence claim requires a stated
measurement resolution. This analysis supplies the CI a referee will ask for.

**Design.** Per draw, per cell: resample the 500 neutral dialogues with replacement, re-derive the
per-layer cleanup bases from the resampled neutral activations, re-clean the saved raw per-layer 171
vectors, then recompute (i) the consolidation **seam** layer under the frozen seam definition —
`phase4_extracts_100T.py` (first layer whose RSA to the analysis layer is ≥ 0.97 sustained over 3
layers), output `extracts_summary.json : seam_per_cell` — and (ii) the readability **onset** layer
under the frozen G6.2 definition, **median own-rank < 10,000** (`onset_layer_median_lt_10k`), using
the saved lens ingredients. Gap = onset − seam per the frozen `seam_gap_metric_100T.py`. Record seam,
onset, and gap per draw. Anchor 27b-base reads repo-root `lens_ingredients.pt` and repo-root vectors;
non-anchor cells read their per-cell artifacts.

**Onset threshold (RATIFIED, LC-51).** The headline gap CI computes at the frozen **< 10,000** onset
and nowhere else — that is the registered G6.2 / P4.5 onset event the seam_gap = 0.000 claim rests
on. The `lt100` / `lt1000` variants may ride as **extra disclosed sensitivity resolutions** (P4.5
precedent), labeled sensitivity, never the headline.

**Frozen parameters.** B = 500 draws, with a **B = 250 fallback** if the pre-run 5-draw timing probe
on the live pod shows B = 500 will not fit the pod budget alongside the floor; the probe result and
the chosen B are recorded here before pod start (this is a cost rule fixed before any results are
seen, not a results-dependent choice). Seed = 20260828. **Onset recompute restricted to the layer
window L18–L28** (the onset norm-lens per-draw per-layer PCA is the pod cost driver; the frozen onset
lies inside this window in both 27B cells and the window is fixed before the run). Seam computed on
all six cells; onset and gap reported for the two 27B cells (onset is undefined at 2B/9B, which is
G6.2's content). The 2B/9B L0-readability status per draw is recorded as a stability check that the
gate's absence holds under resampling.

**Reported quantities, exactly these.**
- Per 27B cell: bootstrap distribution of seam layer, onset layer (at frozen <10k), and gap; 95%
  percentile CI of the gap; fraction of draws with gap = 0.
- Per 27B cell, sensitivity riders: the same gap distribution at lt100 and lt1000, labeled sensitivity.
- Per 2B/9B cell: fraction of draws in which the cell remains readable from L0.
- Per cell: 95% CI of the seam layer (feeds the G6.4 seam-migration sentence).

**Label.** Disclosed measurement-resolution analysis for G6.2/G6.4. Verdicts unchanged. The paper
reports the frozen-<10k gap CI wherever the 0.000 coincidence is claimed.

---

## Analysis C — Dependence-aware bootstrap for G3.2 (template: the G3.3 precedent)

**Why.** G3.2's arousal-scale lead is monotone and above floor in both chains but unresolved at the
registered bar (p = .237 / .200 after Holm) under a test whose independence assumption G3.3's
disclosed analysis showed overstates the SE by ~4×. The house precedent is set by G3.3: the
registered verdict stands, and a dependence-aware bootstrap is reported alongside as a disclosed
calibration.

**Design.** Apply the G3.3 dependence-aware bootstrap machinery — `phase4_c3_grade.py :: gap_boot`,
unmodified — to the registered G3.2 contrast (2b→27b arousal-alignment gain, base and IT chains).
Inherited parameters (RATIFIED, LC-51; cited, not chosen): resample the **45 RM emotions jointly**,
PCA hoisted and fixed, seed 0, **operative B = 2000** — the operative call value, which matches
G3.3's actual run; the module docstring's "5000" is stale and is not used.

**Reported quantities, exactly these.** Corrected p per chain, the bootstrap SE against the
registered test's SE, and the same summary format the G3.3 disclosure used.

**Label.** Disclosed better-calibrated analysis, mirroring the G3.3 precedent verbatim. The
registered G3.2 verdict (lead, not established) stands regardless of outcome, and the paper reports
both numbers the way it already does for G3.3.

---

## Teardown capture

Wrapper code, env freeze, this file's frozen hash, per-analysis result files with manifests and
sha256, and the draw-level CSVs for A and B. The chosen B for Analysis B (500 or 250) and the timing
probe result are recorded above before start. Phase-5 audit covers all three before ledger contact.
Handbacks return to the Coordinator for verification, then route onward — never straight to the
ledger/grading pass. Any analysis idea arising while the pod is open is written down for the Mac
track or a future pre-statement, and does not run on this pod.
