# PHASE-5 FRESH-CONTEXT AUDIT — LC-60 Commission 2 (P-map battery + B1 + B2, 100T)

Date: 2026-09-03
Auditor: fresh-context Claude Code chat (no producer/Coordinator context; Coordinator verification
`COORDINATOR_VERIFICATION_PMAP_BATTERY_100T.md` deliberately NOT read, so the two checks stay independent).
Firewalls held: SR.7 (GGE1 base CLOSED 9/9, not re-graded), SR.13 (no PA/HA basis comparison),
AX (`loving` treated as one of the 171, no special handling), no Qwen mounted/read/computed.

## VERDICT: SUSTAINED (all 7 charges). 1 open disclosure item + 1 fragility note to Adam; the B2-source item independently re-surfaced here was already closed by Adam.

The load-bearing "no adapted constant" proof is bit-perfect. Nothing overturns a committed finding or the
graded evidence base. Of the three items this audit surfaced, one (B2 stageB source) was already ruled on by
Adam hours before this audit ran — a clean independent convergence, now closed; the remaining two
(R-2 Guard-1 disclosure, B1 W5 knee margin) go to Adam before ledger/paper integration. None is a
correctness defect in a shipped metric.

---

## Per-charge verdicts (re-derived from primary sources)

**1. Anchor reproduction — SUSTAINED.** Item 1 pinned analysis row (2b-base L17 → 27b-base L31)
`median_cosine = 0.9716237565124259` is bit-identical (16 digits) to the GRADED object
`pmap_transport_100T.json /pairs/gemma-2-2b-base->gemma-2-27b-base/median_cosine`. All pinned-row metrics
(median_rank 1.0, top1 1.0, top5 1.0) match the graded pair. The frac rule maps L17→L30 (cos 0.9721); the
L17↔L31 pin is carried as the separate explicit extra row, per spec.

**2. Binding terms — SUSTAINED.**
- *No permutation floor recomputed:* `transport()` calls only `P.eval_transport(...,"procrustes")`; the
  estimator's `with_perm` path and `N_PERM` are never invoked; `floor_recomputed=false`. Graded object itself
  carries `perm_status=DEFERRED_PROHIBITIVE_RUNTIME` — consistent.
- *Frozen estimator byte-stable + import-only:* `pmap_transport_100T.py` sha `ee736f5d36c9a04c` (= index).
  Runner imports it and reuses `center/eval_transport/fit_procrustes/cosine_rows` unmodified.
- *gesdd→gesvd shim is driver-only:* module-level monkeypatch of `np.linalg.svd` that returns the original
  result on convergence (bit-identical) and falls back to scipy `gesvd` only on `LinAlgError`. Frozen file
  unedited (sha unchanged).
- *Fold assignment:* `RandomState(0).permutation(arange(171)%5)` in the runner is array-identical to the
  estimator's; fold sizes [35,34,34,34,34].
- *Stored-canonical adapter == graded input (R-2):* proven — FOUR independent code paths (item1-pin,
  item4-baseline, item6-global, item7a-procrustes) all reproduce the graded anchor `0.9716237565124259`
  bit-for-bit.
- *Frozen B (`ca3f961459fa46fc`) + stageB byte-stable; no writes.* No Qwen (hard sweep of all 6 scripts +
  5 deliverables: zero hits).

**3. Item-5 shim + provenance — SUSTAINED.** Counting-shim re-run: `total_svd_calls_item5=108`
(= 3 source-PCA preps + 3 pairs × 7 k × 5 folds = 108), `total_fallbacks=2`, both at k=40 for the two
27b↔27b directions, one fold each, shape [4608,4608]; run log confirms `fb_this_task=1` at exactly those two
tasks and 0 elsewhere; `all_metrics_bit_identical_to_shipped=true`. Null-space non-uniqueness caveat attaches
to exactly those 2 decompositions; shipped retrieval metrics do not move. Matches R-4 / LC-60(A).

**4. Each item vs frozen intent — SUSTAINED.**
- Item 1: `median_rank=1.0` at EVERY layer incl. L0 (retrieval saturated; cos 0.94→0.97) → L0 comparable →
  the **weaker** stimulus-driven-structure branch, i.e. the conservative committed in-view read. Honest.
- Item 2: exactly the 16 artifact pairs on raw vectors; primary raw 0.9702 ≈ cleaned 0.9716 → confound closes.
- Item 3: within-cell split-half ceilings 0.944–0.958 (near the .97 cross-model); required minimum
  (2b-base, 27b-base) coverage ok.
- Item 4: own top-2 PCs ablated; retrieval survives (rank 1.0, top1 0.965–0.988) → structure beyond the VA
  plane; baseline == anchor.
- Item 5: k-grid {2,5,10,20,40,80,136}; near-ceiling retrieval by k≈20–40.
- Item 6: train-only per-fold centering; Δmedian_cosine = −0.00048, Δrank = 0 → negligible leakage.
  (Centering is linear, so the runner's single train-only centering and the module's global+train-only path
  are mathematically identical; the result is robust to that difference.)
- Item 7: (a) Procrustes/ridge parity primary-pair-only, faithful to R-1 (only the primary pair carries a
  ridge fit in the Branch-B graded artifact — no new compute); (b) exactly the 14 `[NEW]` reverse keys
  (verified against the spec set) and 15 matched directional gaps, all ≈0 → symmetric transport;
  (c) non-direction-balanced disclosure present.
- Item 8: leave-one-family-out on the anchor k=10 partition; 10 families summing to 171; pooled cos 0.958;
  labeled exploratory/ungraded under the spec's own registration.

**5. B1 — SUSTAINED.** Re-derived from the stored `y(l)` independently (no estimator call):
27b-base knee 25.669 → DEEPER; 27b-it knee 25.467 → COINCIDENT; seam L23 CI[23,24] — all matching to 1e-9.
Window-sensitivity NO-FLIP reproduced exactly under W4=[42,45] and W5=[41,45]; both cells keep their branch.
Onset-curve → knee → window chain internally exact (max |log10(onset_rank) − knee_y| = 0.0).
*Fragility (commission-flagged, confirmed):* 27b-base under W5 has knee_frac 25.555, only **+0.055** above the
25.5 nearest-sampled rounding boundary; a hair lower rounds 26→25 and flips DEEPER→COINCIDENT. The branch
holds, but the window-sensitivity artifact reports "NO FLIP" without surfacing this thin margin. Carry it.

**6. B2 — SUSTAINED.** Covariate-model Spearman 0.2922 vs raw cross-scale baseline 0.7901 → covariate does
NOT beat raw → **no-reconciliation** (committed branch). T.3's 0.23 is context-only, not the baseline.
n=137; 34 drops enumerated (10 no-NRC + 24 no-L0). The raw baseline reproduces EXACTLY (0.7900824903) from
the CSVs B2 read, over the 137 covered emotions. The stageB source-of-record question this audit raised (see
Finding 1, resolved) does not affect the arithmetic.

**7. Realization-of-intent — SUSTAINED, one Guard unmet (Finding 2).** R-1..R-4 are faithful to the code and
disclosed; the two LC-60-ratified realizations (B1 floor-window clamp [42,46]→[42,45]; Item 8 anchor k=10
partition) are disclosed in the artifacts.

---

## Newly-surfaced items

**Finding 1 (B2 stageB source) — ALREADY RESOLVED (independent convergence; no action).** This audit,
working only from the frozen commission scope, flagged that B2 read stageB CSVs from
`gemma_grid/geometry_out_100T/` (2b `624a5c010015…`, 27b `2d2db2d16744…`, mtime Aug 19) while the commission
scope and index register the frozen stageB objects under `gemma_grid/geometry_out_HA/…` (`880d45b1ff11…`,
`a4564afd2747…`, mtime Aug 1) — different files, no directory disambiguated in the B2 spec. On checking the
live manifest I found this was already closed BEFORE the audit ran: `b2_source_provenance_100T.json`
(sha `23d8f438…`, SR.5-registered 2026-09-03T19:50Z; sidecar "corrects the HA-tree provenance slip") records
**Adam's 2026-09-03 confirmation** that `geometry_out_100T` stageB is the source of record per **SR.2**
(displacement is a basis-dependent cleaned-vector quantity; only the 100T canonical stageB CSVs may be read
by canonical analysis; PA/HA trees are provenance-only), pins the exact two 100T input hashes, and states the
HA files differ and were not read (SR.13 intact). The independent fresh-context audit converged on the same
provenance question Adam had already ruled on — validating both checks. No further action; the commission's
geometry_out_HA line is the slip, now corrected of record. (This audit performed no PA/HA comparison.)

**Finding 2 (disclosure completeness — R-2 Guard 1) — OPEN.** R-2 Guard 1 required the handback to report the
ACTUAL max absolute deviation between stored and recomputed cleaned vectors ("a number, not '~1e-6' on
assertion"). `RETURN_PMAP_BATTERY_plus_B1B2_100T.md` still asserts "~1e-6." The load-bearing fidelity is
independently proven here by the bit-identical anchor (deviation effectively zero for every graded-object
item), so this is a documentation-completeness gap, not a correctness defect. Requested: fill in the measured
figure, or cite the bit-identical anchor as the stronger evidence.

**Observation (fragility, already commission-flagged).** B1 27b-base W5 knee margin +0.055 to the
nearest-sampled boundary (see charge 5). Worth one explicit sentence in the B1 write-up.

---

## Routing
SUSTAINED → the committed-branch language may enter the ledger grading-free. Finding 1 is already closed by
Adam's SR.2 ruling. Finding 2 (Guard-1 disclosure) and the B1 W5 fragility note are the only open items and,
per the commission ("anything NOT-SUSTAINED or any newly-surfaced defect returns to Adam first"), go to Adam
for a one-line ruling before integration — neither blocks the committed findings. No ledger or paper contact
made by this audit.
