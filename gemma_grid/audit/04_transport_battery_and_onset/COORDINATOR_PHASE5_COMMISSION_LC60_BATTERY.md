# PHASE-5 COMMISSION — LC-60 Commission 2: fresh-context audit of the P-map battery + B1 + B2 (100T)

Date: 2026-09-03T18:44:51Z
From: GGE1 100T Data Coordinator  (authorizing: MEMO LC-60, Commission 2)
To: a FRESH Claude Code chat, acting as the Phase-5 auditor (no prior context from producer or Coordinator)
Gate: this audit is the LAST step before any ledger entry or paper contact. Integration follows a
      SUSTAINED verdict; anything NOT-SUSTAINED or any newly-surfaced defect returns to Adam first.

## Your role (read first)
You are a fresh-context adversarial auditor. Re-derive from PRIMARY SOURCES (the frozen specs, the
artifacts, the manifests, the runner scripts). Audit the producer's claims; do not accept them on
assertion. A Coordinator verification of this batch EXISTS SEPARATELY and is deliberately NOT given to
you as an input -- form your own verdict from the artifacts first, so the two independent checks can be
compared afterward. Two firewalls stand: SR.7 (do NOT re-grade any already-registered verdict; the
GGE1 evidence base is CLOSED at 9/9 SUSTAINED and is out of scope) and SR.13 (no PA/HA basis
comparison). AX firewall holds (loving is one of the 171, no special handling). No Qwen artifact is
mounted, read, or computed on.

## Scope (exact objects + verified sha; full index in COORDINATOR_ARTIFACT_INDEX_LC60_BATTERY_100T.md 43bc6d40fa74)
Deliverables (gemma_grid/geometry_out_100T/, each SR.5-verified):
- pmap_battery_100T.json            2757ed848959...  (Items 1-8, retrieval-only; 30 ordered pairs incl 14 reverse)
- item5_svd_provenance_100T.json    64981539c944...  (gesdd->gesvd fallback provenance, 2/108)
- b1_onsetcurve_100T.json           f7b4f593bd89...  and b1_onsetcurve_knee_100T.json 713c494057...
- b1_window_sensitivity_100T.json   c264c3fe1b97...  (LC-60 C1, verified NO-FLIP)
- b2_covariate_transfer_100T.json   ae09c101590a...
Frozen/read-only (must be byte-stable; do NOT modify): pmap_transport_100T.py ee736f5d36c9,
analysisB_seam_onset_bootstrap_100T.json ca3f961459fa, geometry_out_HA/stageB_displacement__*.csv.
Specs of record: PMAP_STRESSTEST_BATTERY_PRESTATEMENT_100T.md 089fec1e4222,
B1_B2_EXPLORATORY_PRESTATEMENT_100T.md ae05e8bdcd56, P_MAP_TRANSPORT_SPEC_FROZEN_100T.md,
RUNNER_DISPATCH_..._plus_B1B2_100T.md e75f6b193455.
Runner scripts (realization-of-intent; READ, do not execute): pmap_battery_serial_100T.py,
b1_onsetcurve_100T.py, b2_covariate_transfer_100T.py, battery_handback.py, lc60_c1_b1_window_sensitivity.py.
Governance to audit against: COORDINATOR_BATTERY_REALIZATION_LOG_100T.md (R-1..R-4),
COORDINATOR_MEMO_TO_LEDGER_ITEM5_SVD_REMEDY_100T.md, and the LC-60 amendment record.

## Audit charge
1. Anchor reproduction: confirm pmap_battery Item 1 pinned analysis L17->L31 reproduces the GRADED
   primary pair (gemma-2-2b-base->gemma-2-27b-base) median_cosine 0.9716237565124259 to full precision
   against geometry_out_100T/pmap_transport_100T.json. This is the load-bearing "no adapted constant" proof.
2. Binding terms, re-audited from primary sources: NO permutation floor recomputed (every item
   retrieval-only: median_rank/top1/top5/median_cosine; estimator's perm path uncalled); frozen
   estimator byte-stable and reused import-only; the stored-canonical cleaned-vector adapter equals
   the reference clean_layer (R-2); frozen B and stageB byte-stable; no Qwen.
3. Item-5 shim + provenance: read the driver-only shim (try gesdd; only on np.linalg.LinAlgError fall
   back to scipy gesvd; frozen file untouched). Audit the provenance claim total_fallbacks=2 of 108,
   both 27b<->27b k=40 one fold, C=[4608,4608], and all_metrics_bit_identical_to_shipped=true (the
   shipped pmap_battery_100T.json is byte-stable, so the count is authoritative). Assess that the
   null-space non-uniqueness caveat attaches to EXACTLY those 2 decompositions and that the reported
   Item-5 retrieval metrics do not move. Confirm this matches the LC-60(A) freeze-amendment.
4. Each item vs its frozen/committed intent: Items 1-8 including Item 7a primary-pair Procrustes/ridge
   parity (R-1), Item 7b's 14 reverse runs and the 15 matched directional gaps (symmetry), Item 6
   centering (leakage), Item 2 raw-vs-cleaned, Item 3 split-half ceilings, Item 4 VA-plane ablation,
   Item 5 PC-truncation curve, Item 8 leave-one-family-out (exploratory/ungraded, anchor k=10 partition).
5. B1: onset curves + knee (27b-base 25.67 DEEPER, 27b-it 25.47 COINCIDENT, seam L23 CI[23,24]) + the
   window-sensitivity artifact (NO-FLIP under W4/W5; note 27b-base's ~0.05-layer margin above the
   nearest-layer boundary under W5). Confirm the committed three-way branch language.
6. B2: covariate model (arousal + L0-readability, fit on 2B) -> 27B Spearman 0.292 vs raw
   displacement-ranking baseline 0.790 -> no-reconciliation (committed); T.3's .23 context only;
   n=137, 34 drops enumerated. Confirm the committed branch.
7. Realization-of-intent notes (R-1..R-4 + the two LC-60-ratified realizations, B1 clamp and Item 8
   anchor partition): confirm each is faithful to the frozen texts and disclosed.

## Committed-branch discipline
The batch's findings enter the ledger ONLY after SUSTAINED, written EXACTLY as the committed branch
language requires -- including Item 1's in-view weaker (stimulus-driven-structure) branch and B2's
no-reconciliation. Do not rewrite committed language; confirm the results match it.

## Output + handback
Per-component verdict SUSTAINED / NOT-SUSTAINED with reasons, plus any newly-surfaced defect. Write it
as a fresh-context report artifact (SR.5'd if it lands in the repo). Route back to the Coordinator:
SUSTAINED -> Ledger grading-free integration in the committed language; NOT-SUSTAINED or any defect ->
Adam before any integration. Do NOT contact the ledger or paper directly.
