# FROZEN PRE-STATEMENT — P-map robustness battery (8 items) — for Ledger hash-freeze
**Drafted by the GGE1 100T Data Coordinator per LC-57, amendments incorporated. 2026-09-02. NOTHING EXECUTES UNTIL THE LEDGER HASH-FREEZES THIS TEXT.**
Governing: frozen P-map spec `P_MAP_TRANSPORT_SPEC_FROZEN_100T.md` (`09b4dacd…`) and frozen estimator `pmap_transport_100T.py`. **The P-map grade (CONFIRMED) is untouched by every branch below.** All eight items are disclosed robustness attached to P-map; Item 8 runs under its existing spec registration, folded into the same pass. Return path: Coordinator verify → Ledger grading-free integration → Phase-5 before ledger/paper contact.

## Standing rules (binding)
One pre-statement, one freeze, one execution pass, whole battery reported — no running items singly and stopping when the picture looks good. Every item reuses the frozen estimator, fold assignment, and permutation machinery UNMODIFIED; new code is input-preparation and collection only. Both interpretation branches are stated for every graded-adjacent item, including the unflattering one. No Qwen artifact is mounted, read, or computed on. Outputs to `gemma_grid/geometry_out_100T/` with manifests and sha256. Phase-5 audit before ledger or paper contact. **No battery item recomputes the permutation floor; the graded floor exists once, at the graded configuration. All items report retrieval metrics (top-1, top-5, median rank, median held-out cosine) only.**

## Frozen reuse block (verbatim from the estimator, unmodified)
`FOLD_SEED=0`, `N_FOLDS=5`, `N_PERM=10000`, `PERM_SEED=0`, `RIDGE_GRID=[1e-3,1e-2,1e-1,1.0,10.0,100.0,1000.0]`, `ANALYSIS_LAYERS={2b:17, 9b:28, 27b:31}`. Fold assignment `RandomState(0).permutation(arange(171)%5)`. Centering `center(V)=V - V.mean(0)` applied once to the full 171 per cell (global-171, confirmed by the script author, LC-57). Map kinds: semi-orthogonal Procrustes (primary) and ridge (secondary). Floor: add-one permutation p on top-1 accuracy.

## Master ordered-pair list (amendment ii + LC-58 Item-7 amendment — ALL 30 ordered pairs, explicit)
15 unordered pairs x both directions = 30 ordered. The artifact already holds 16; the 14 runs marked [NEW] are the reverse directions, computed in this pass on the same cleaned vectors (frozen estimator/folds/seeds unmodified, input-prep only). 14 = 3 tuning + 4 (2b<->9b) + 3 (2b<->27b) + 4 (9b<->27b).
- Tuning (same size): 2b-base->2b-it | 2b-it->2b-base[NEW] | 9b-base->9b-it | 9b-it->9b-base[NEW] | 27b-base->27b-it | 27b-it->27b-base[NEW]
- 2b<->9b: 2b-base->9b-base | 9b-base->2b-base[NEW] | 2b-base->9b-it | 9b-it->2b-base[NEW] | 2b-it->9b-base | 9b-base->2b-it[NEW] | 2b-it->9b-it | 9b-it->2b-it[NEW]
- 2b<->27b: 2b-base->27b-base | 27b-base->2b-base | 2b-base->27b-it[NEW] | 27b-it->2b-base | 2b-it->27b-base[NEW] | 27b-base->2b-it | 2b-it->27b-it[NEW] | 27b-it->2b-it
- 9b<->27b: 9b-base->27b-base[NEW] | 27b-base->9b-base | 9b-base->27b-it[NEW] | 27b-it->9b-base | 9b-it->27b-base[NEW] | 27b-base->9b-it | 9b-it->27b-it[NEW] | 27b-it->9b-it
Direction convention for the asymmetry read: cross-scale UP = smaller->larger, DOWN = larger->smaller (matched within each unordered pair); tuning pairs read base->it vs it->base.

## Item 1 — layer-wise transport curve (shared-stimulus test; highest priority)
Run the frozen protocol at every sampled layer (L0 included), pairs = 2b-base→27b-base (primary) plus both 27B directional pairs (27b-base→27b-it, 27b-it→27b-base), per the master list. **Layer-pairing (amendment i + ratified rule):** frac = layer/n_layers with n_layers 2b=26, 9b=42, 27b=46; target layer = round(f·N_t). **PIN the registered analysis-layer graded configuration 2b-L17 ↔ 27b-L31 as an explicit extra row** beside the frac-mapped curve (the frac rule alone maps L17→L30). Per layer report top-1, top-5, median rank, median held-out cosine. **Branches:** deep-layer transport clearly beats L0 → claim sharpens to transport of contextually constructed geometry; L0 comparable to the analysis layer → the P-map sentence weakens to transportability of stimulus-driven structure and that weaker sentence ships. Secondary, report-only: the layer where transport consolidates, against the G6.4 seams.

## Item 2 — raw-vector rerun (cleanup confound)
Frozen protocol on the saved raw vectors at each cell's analysis layer, over **the 16 pairs the P-map artifact computed on cleaned** (the non-[NEW] entries in the master list), matched to the graded P-map object, same metrics. **Branches:** comparable raw retrieval closes the confound in one row; materially worse means cleanup is load-bearing for transport and the claim is scoped to cleaned vectors.

## Item 3 — split-half ceiling (new; original spec forbade it in-run)
**Reconciliation CLEARED:** `per_story_NOT_materialized.txt` is L45-probe-context only (run_probe_l45.py at layers {31,42,45}); it says nothing about the main `{cell}__per_story_slice.pt` files. **Basis (amendment iii):** each half's 171 vectors are cleaned with the standard 100T cleanup, matching the graded P-map object. Within a cell, split the stories into halves (rule/seed/half-assignment frozen: `RandomState(0)`, story-index parity after a fixed shuffle), derive the two half-sample 171×d matrices with the reference pipeline, run the identical transport protocol between the halves of the same cell. **Run-time coverage check (built-in):** if any cell's per-story coverage cannot support 171×d half-matrices, that cell reports coverage only; 2b-base and 27b-base are the required minimum. Report the P-map metrics. **Branches:** ceiling near the cross-model .97 → "transport across models is as good as re-measurement within one"; ceiling well above .97 → a real cross-model gap, reported as such.

## Item 4 — VA-plane ablation
**Axis (amendment iv):** each cell's OWN PC1/PC2 at its analysis layer (the G1.1 objects). Project those axes out of source and target matrices, rerun the frozen protocol at the analysis layer, pairs = 2b-base→27b-base + both 27B pairs. **Branches:** surviving retrieval → transport carries fine structure beyond the circumplex plane; collapsed retrieval → the transportable object is principally the VA plane and the claim is scoped accordingly.

## Item 5 — PC-truncation curve
Fit the map on the source's top-k PCs only, k ∈ {2,5,10,20,40,80,136} (frozen grid), rerun retrieval, pairs = 2b-base→27b-base + both 27B pairs. Report retrieval against k. Read-across, report-only: the k for near-ceiling retrieval against the G3.1 spreading result.

## Item 6 — centering audit (RESOLVED; robustness row LIVE)
Confirmed from the frozen code and by the script author (LC-57): centering is the global-171 mean at load, so the fold fit carries a real (doubly tiny — cleaned vectors are already per-cell centered) train/test leakage. Rerun the primary pair (2b-base→27b-base) with train-only (per-fold) centering as a robustness row; report the delta. Expected to change nothing; the point is retiring the leakage quibble before a referee raises it.

## Item 7 — up-vs-down asymmetry + Procrustes/ridge parity (RE-CLASSED report-only -> small-compute, LC-58)
(a) **Procrustes-versus-ridge gap per pair** from the existing artifact (parity -> the correspondence is genuinely rotation plus scale). No new compute.
(b) **Up-vs-down asymmetry, direct and matched (PRIMARY):** compute the 14 [NEW] reverse-direction runs in this pass (cleaned vectors, frozen estimator/folds/seeds unmodified, input-prep only) so all 15 unordered pairs are bidirectional (30 ordered). Report the per-pair directional gap in each metric, cross-scale smaller->larger vs larger->smaller and tuning base->it vs it->base, matched at every pair.
(c) **Secondary, demoted (report-only):** the artifact-only aggregate over the original 16 keys, carried with the explicit non-matched-composition disclosure (the 16 are not direction-balanced, so that aggregate mixes pair composition; reported for continuity, not as the asymmetry result). Read the asymmetry against the dimensionality story (Item 5).

## Item 8 — family-out split (executes the spec's OWN exploratory arm; runs under existing registration)
Per the frozen P-map spec verbatim: leave-one-family-out splits on the locked k=10 partition, testing extrapolation rather than interpolation. Execute as spec'd, no redesign; label stays **exploratory, ungraded**, per the original spec. Folded into this pass for one execution, but carries its existing registration, not this freeze's.

## Reporting + gate
The whole battery is reported regardless of outcome. Outputs land in `geometry_out_100T/` with manifests + sha256. Coordinator verifies; Ledger integrates grading-free (no verdict attaches); Phase-5 audits before any ledger or paper contact.
