# Phase 5 — Stage A–D (+ CPU tail) adversarial audit, 100T — FINAL REPORT

**Auditor:** fresh-context Claude Code, no access to any producing session's
reasoning. Every number below was recomputed with independently-written code from
the artifacts; nothing in the repo, ledger, or artifacts was modified.
**Gate posture:** this audit is the process-integrity condition sustaining the 100T
canonical flip. **Prompt:** `PHASE5_STAGE_AD_AUDIT_PROMPT_100T.md`.

## Verdicts per task

| Task | Verdict |
|---|---|
| (a) Anchor identity + gate | SUSTAINED |
| (b) Rebase / cleanup basis | SUSTAINED |
| (c) Stage A | SUSTAINED |
| (d) Stage B + displacement | SUSTAINED |
| (e) Bootstrap gate (B=1000) | SUSTAINED |
| (f) Stage C grading | SUSTAINED (verdicts) — one required ledger value-update noted |
| (g) G1.3 embedding + c3grade incident | SUSTAINED |
| (h) SR.5 completeness | SUSTAINED |
| (i) CPU tail | SUSTAINED (Zhang strand UPHELD-WITH-CAVEAT, unregistered) |
| (j) Item 10 relabel/hash anomaly | SUSTAINED (working anchor untouched; anomaly recorded) |

## Evidence

**(a)** `emotion_vectors_171_100T.pt` sha256 = `6de2e293…971b` ✓. Independent
re-clean of `vectors_raw` against `neutral_100T.pt` at L31: rel = 7.111e-07
(producer 7.11e-07; gate ≤1e-6). k@L31 = 28 (cumvar crosses 0.50 between
PC27=0.4989 and PC28=0.5053).

**(b)** All 5 mode-A cells bit-reproduce `canonical_100T.pt` at float32 (rel
~2.5e-8); anchor cross-validates at rel 7.63e-7 over all 46 layers (E1-produced,
not regenerated). `vectors_raw` identical (rel 0.0). 100T cleaned differs from both
PA (rel 0.28–0.33) and HA (0.29–0.33) at the analysis layer → no PA/HA basis
contamination.

**(c)** All 6 cells × 13 columns of `stageA_cross_cell_table.csv` reproduce exactly
at table precision — including cohesion, mutual_pairs (45/45/45/44/46/47), and the
kmeans-based self_ari_mean/std to 4 dp. SR.4 aggregator column→key map verified
correct.

**(d)** All 9 pairs: RSA cosine, Procrustes-k2 disparity, and cross-ARI match to
0.00e+00; the 9 displacement CSVs match to the 6-dp write floor (~5e-7).

**(e)** All 6 reproduction gates match the npz meta exactly (raw/cleaned ~1e-6,
≤1e-4). 34 sampled bootstrap replicates per cell reproduce from per-story slices +
the 100T basis to ~3e-8; basis source = `cleanup_basis_100T.pt`, k =
11/14/11/19/28/27. No PA-scratch reuse: the replicate statistics are only
reproducible from the 100T basis.

**(f)** My full-precision det matches producer stageA 0.00e+00 (all cells); all 24
Stage C contrasts match to ≤8e-9 (deltas + p's); every verdict reproduces —
var_pc1 scale REAL_OPPOSITE_dir both chains, arousal_r scale NULL_after_Holm, DiD
NULL_subfloor, pooled valence/arousal NULL_subfloor, refit-45 0/6 survives (best
PC=3). c3 finalization at m=10 produces no verdict flips. No verdict flips vs the
ledger.

> ⚠️ **Required ledger update (not a verdict flip):** the ledger currently records
> HA values (e.g. G3.1 var_pc1 drop −3.91/−3.04 pp; 2b-base var_pc1 29.98), which I
> confirmed equal the `geometry_out_HA` table. The 100T canonical values differ and
> are stronger (−4.88/−4.62 pp; 2b-base 31.62). This shift is the flip this audit
> gates; the G3.1/G3.2/G1.3/op1-spectrum numeric entries must be restated to 100T
> when the flip lands. Verdicts are unaffected.

**(g)** Embedding valence/arousal reproduce to ~1e-8; valence present everywhere (r
0.479–0.507, p<0.05), arousal absent everywhere (r 0.125–0.179, p>0.05). Incident
remediation fully confirmed: `c3_grading.json` sha256 = `4cc8584e…` ✓; all 6
`c3_vectors.pt["vectors"]` hashes match the embedded references; `["vectors"]`
differ from PA (rel 0.27–0.47) while `["vectors_raw"]` match PA exactly
(basis-invariant control). Quarantine held: the good and BAD files differ only in
se/p_dependence_aware; my independent gap-bootstrap from the 100T vectors gives se
0.0676/0.0716, p 0.0027292/0.0014063 = the good file (≠ BAD 0.0663/0.0653,
0.00225/0.00046). Nothing reads the quarantined artifact.

**(h)** `preflight_sha256.txt` 9/9 inputs OK; `100T_LOCAL_SHA256.txt` 89/89 outputs
OK; op1 / seam_gap / zhang manifests hash every input (canonical_100T,
per_story_slice, cleanup_basis, ratings CSV, stageA table) and all verify on disk
today. SR.5 tripwire fired (100T stageA table ≠ HA table).

**(i)** Lens anchor onset L23; median self-rank collapse L22 25642 → L23 6282 → L24
5141 → L25 3245, reproduced from `lens_ingredients.pt` via the tokenizer-free
preflight (vocab 256000, 140 fair emotions, ln_final) to ~0.5-rank median tie
precision; anchor gate holds. op1 var_pc1 cross-check PASS. seam_gap anchor L23/L23
→ gap 0.0 hard-pass; the 27b-it seam 23-vs-24 is recorded in
`ledger_reference_mismatch` under the anchor-hard/non-anchor-warn convention
(LC-23). Layerstab per-cell + contrasts derive from verified Stage A (tuning
+0.01419 REAL, scale-base −0.0424 REAL [G3.4 anchor-L45-localized], scale-it
−0.00027 NULL). Zhang z1/z2/z3 manifests verify (19/9/6 inputs) and reproduce
exactly (z1 2b-base raw 0.17816 / cleaned 0.19565; z3 cosines maxdiff 0.0000,
angry↔disgusted percentile 94.8/92.5).

**(j)** Working anchor `emotion_vectors_171_100T.pt` untouched: sha256 `6de2e293…`
and `variant: "neutral_HA"` both present and intact (76 references to the hash
under `repo/`). Recorded anomaly (no action taken): the `variant` field of the 100T
canonical anchor still reads `neutral_HA`; this is the documented item-10 stale
label whose fix is export-only. It must remain untouched here.

## Refutation attempts that failed to refute

Every independently-recomputed number matched the producer within numerical
tolerance (bit-exact in most Stage A/B/C cases). No PA- or HA-cleaned vector was
found reachable by any canonical 100T quantity. The one substantive discrepancy —
HA vs 100T point values in ledger entries — is the intended effect of the flip, not
an error, and shifts no verdict.

## Net

The 100T Stage A–D core + CPU tail is reproducible, canonically clean (100T all the
way down), and internally consistent. The process-integrity condition for the 100T
canonical flip is met, contingent on the ledger's numeric entries being restated
HA→100T at flip time (task-f caveat) and the item-10 variant label remaining
export-only.

Full recompute scripts and a running verdict log are in the auditor's scratchpad
(`AUDIT_NOTES.md`, `task_[a–i]_*.py`). This report returns to the GGE1 100T
grid/ledger session for integration.
