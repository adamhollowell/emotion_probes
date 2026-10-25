# L45 Probe — Handoff to Claude Code

**From:** the Gemma Grid Phase 4/5 session (Cowork) · 2026-07-27
**To:** Claude Code, working in this repo (`emotion_probes_project/`), self-contained
**Decision authority:** Adam (GATE discipline applies — see §7)
**Ordered at GATE 4, option A:** probe prepped now, run batched with the E1
pre-submission verification pass.

---

## 1. What this is and why (read first)

The grid study found that base-27B's cosine structure at its **final sampled
layer, L45**, diverges from the rest of the model (layer-structure corr 0.84
cleaned / 0.797 raw, vs ≥0.98 for every other layer in every cell). Two
audited hypotheses remain (`gemma_grid/geometry_out/phase5_layerstab_addendum.md`):

- **GENUINE:** base-27B's late layers really reorganize (plausible — it's
  where base-model next-token specialization lives; 27B-IT shows a milder
  same-direction dip; E1 independently reported the derived 0.949 band value).
- **VINTAGE ARTIFACT:** something specific to round one's single full-depth
  extraction at L45. No defect is suspected; the point is that **no second
  full-depth base-27B extraction exists** — the grid round's anchor rerun was
  L31-only by design, so L42–46 were never independently re-captured.

Stakes: for the grid, this only settles the wording of a flagged
not-a-finding row (ledger v2 G3.4/OP.3). **The bigger client is E1**, which
reports late-layer band statistics (mid-late stability 0.949, Table 2 band
values) resting on that same single extraction, and is not yet submitted.
Both outcomes are useful; the probe is the only way to know which.

## 2. The probe, exactly

One fresh extraction on a GPU pod: `google/gemma-2-27b` (base), forward pass
over a **fixed 120-story-per-emotion subsample** of the canonical corpus
(all 171 emotions), truncated at the final block, capturing `hook_resid_post`
at layers **{31, 42, 45}** only. Then per-emotion means → raw diff-of-means
(subtract unweighted grand mean over the 171 emotion means) → 171×171 cosine
structure per layer → compare against the locked structures from
`data/vectors/vectors_171.pt` (`vectors_raw`, same layers).

- **RAW vectors only.** No neutral dialogues, no cleanup basis. The anomaly
  is LARGER on raw (0.797), so raw answers the question and removes a whole
  subsystem from the probe.
- **L31 is the built-in pipeline control** (costs nothing — the pass goes
  through it) and **L42 is the near-band secondary control**.
- n=120 is calibrated as decisive — see §5.

## 3. Preconditions (hard gates, in order)

1. **Fingerprint gate** (required by `gemma_grid/handoff/RUN_MANIFEST.json`
   → `REQUIRED_GATE_before_future_extraction`): before extraction, resolve
   the HF revision actually being downloaded and assert it equals the
   recorded SHA for `google/gemma-2-27b` in
   `gemma_grid/handoff/hf_model_revisions_ASSUMED.json`
   (`938270f5272feb02779b55c2bb2fffdd0f53ff0c`). Pin
   `revision=` explicitly in the load call. Record the resolved SHA in the
   probe's provenance output. If it does not match: STOP, report to Adam.
2. **Invariants** (`gemma_grid/CLAUDE.md` — read it in full): raw HF
   numerics — `tl_compatibility_mode=False` exactly as the anchor (this is a
   deliberate per-call-site override; see the `gemma-27b` branch of
   `scripts/run_extraction_pipeline.py`); bfloat16; `hook_resid_post`; raw
   story text (no chat template — moot here, base model); the corpus file
   verbatim (`data/stories/stories_171_canonical_clean.jsonl`, filter
   `variant == "canonical"`); pooling = mean over tokens from position 50
   (BOS-inclusive indexing), short-text fallback `start = min(1, n-1)` —
   copy `emotion_probes/activations.py::_mean_resid_post_single`, do not
   reimplement.
3. **Layer indexing:** assert `n_layers == 46` from the model config at
   load. L31 is the derived analysis layer (round(2/3·46)); 42 and 45 are
   the probe's named targets in the same indexing as `vectors_171.pt`
   (i.e., `blocks.{i}.hook_resid_post`). Do not renumber anything.
4. **Subsample (AMENDED 2026-07-27, approved by the grid session):**
   **seeded-random 120 per emotion, without replacement** — NOT first-120:
   the corpus is topic-structured within each emotion, so a corpus-order
   prefix risks topic skew, and the §5 calibration distribution was measured
   on seeded random 120-of-1200 draws, so random sampling is what the
   locked thresholds characterize. Scheme: one draw per emotion via
   `np.random.default_rng(np.random.SeedSequence([20260727, i]))` with `i` =
   the emotion's index in the ALPHABETICAL canonical list, choosing 120 of
   that emotion's canonical stories indexed in corpus order. Document the
   scheme in the probe report/provenance and record the per-emotion story
   indices in the artifact (already contracted).
5. **Gate discipline:** dry-run (CPU shapes, synthetic) → tiny pilot on the
   pod (3 emotions × 5 stories; assert layer/dim/shape sanity) → full probe.
   Show Adam the pilot output before the full run. Do not skip the pilot.

## 4. Build notes (reuse, don't fork)

- The extraction machinery exists: `gemma_grid/scripts/run_cell_extraction.py`
  mode B is exactly this pattern (truncated forward, per-story slice at named
  layers). Adapt with a probe config (three capture layers, 120-story
  subsample, no neutrals, no reproduction-gate-vs-vectors business — the
  probe's own controls replace it). `stop_at_layer=46`.
- Pod provisioning pattern: `scripts/runpod_bootstrap.sh`, `docs/GPU_SETUP.md`,
  and `gemma_grid/PHASE2_PILOT.md` (HF token gotcha documented there: use a
  persisted token file, not env-only). All previous pods are terminated —
  this is a fresh pod. Budget expectation: single A100-80GB/H100; ~60GB
  model download; ≤1 h compute for 171×120 stories at 3 captured layers;
  order of $10–15 total.
- Output artifact (single file):
  `gemma_grid/probe_l45/probe_l45_means.pt` containing
  `{emotions: [171 names], layers: [31,42,45], means: (171,3,4608) float32,
  story_indices: {emotion: [120 ints]}, provenance: {hf_revision_resolved,
  torch/transformer_lens versions, tl_compatibility_mode, pod_gpu, date}}`.
  Plus `SHA256SUMS.txt` for it, both pulled to Dropbox. Small (~12 MB) — no
  disk-management gymnastics needed.

## 5. Analysis + pre-stated decision rule (LOCKED before the run — do not adjust after seeing results)

Compute, for each probe layer L ∈ {31,42,45}: raw diff-of-means from the
probe means; upper-triangle cosine-structure vector; Pearson RSA against the
locked `vectors_raw[:, L, :]` structure from `vectors_171.pt`.

**Calibration (already measured, no GPU — from the anchor's per-story slice,
30 draws of 120-of-1200 stories without replacement, first-120 is one such
draw):** subsample structure-RSA to the full-1200 structure at L31 =
**mean 0.9856, min 0.9833, p05 0.9841**. So ~0.985 is the ceiling any
faithful probe layer can reach against its locked counterpart, and the
artifact signature (~0.80–0.84) sits ~0.15 below it against noise of ~0.002.

| Check | Rule | Meaning |
|---|---|---|
| Pipeline control | RSA(probe-L31, locked-L31) ≥ **0.980** | must pass or the probe is VOID — stop, diagnose, do not interpret L42/L45 |
| Near-band control | RSA(probe-L42, locked-L42) ≥ **0.95** | expected under BOTH hypotheses; if it fails while L31 passes, suspicion widens to a broader late-layer vintage issue — report as ESCALATE (E1-relevant), do not force a verdict |
| **Verdict** | RSA(probe-L45, locked-L45) ≥ **0.95** → **GENUINE** (fresh extraction reproduces the divergent L45 structure) | G3.4's caveat upgrades to "replicated, genuine property"; E1's late-layer numbers gain an independent replication |
| | RSA(probe-L45, locked-L45) ≤ **0.90** AND RSA(probe-L45, locked-L42) ≥ 0.95 → **ARTIFACT** (fresh L45 is well-behaved and band-like) | locked L45 is a single-extraction artifact; grid rows updated; **E1 must review its late-layer band numbers before submission** |
| | anything else → **INDETERMINATE** | report numbers, no interpretation; escalate to Adam |

## 6. Deliverables back

1. `gemma_grid/probe_l45/probe_l45_means.pt` + `SHA256SUMS.txt` (Dropbox).
2. `gemma_grid/probe_l45/PROBE_REPORT.md`: fingerprint-gate record, pilot
   record, the three RSA values, the rule-table verdict, and — explicitly —
   the E1 implication line for whichever outcome occurred.
3. Do NOT update `Gemma_Grid_Findings_Ledger v2.md`. Per project discipline
   the producing session does not grade its own result: hand the report back
   to Adam, who will return it to the grid session (Cowork) for ledger
   updates against the pre-stated rule above (and a quick cross-check of the
   RSA arithmetic from the artifact).

## 7. Stop-and-ask triggers

Fingerprint mismatch; pilot shape/layer assert failure; pipeline control
below 0.980; any need to deviate from an invariant in §3 (cost/speed is
never a reason — CLAUDE.md rule); INDETERMINATE or ESCALATE outcomes.
