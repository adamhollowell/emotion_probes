# OP.3 Band Sweep — pod job spec (batches at the 100T session's 27b-base load)

**Commissioned:** coordination chat → Grid Chat, 2026-08-10. **Owner:** GGE1.
**Question (OP.3 remainder):** the late-band shape L42–L45 of base-27B rests on the
round-one extraction with no per-layer replication between the probed points {42, 45};
the divergence's mechanism is open. This sweep closes the band-coverage gap and takes
the first mechanism measurement.

**Registration posture (Grid Chat call, flagged for Adam's veto):** unregistered
characterization under the `PROBE_L45_CLAUDE_CODE_HANDOFF.md` discipline — every decision
rule pre-stated below, BEFORE any data exists; per-layer ceilings measured, not assumed
(the run-1 INDETERMINATE lesson); producing session does not grade. No registered claim
is created or touched; results land as an OP.3 update graded by the Grid Chat.

## Job

- **Job ID:** `GGE1-G3` (add to `coordination_100T/manifest_GGE1.md` §B; batches into the
  shared-head-A 27b-base load — zero extra model loads; the tracker's "small-job bundle
  at 27B load" slot).
- **Extraction:** per-emotion mean vectors at layers **{31, 42, 43, 44, 45}** (resid_post,
  locked regime), from **two disjoint seeded-random-120 story draws per emotion**
  (`probe_l45/_sampling.py` sampler verbatim; seeds 0 and 1; seed-0 indices must
  reproduce the existing probe draw exactly — assert against `probe_l45/` provenance).
  L31 is the pipeline control; 43/44 are the new band points; 42/45 re-extract for
  within-run consistency. RAW structures only — basis-immune by construction (no
  cleanup applied), so this job is valid under HA and 100T alike and neither blocks it.
- **Gates:** fingerprint gate (anchor revision `938270f5`, per
  `hf_model_revisions_VERIFIED.json`); L31 control vs locked structure (point
  prediction 0.9849 for the seed-0 draw, per the probe record); SR.5 input-hash
  manifest; pod-side sha256 + local `shasum -c` before teardown.
- **Est. GPU:** ~1–2 h at the existing load (probe_l45 precedent; five layers capture in
  the same forward passes). Checkpointable per seed.
- **Outputs:** `gemma_grid/probe_band/means_L{31,42,43,44,45}_seed{0,1}.pt` + provenance
  dir per the probe_l45 pattern.

## Pre-stated analysis (CPU, post-session; graded by the Grid Chat, not the producer)

1. **Per-layer ceilings:** R_ff(ℓ) = RSA(seed0-ℓ, seed1-ℓ); P(ℓ) = √R_ff(ℓ). The L45
   probe measured P(45)=0.9462 and P(31)=0.9856 — both must reproduce within ±0.005
   (internal consistency gate) before anything else is read.
2. **Band shape:** RSA(fresh-ℓ, locked-ℓ) for ℓ ∈ {42,43,44,45} against P(ℓ)−0.010
   (the validated attenuation rule): each layer graded GENUINE / vintage-flavored /
   INDETERMINATE by the probe record's exact three-branch rule. The open question this
   answers: is the divergence a cliff at L44–45 (per Z2's final-two-layer localization)
   or a graded slope across the band? Pre-stated statistic: corr(ℓ, next-ℓ) sequence
   plus each fresh-ℓ's band-unlikeness vs locked-L42.
3. **Mechanism, first measurement:** dimension-level attribution of the L44→L45
   divergence — share of the RSA drop carried by the top-k activation dimensions
   (pre-stated k = 1, 10, 100; dim 4140, M.5's massive-activation dimension, reported
   individually). Distinguishes "few massive dims rotate" from "diffuse reorganization."
   No verdict attached — this seeds the OP.3 mechanism question, nothing more.
4. **IT comparison (CPU-only, free):** the same statistics on 27b-it's existing all-layer
   means (`canonical_100T.pt::vectors_raw`) for the same band — quantifying the "IT's
   milder dip" note with the same instrument.

## What this does NOT do

No cleanup-basis contact, no registered-row contact, no E1 grading (E1's late-band
numbers gain or lose nothing until the Grid Chat writes the OP.3 update). If any gate
fails, the job stops and returns the numbers — the probe record's escalation pattern.
