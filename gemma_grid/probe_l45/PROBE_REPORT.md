# L45 Probe — Report

**Verdict: INDETERMINATE** — neither branch triggered — report numbers, escalate to Adam (§7)

E1 implication: unresolved — escalate to Adam; no E1 change either direction.

## Fingerprint / provenance
- hf_revision_resolved: `938270f5272feb02779b55c2bb2fffdd0f53ff0c`
- hf_revision_expected: `938270f5272feb02779b55c2bb2fffdd0f53ff0c`
- torch: `2.8.0+cu128`
- transformer_lens: `unknown`
- tl_compatibility_mode: `False`
- dtype: `bfloat16`
- skip_first_n_tokens: `50`
- n_stories_per_emotion: `120`
- pod_gpu: `NVIDIA H100 80GB HBM3`
- date_utc: `2026-07-27T17:52:19Z`
- pilot: `False`
- **fingerprint gate: PASS**

## RSA (Pearson, 171x171 cosine-structure upper triangle; raw diff-of-means)

| check | value | threshold | result |
|---|---|---|---|
| pipeline control RSA(probe-L31, locked-L31) | 0.9849 | ≥ 0.980 | PASS |
| near-band control RSA(probe-L42, locked-L42) | 0.9890 | ≥ 0.95 | PASS |
| **verdict RSA(probe-L45, locked-L45)** | **0.9486** | ≥0.95 GENUINE / ≤0.90 ARTIFACT | INDETERMINATE |
| ARTIFACT cond RSA(probe-L45, locked-L42) | 0.7955 | ≥ 0.95 | not met |

Full probe×locked RSA matrix:

| probe\locked | L31 | L42 | L45 |
|---|---|---|---|
| L31 | 0.9849 | 0.9682 | 0.7712 |
| L42 | 0.9758 | 0.9890 | 0.8212 |
| L45 | 0.7542 | 0.7955 | 0.9486 |

Calibration ceiling (120-of-1200 subsample structure-RSA, measured): ~0.9856 mean, p05 0.9841. A faithful layer sits near this; the artifact signature sits ~0.15 below.

## Handoff §6.3 — this session does NOT grade the ledger
Hand this report + `probe_l45_means.pt` back to Adam → grid session (Cowork) applies the pre-stated rule to the ledger and cross-checks the RSA arithmetic from the artifact.