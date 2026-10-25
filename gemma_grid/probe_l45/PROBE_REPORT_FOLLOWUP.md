# L45 Probe — FOLLOW-UP report (attenuation-model resolution)

**Verdict (per the pre-registered follow-up rule): GENUINE**
- both seed-vs-locked >= P-0.010 (0.9362); at L45's own measured ceiling

## The resolution at L45

- R_ff = RSA(seed0-L45, seed1-L45) = **0.8954**  (fresh-vs-fresh, no vintage confound)
- P = sqrt(R_ff) = **0.9462**  → at-ceiling prediction for fresh-vs-locked at L45
- seed0-vs-locked L45 = **0.9486**   (margin P-0.010 = 0.9362, P-0.020 = 0.9262)
- seed1-vs-locked L45 = **0.9450**

Note: L45's ceiling P is measured here to be **below** L31's 0.9856 iff R_ff is low — the missing-ceiling assumption that made this INDETERMINATE is now measured directly.

## Attenuation-model self-check (L31, our own data)
- sqrt(R_ff[L31]) = 0.9854  vs  seed0-vs-locked[L31] = 0.9849  |Δ|=0.0005 (model holds)

## Band-unlikeness (ARTIFACT branch — should be LOW if L45 divergence is real)
- seed0-L45 vs locked-L42 = 0.7955 | seed1-L45 vs locked-L42 = 0.7906  (≥0.95 would be band-like)

## Per-layer table

| L | R_ff (s0,s1) | P=√R_ff | s0-vs-locked | s1-vs-locked |
|---|---|---|---|---|
| 31 | 0.9710 | 0.9854 | 0.9849 | 0.9866 |
| 42 | 0.9784 | 0.9892 | 0.9890 | 0.9890 |
| 45 | 0.8954 | 0.9462 | 0.9486 | 0.9450 |

## Provenance (both draws)
- seed0: rev `938270f5272f` | sampling `{'sampling': 'seeded_random', 'seed': 0, 'n_stories': 120}` | fingerprint `PASS` | 120 stories | NVIDIA H100 80GB HBM3
- seed1: rev `938270f5272f` | sampling `{'sampling': 'seeded_random', 'seed': 1, 'n_stories': 120}` | fingerprint `PASS` | 120 stories | NVIDIA H100 80GB HBM3

## §6.3 — this session does NOT grade the ledger
Hand this + both artifacts back to Adam → grid/Cowork session for the ledger, who cross-check the arithmetic. The rule above was pre-registered by that session.