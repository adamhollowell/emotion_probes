# Table 2 (§5.1) — The registered tuning battery (100T-canonical)

<!-- Built by build_tables_100T.py. Per-size deltas transcribed from Ledger G2.1 'Per-size serialization' (closed 2026-09-05; source of record per coordinator ruling 5), gated here against the pooled stageC deltas. Pooled/floor/p read from stageC_grading.json. Pairwise Fisher p for the two correlation rows are in G2.1. -->

| registered measure | Δ 2B | Δ 9B | Δ 27B | pooled Δ | floor | p |
|---|---|---|---|---|---|---|
| valence r (Δ, tuned − base) | +0.0535 | +0.0620 | +0.0520 | +0.0558 | 0.15 | .320 |
| arousal r (Δ) | -0.1362 | -0.0233 | -0.0892 | -0.0829 | 0.15 | .357 |
| PC1 variance share (Δ pp) | +0.2706 | -0.6908 | +0.5362 | +0.0387 | 1.0 pp | .584 |
| PC2 variance share (Δ pp) | -0.0865 | -0.3576 | -0.5258 | -0.3233 | 1.0 pp | 1.8e-10 |
| PC3 variance share (Δ pp) | -0.2049 | +0.3829 | +0.2856 | +0.1545 | 1.0 pp | .001 |
| off-diagonal PC1·arousal (Δ) | +0.0316 | +0.0334 | +0.0579 | +0.0409 | 0.15 | .743 |
| off-diagonal PC2·pleasure (Δ) | -0.0807 | -0.0364 | -0.1471 | -0.0881 | 0.15 | .472 |
| neighbor cosine, 9 anchor pairs (Δ) | +0.0012 | +0.0001 | +0.0059 | +0.0024 | 0.03 | .017 |
| partition self-agreement (Δ ARI) | -0.0058 | -0.0399 | -0.0041 | -0.0166 | — | .798 |

**Binding footnote (travels with this table, verbatim).** The neighbor-cosine per-size values come from the producer's registered rule, a bootstrap-mean cosine at the anchor's nine official top-1 pairs. The per-cell point estimates in the Stage A artifacts are a different, non-graded object and must never be substituted.
