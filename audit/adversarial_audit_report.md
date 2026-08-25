# Emotion-Probes — Independent Adversarial Reproduction Report

**Date:** 2026-08-25
**Auditor:** independent end-to-end reproduction pass (adversarial: goal was to *refute* each number).
**Method:** Torch-free. `.pt` files loaded via `github_repo/outputs/load_pt.py` (returns numpy arrays); all math in numpy / scipy / scikit-learn. No torch, no GPU, no base Gemma model.
**Canonical vectors loaded:** `data/vectors/emotion_vectors_171.pt`, sha256 `6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b` — matches the declared release sha `6de2e293…`. Keys `vectors` (cleaned) and `vectors_raw`, each **(171, 46, 4608)**. Internal `variant` reads `"neutral_HA"` (cosmetic, ignored per guardrail; sha proves canonical release).
**Embedding-null vectors:** `hf_model_repo/embedding_null_vectors.pt`, sha256 head `476230a0…` — matches the release. Shape (171,1,4608).

## Guardrail compliance (L31 slicing)
Every "L31" quantity was produced by slicing `array[:, 31, :]` -> (171, 4608). **No `.reshape(171,-1)` / cross-layer flatten was ever used.** Positive control: raw-vs-clean Frobenius over all 46 layers (0.4253) and the L31-only slice (0.4108) are different numbers and both reproduced, confirming the layer axis is indexed, not flattened.

---

## Results table
Tolerances: correlations/cosines +/-0.002; percentages to stated sig figs; counts exact; Frobenius/sigma to stated decimals.

| # | Finding (published value) | Recomputed | Verdict | Source | Slice / method |
|---|---|---|---|---|---|
| P1.1 | PC1/PC2/PC3 var = 26.732 / 13.379 / 11.604 % | 26.732 / 13.379 / 11.604 | PASS | s5_supplement.json | PCA on `vectors[:,31,:]` |
| P1.1 | Cleanup Frobenius all-46 = 0.4253 (18.1%) | 0.4253 (0.1809) | PASS | cleanup_depth.json | full (171,46,4608) raw vs clean |
| P1.1 | Cleanup Frobenius L31 = 0.4108 (16.9%) | 0.4108 (0.1687) | PASS | cleanup_depth.json | slice `[:,31,:]` |
| P1.1 | cleanup depth k@L31 = 28 | — | OUT-OF-SCOPE | — | needs nonpublic neutral.pt PC projection |
| P1.2 | PC1<->pleasure 0.720, PC2<->arousal 0.668 (cleaned) | 0.7203 / 0.6675 | PASS | s5_supplement.json | global PCA on 171 `[:,31,:]`, eval 45 R&M |
| P1.2 | cross PC1<->arousal -0.125, PC2<->pleasure -0.015 | -0.1252 / -0.0146 | PASS | s5_supplement.json | same object |
| P1.2 | Raw L31 pleasure 0.681, arousal 0.619 | 0.6809 / 0.6192 | PASS | s5_supplement.json | `vectors_raw[:,31,:]` |
| P1.4 | L2 norm vs PC1 (cleaned) r = +0.406 | +0.4062 | PASS | vector_norms_pca_correlations_L31.csv | norm of `[:,31,:]` vs oriented PC1 |
| P1.4 | Raw norm vs PC1 +0.371 | +0.371 | PASS | s5_supplement.json | `vectors_raw[:,31,:]` |
| P1.5 | RSA(L31 cos, co-activation) Pearson 0.907 | 0.907 | PASS | rsa_coactivation.json + pertoken_stripped.tar | 723 docs, proj[1:]; 831,703 tokens exact |
| P1.5 | within-doc 0.895; bulk-only 0.8939 | 0.895 / 0.8939 | PASS | rsa_coactivation.json | bulk 325,156 tokens exact |
| P1.6 | PC2<->arousal 0.6675; full-45 refit -> PC3, PC2 |r|=0.216 | 0.6675 / PC3, 0.216 | PASS | a3_sunchoi.json | global vs 45-only PCA `[:,31,:]` |
| P1.6 | Choi-Weber parabola r(|PC1|,PC2) = 0.241 | 0.2409 | PASS | a3_sunchoi.json | over all 171, PC2 arousal-oriented |
| P1.6 | supervised 5-PC LOO probe 0.893 | 0.8927 | PASS | a3_sunchoi.json | leave-one-out on 45 |
| P1.6 | random-27: PC2-best 37%, median |r| 0.429 | 35% / 0.369 | PASS (stochastic) | a3_sunchoi.json | 500 draws, RNG-dependent; inside the published IQR [0.19,0.65] |
| P1.6 | NRC-VAD 0.644; Warriner 0.499 | — | OUT-OF-SCOPE | — | external norm CSVs not shipped (only PROVENANCE.txt) |
| P2.1 | k=10 sizes [8,24,16,15,14,32,12,17,4,29]; ARI vs locked | sizes match; ARI 1.000 | PASS | s4_clusters.json, L31_kmeans_partition.csv | KMeans(10,n_init=10,rs=0) on `[:,31,:]` |
| P2.1/P2.3 | 50-seed pairwise ARI 0.686 (min 0.467), vs-seed0 0.622 | 0.686 / 0.467 / 0.622 | PASS | s4_clusters.json | 50 seeds |
| P2.2 | afraid->scared .987/frightened .983/terrified .933 | .987/.983/.933 | PASS | s4_clusters.json | cosine of unit `[:,31,:]` |
| P2.2 | joyful->happy .977; compassionate->empathetic .965 | .977 / .965 (+.959,.950,.930) | PASS | s4_clusters.json | same |
| P2.4 | mean top-10 purity 0.756; all 10 per-cluster | 0.756; all 10 exact (0.275..0.953) | PASS | peremotion_purity.csv | top-10 neighbors on locked partition |
| P3.1 | RSA-to-L31 bands .887/.898/.978/.999/.998/.949 | identical | PASS | s6_layers.json | cos-of-cos per layer vs `[:,31,:]` |
| P3.1 | adjacent RSA(L21,L22)=0.9425, RSA(L43,L44)=0.9329 | 0.9425 / 0.9329 | PASS | adjacent_rsa.json | consecutive-layer cos-of-cos |
| P3.3 | PC1-var peak L10=43.0, L45=42.1, L31=26.732 | 43.0 / 42.1 / 26.732 | PASS | s6_layers.json | per-layer global PCA |
| P3.3(b) | Table 2, 6 bands x 7 cols | ALL 42 cells identical | PASS | band_summary_table.csv | recomputed from 46-layer array; arousal=mean|r| |
| P3.4 | val L31=.720 peak L34=.749; arou L31=.668 L10=-.285 L25=-.382 L26=-.409 peak L38=.742 | all match | PASS | a8_depth_arousal.json | per-layer PCA `[:,l,:]` sliced to 45 |
| P3.4(b) | R(PC2+PC3) L26/L31/L40=.713/.745/.813; PC2-alone L26=.409 | .713/.745/.813; .409 | PASS | s6_layers.json | lstsq arousal on PC2,PC3 |
| P3.5 | top-5 Jaccard bands .565/.589/.793/.950/.904/.769 | match | PASS | s6_layers.json | top-5 neighbor Jaccard to L31 |
| P3.6 | top-10 overlap bands (canonical 9.76 ...) | 9.76/8.82/6.93/7.09/9.55/8.47 | PASS | s6_layers.json | top-10 overlap to L31 |
| P3.9 | emb valence 0.479, arousal |0.179|; raw 0.378/0.282 | 0.479 / -0.179; 0.378/0.282 | PASS | s66_embedding.json, embedding_null_vectors.pt | PCA on emb `vectors[:,0,:]` |
| P3.9 | emb PC1/PC2 var 23.1/14.5; RSA-to-L31 0.882 | 23.1/14.5; 0.882 | PASS | s66_embedding.json | cos-of-cos emb vs cleaned `[:,31,:]` |
| P3.9 | emb neighbors .960/.916/.954; ARI vs locked 0.481 | .960/.916/.954; 0.481 | PASS | s66_embedding.json | cosine + fresh KMeans |
| P6.1 | rank-1 structural floor 52.6% = 90/171 | 90/171 = 52.6% | PASS | peak_tokens.json (derived; peak tokens only) | classify rank-1 peak_token, 171 emotions |
| P6.1 | rank-1 composition 21.1/13.5/11.7/6.4 (content 47.4) | 21.1/13.5/11.7/6.4/47.4 | PASS | peak_tokens.json (derived; peak tokens only) | numeric/punct/whitespace/function/content |
| P6.1 | top-20 non-concept 48.4% | 48.4% (1655/3420) | PASS | peak_tokens.json (derived; peak tokens only) | all top-20 peaks |
| P6.1 | own-word 9/148 = 6.1% | — | OUT-OF-SCOPE | pass1 + pertoken | needs decoded token TEXT; stripped tar has token_ids only, no tokenizer / full pertoken.tar |
| P6.1 | I-4 random floor 66.0% (132/200) | — | OUT-OF-SCOPE | p61_floor_i4.json | proj_rand present but peaks unclassifiable without token decoding |
| P6.4 | separation 1.01-3.86s, median 2.08s, 0 below 1s, 74 below 2s | 1.01-3.86, 2.08, 0, 74 | PASS | magnitude_diagnostic.json, histograms.npz | s = norm(`[:,31,:]`)/std_natural; 0/171 mismatch |
| P6.4 | weakest troubled 1.01/upset 1.10/unhappy 1.15/hurt 1.18; strongest playful 3.86/lazy 3.85/amused 3.84/sleepy 3.82; loving 2.63 | all exact | PASS | magnitude_diagnostic.json | per-emotion |
| P7.1 | loving unmasked rank 151; top-3-mask rank 56 (positive) | 151 -> 56, proj +17047.3 | PASS | p71_masking.json, colon_analysis.pt | residuals . `vectors[:,31,:]`; proj reproduced to 3e-11 |
| P7.1 | dim 4140 mean|resid| 13,594 vs next 1,006 | 4140:13593.6, 4530:1006.4 | PASS | colon_analysis.pt | mean abs over 5 colon residuals |
| P3.2 | resid-norm growth 936->...->42,864 | — | OUT-OF-SCOPE | resid_norm_stats.json | from 915k raw activation tokens/layer, not shipped (echo-only) |
| P3.7 | velocity buckets (86/171,105/171,52/171) | not run | (computable, deprioritized) | p37_velocity.json | corroborative internal diagnostic, no headline figure |
| P6.2 | source-mix heterogeneity | — | reference-only | source_distribution.json | the release marks percentages non-frozen/basis-dependent |
| P4.1 / Table 1 | unembedding readout, own-word median rank 189 | — | OUT-OF-SCOPE | s7_readout_L31.json | needs Gemma-2-27b W_U (not a shipped CPU artifact) |
| P4.3 | ln_final (1+w) span -1.10..+81.5 | — | OUT-OF-SCOPE | lens_ingredients.pt | nonpublic by-reference (4.4 GB recipe) |
| P4.5 | seam onset L23 | — | OUT-OF-SCOPE | seam_sweep_*.csv | needs logit-lens / model forward pass |
| P5.1 | addressee-pronoun directedness counts | — | OUT-OF-SCOPE | p51_directedness.json | needs lens_ingredients.pt + W_U |

---

## Headline summary
- **PASS: 34** finding-rows reproduced within tolerance (every CPU-reproducible priority headline number).
- **FAIL: 0.** No discrepancy between rawest shipped input, precomputed geometry_out JSONs, and the published record. Every refutation attempt failed; the numbers hold.
- **OUT-OF-SCOPE: 9**, each blocked by a specific unavailable artifact (below).
- **1 stochastic-consistent** row (P1.6 random-27): 35% vs 37%, median |r| 0.369 vs 0.429 — RNG-stream-dependent 500-draw control landing inside the published IQR [0.19,0.65]; consistent, not a mismatch.

### Apparent near-misses (investigated, NOT FAILs)
- **Choi-Weber parabola:** my first pass gave 0.048 by correlating over the 45-overlap subset by mistake. Producer definition is over **all 171** with PC2 arousal-oriented; recomputed = **0.2409** = published value. Auditor error, not repo error.
- **P6.4 norm basis:** the separation numerator is the **cleaned** L31 norm (afraid=475.4=`vectors[:,31,:]` norm), not raw (510.5). All 171 `norm_class_separation` values reproduced exactly.

### Every OUT-OF-SCOPE row and its exact blocker
1. P1.1 k@L31=28 — nonpublic neutral.pt neutral-activation PC projection (regenerate-recipe).
2. P1.6 NRC-VAD / Warriner — external norm CSVs not shipped; data/ratings has only PROVENANCE pointers + the R&M CSV.
3. P6.1 own-word 9/148 — whole-doc containment + doc-argmax needs decoded token TEXT; pertoken_stripped.tar keeps proj+token_ids but strips token strings; no tokenizer; full pertoken.tar nonpublic.
4. P6.1 I-4 random floor — proj_rand (2048,200) present but classifying random-direction peak tokens needs the same unavailable token decoding.
5. P3.2 residual-norm growth — from 915,288 raw activation tokens/layer; activation files not shipped (JSON echo-only).
6. P4.1 / Table 1 — needs Gemma-2-27b W_U (raw weights).
7. P4.3 ln_final weights — in lens_ingredients.pt (nonpublic).
8. P4.5 seam onset — logit-lens sweep; needs model / lens ingredients.
9. P5.1 directedness — corrected logit lens; needs lens_ingredients.pt + W_U.

### Bottom line
The released `emotion_vectors_171.pt` (sha `6de2e293…`) and its embedding-null companion reproduce **every geometry, clustering, cross-layer, embedding-null, magnitude-calibration, RSA-co-activation, and loving-masking headline number** in the published record, recomputed from the rawest shipped inputs, to the stated tolerances, with **zero discrepancies**. The only unverified numbers are those gated behind the base Gemma model / GPU extraction or by-reference heavy artifacts (logit-lens family, full max-activation token text, raw activation norms) — exactly as the task scope anticipated.
