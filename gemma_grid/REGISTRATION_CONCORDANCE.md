# Registration concordance

This is the full registration concordance for the paper, a companion to §1's definition of the registered level and to §3. The preregistration was locked on 2026-07-20, before the five new cells were extracted and with the anchor's published values in hand, and ships beside this file as `PREREGISTRATION.md`, unedited. It fixed the signals, their predicted directions, the floors, the test for each kind of measure, and the correction scheme. The rows in each correction family, the endpoint form of the scale contrast, the self-agreement test, and the resampling unit of the layer-stability bootstrap were fixed in the grading code on 25 July 2026, after extraction and before any contrast was graded (§3), with each cell's own measured values in hand, so those choices carry the pre-stated level of commitment and not the registered one. Four statuses are used. As registered: tested or reported as the registration specified. Operationalized differently: tested, with a named departure from the registration's wording. Reported without a grade: a registered quantity reported descriptively because the registration gave it no test or floor. Not reported: a registered quantity or directional prediction that this paper does not report. The two tables below list every registered signal, floor, and procedure, where the paper reports each, and its status. The paper's Appendix C carries the directional predictions and their outcomes (Table C.1) and repeats every registered item whose reporting departs from, qualifies, or omits the registration (Table C.2). The per-cell family membership tables cited in row C1-8 are in `geometry_out/family_membership/`. Pre-stated analyses fixed after the registration (the transport battery, the §6.3 calibration, the Appendix G prediction, the Appendix H replication rule) are not in these tables and are disclosed where they appear. Registered items whose results are reported in a companion study are not listed.

Paper sections are cited by § number, tables by number, and appendices by their final letters.

## Design and signals

| Registration item | The registration says | Reported at | Status |
|---|---|---|---|
| Analysis layer | Strict two-thirds rule, round(2/3 of n_layers): L17, L28, L31 | §3 | As registered |
| 2B sensitivity | L18 reported as a sensitivity check, "and a small ±band" | §3, §5.3 | L18 as registered. The band beyond L18 is not reported |
| C1-1 Variance shares | PC1, PC2, PC3 share per cell | Table 1 (PC1, PC2), Table 4 and Table E.4 (contrasts) | PC1 and PC2 as registered. PC3's per-cell share is not tabulated, only its tuning contrast |
| C1-2 Valence correlation | PC1 against Russell and Mehrabian pleasure, 45-emotion overlap | Table 1 | As registered |
| C1-3 Arousal correlation | PC2 against arousal, same overlap | Table 1 | As registered |
| C1-4 Off-diagonals and permutation p | PC1 against arousal, PC2 against pleasure, 5,000-shuffle permutation p | §4.1 (permutation p for both axes), Table 4 (tuning contrasts) | As registered for tuning. The off-diagonals' per-cell values are not tabulated |
| C1-5 Negative control | Refit the components on the 45 rated emotions alone and ask whether arousal stays on the second component | §8.2 | As registered |
| C1-6 Vector norms | Norm against PC1 position, norm against absolute PC2 position, norm variability across depth | §7.1, Appendix E | Tuning contrasts as registered, outside the Holm family as registered. Norm variability across depth is not reported |
| C1-7 Neighbor cosines | Nine anchor pairs | Table 4 | As registered for tuning |
| C1-8 Families and self-agreement | k = 10 partition, cross-seed self-ARI | Table 1, Table 4, and the released membership tables | Per-cell values as registered. The tuning contrast on self-agreement has no registered test. Its test, from the spread across seeds, was fixed at grading (§3, §7.1) |
| C1-9 Cohesion and boundary | Ten-nearest-neighbor in-family fraction, boundary emotions | Table 1, §7.1, §7.4, Appendix E | Reported without a grade. The registration gave cohesion a direction and no floor or test (§7.1) |
| C2-10 Cross-layer similarity | Cosine of each layer's 171-by-171 cosine table against the analysis layer's | §8.3 | Operationalized differently: the mean pairwise agreement among fourteen sampled layers, in Pearson, recomputed in cosine with no verdict change. The pairwise form was in the code that set the .01 floor on the day of registration. Appendix H discloses both departures |
| C2-11 Cross-layer neighbor retention | Top-5 retention and top-10 overlap against the analysis layer's neighborhood, all layers | none | Computed in a variant form (a top-5 shared count and a top-10 Jaccard, the registration's assignment reversed) and stored in the released per-cell artifacts. Not graded or reported |
| C3-12 Embedding-table baseline | Valence and arousal correlation, PC variance, similarity structure, ARI against the locked partition, neighbor cosines, all at the static table with the cleanup refit on neutral embeddings | §5.1, Table E.1, §5.3 | Correlations as registered. PC variance, similarity structure, ARI, and neighbor cosines at the table are not reported |
| C4-i Cross-cell similarity | Cosine of two cells' 171-by-171 tables, noise reference measured at the first contrast | §4.2 | As registered. The replicate ceiling was measured on the day the cross-cell point estimates were first computed, after those estimates were written and before any contrast was graded |
| C4-ii Procrustes alignment | Rotation and residual on per-emotion component coordinates, noise reference measured at the first contrast | §4.2 | As registered, same ordering as C4-i |
| C4-iii Cross-cell ARI | Against the .51 floor, the anchor's self-ARI mean minus two standard deviations | §4.2 | As registered |
| C4-iv Per-emotion displacement | Post-Procrustes distance per emotion between cells, noise reference measured at the first contrast | §7.4 | Displacement is a registered signal. The ranking, the readings drawn from it, and the regression in §7.4 are unregistered and labeled so. The registered noise reference was not measured, and §7.4 says so |
| C5 Logit-lens readout | Include; capture the unembedding and final norm weight per cell | §6 | As registered. The readability analyses of §6 carry no registered prediction and §6.3 and Appendix F are labeled accordingly |
| C5 Held-out-text activation | Defer to a follow-up | none | Deferred as registered and not run. §7.2's neutral dialogues and Appendix G's twelve scenarios are separate items and are labeled as such |

## Test definitions, floors, and procedures

| Registration item | The registration says | Reported at | Status |
|---|---|---|---|
| Tuning main effect | Pooled mean of the three per-size base-to-tuned differences, one contrast | §3, Table 4 | As registered |
| Scale main effect | "Trend across {2B, 9B, 27B} within a fixed tuning level" | §5.2, Table E.4 | Operationalized differently: the 2B to 27B endpoint contrast within each chain, fixed in the grading code of 25 July 2026 before any contrast was graded (§3, §5.2) |
| Interaction | Difference in differences, 27B minus 2B | §7.1 | As registered |
| Correction families | Holm within three families, tuning, scale, interaction | §3 | Holm within three families as registered. The registration does not list the rows of each family. The lists (eleven, ten, and one) were fixed in the grading code of 25 July 2026 (§3) |
| Floors | 1.0 points for variance shares, .15 for correlations at n = 45, .10 for the norm correlations at n = 171, .03 for neighbor cosines, .01 for layer stability, .51 for cross-cell ARI | §3, §4.2 | As registered, with each origin stated in §3 |
| Noise reference for i, ii, iv | Measured at the first contrast and applied before any result is read | §4.2 | Measured for i and ii from replicate pairs, before grading (see C4-i). Not measured for iv (§7.4) |
| Correlation tests | Fisher r-to-z, two-sample, n = 45 or 171 | §3 | As registered, with the independence assumption and its effect on the .15 and .10 floors disclosed in §3, §5.3, §7.1, §9.3, and Appendix E |
| Variance and cosine tests | Story-level bootstrap, about 1,000 resamples, each cell's official cleanup basis held fixed | §3, Table 1 | As registered, at 1,000 resamples with the basis held fixed |
| ARI test | Read against the self-ARI band | §4.2 | As registered for cross-cell ARI. No test is registered for a within-cell change in self-agreement (see C1-8) |
| Layer-stability test | "Reuse the §6.4 bootstrap-CI code, cross-cell" | §8.3 | Operationalized differently: the replication study's case bootstrap, ported with the resampling unit changed from the 45 rated emotions to the 171, resampling cells in pairs, 2,000 draws, fixed at grading (§3) |
| Verdict rule | Sub-floor or non-significant after correction is null | §3 | As registered |
| Anchor rerun and reproduction gate | Full L31-truncated rerun of the anchor, reproduction gate against the locked vectors, bootstrap void on failure | §3 | As registered. The gate passed 24 of 24 checks against the replication study's locked vectors, to about 1 × 10⁻⁹ |
| Dimension safety | Between-cell signals operate only on dimension-free objects | §4.2, §4.3 | As registered for the registered comparisons. The transport maps of §4.3 compare vectors across widths, were pre-stated later, and sit outside this constraint, as §4.3 says |
