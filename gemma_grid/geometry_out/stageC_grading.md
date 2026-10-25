# Stage C grading (tier-2, Holm within families) — PROVISIONAL (C3 pending)

| family | signal | contrast | Δ | floor | p | verdict |
|---|---|---|---|---|---|---|
| tuning | valence_r | 2b | +0.0535 | 0.15 | 0.5902 | descriptive/pairwise (not a Holm unit) |
| tuning | valence_r | 9b | +0.0620 | 0.15 | 0.5197 | descriptive/pairwise (not a Holm unit) |
| tuning | valence_r | 27b | +0.0520 | 0.15 | 0.5893 | descriptive/pairwise (not a Holm unit) |
| tuning | valence_r | POOLED[IT-base] | +0.0558 | 0.15 | 0.32 | NULL_subfloor |
| tuning | arousal_r | 2b | -0.1362 | 0.15 | 0.4416 | descriptive/pairwise (not a Holm unit) |
| tuning | arousal_r | 9b | -0.0233 | 0.15 | 0.8755 | descriptive/pairwise (not a Holm unit) |
| tuning | arousal_r | 27b | -0.0892 | 0.15 | 0.5026 | descriptive/pairwise (not a Holm unit) |
| tuning | arousal_r | POOLED[IT-base] | -0.0829 | 0.15 | 0.3567 | NULL_subfloor |
| tuning | offdiag_pc1_arousal | POOLED[IT-base] | +0.0409 | 0.15 | 0.743 | NULL_subfloor |
| tuning | offdiag_pc2_pleasure | POOLED[IT-base] | -0.0881 | 0.15 | 0.4721 | NULL_subfloor |
| tuning | var_pc1 | POOLED[IT-base] | +0.0387 | 1.0 | 0.5837 | NULL_subfloor |
| tuning | var_pc2 | POOLED[IT-base] | -0.3233 | 1.0 | 1.776e-10 | NULL_subfloor |
| tuning | var_pc3 | POOLED[IT-base] | +0.1545 | 1.0 | 0.001342 | NULL_subfloor |
| tuning | neighbor_cos(9 anchor pairs) | POOLED[IT-base] | +0.0024 | 0.03 | 0.01741 | NULL_subfloor |
| tuning | self_ARI(k=10) | POOLED[IT-base] | -0.0166 | — | 0.7976 | NULL_after_Holm |
| tuning | cohesion_purity_top10 | POOLED[IT-base] | +0.0170 | — | — | descriptive/pairwise (not a Holm unit) |
| tuning | layerstab_all | POOLED[IT-base] | +0.0007 | 0.01 | — | n/a |
| tuning | layerstab_mid_late | POOLED[IT-base] | +0.0142 | 0.01 | — | n/a |
| scale | valence_r | 2b->27b (base) | +0.0096 | 0.15 | 0.9284 | NULL_subfloor |
| scale | arousal_r | 2b->27b (base) | +0.1683 | 0.15 | 0.2371 | NULL_after_Holm |
| scale | var_pc1 | 2b->27b (base) | -4.8848 | 1.0 | 0 | REAL_OPPOSITE_dir |
| scale | var_pc2 | 2b->27b (base) | +0.6286 | 1.0 | 4.705e-13 | NULL_subfloor |
| scale | layerstab_mid_late | 2b->27b (base) | -0.0424 | 0.01 | — | n/a |
| scale | valence_r | 2b->27b (it) | +0.0082 | 0.15 | 0.9273 | NULL_subfloor |
| scale | arousal_r | 2b->27b (it) | +0.2153 | 0.15 | 0.2001 | NULL_after_Holm |
| scale | var_pc1 | 2b->27b (it) | -4.6192 | 1.0 | 0 | REAL_OPPOSITE_dir |
| scale | var_pc2 | 2b->27b (it) | +0.1894 | 1.0 | 0.02898 | NULL_subfloor |
| scale | layerstab_mid_late | 2b->27b (it) | -0.0003 | 0.01 | — | n/a |
| interaction | arousal_r DiD (27B-2B) | ([IT-base]@27B)-([IT-base]@2B) | +0.0211 | 0.15 | 0.9442 | NULL_subfloor |
| exploratory | norm_valence_r | POOLED[IT-base] | +0.0955 | 0.1 | 0.04257 | descriptive/pairwise (not a Holm unit) |
| exploratory | norm_arousal_abs_r | POOLED[IT-base] | +0.0448 | 0.1 | 0.4358 | descriptive/pairwise (not a Holm unit) |

## C4 contrasts vs bootstrap floors

| pair | RSA obs (ceiling) | verdict | Procrustes obs (floor) | verdict | ARI | verdict | loving>median frac | loving-H verdict |
|---|---|---|---|---|---|---|---|---|
| gemma-2-2b-base<->gemma-2-2b-it | 0.9944 (0.9976) | REAL_geometry_difference | 0.0291 (0.0053) | REAL_plane_difference | 0.593 | NULL(>=0.51) | 0.012 | DISCONFIRMED (moves LESS than median) |
| gemma-2-9b-base<->gemma-2-9b-it | 0.9930 (0.9977) | REAL_geometry_difference | 0.0182 (0.0025) | REAL_plane_difference | 0.757 | NULL(>=0.51) | 0.742 | NULL |
| gemma-2-27b-base<->gemma-2-27b-it | 0.9960 (0.9977) | REAL_geometry_difference | 0.0329 (0.0070) | REAL_plane_difference | 0.668 | NULL(>=0.51) | 0.216 | NULL |
| gemma-2-2b-base<->gemma-2-9b-base | 0.9964 (0.9976) | REAL_geometry_difference | 0.0172 (0.0053) | REAL_plane_difference | 0.606 | NULL(>=0.51) | 0.090 | NULL |
| gemma-2-9b-base<->gemma-2-27b-base | 0.9946 (0.9977) | REAL_geometry_difference | 0.0162 (0.0033) | REAL_plane_difference | 0.624 | NULL(>=0.51) | 0.108 | NULL |
| gemma-2-2b-base<->gemma-2-27b-base | 0.9917 (0.9976) | REAL_geometry_difference | 0.0488 (0.0053) | REAL_plane_difference | 0.530 | NULL(>=0.51) | 0.000 | DISCONFIRMED (moves LESS than median) |
| gemma-2-2b-it<->gemma-2-9b-it | 0.9949 (0.9979) | REAL_geometry_difference | 0.0431 (0.0040) | REAL_plane_difference | 0.755 | NULL(>=0.51) | 0.362 | NULL |
| gemma-2-9b-it<->gemma-2-27b-it | 0.9964 (0.9978) | REAL_geometry_difference | 0.0048 (0.0070) | NULL_within_sampling_noise | 0.808 | NULL(>=0.51) | 1.000 | CONFIRMED |
| gemma-2-2b-it<->gemma-2-27b-it | 0.9924 (0.9978) | REAL_geometry_difference | 0.0450 (0.0070) | REAL_plane_difference | 0.704 | NULL(>=0.51) | 0.116 | NULL |