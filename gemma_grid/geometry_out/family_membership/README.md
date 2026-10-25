# Per-cell family membership tables

One table per model. Each gives the ten-family k-means partition of all 171 emotions at that model's analysis layer, on the canonical clustering seed (seed 0). Columns are rank (by the cluster's mean position on the model's own first principal component), n, and members.

| File | Model | Layer |
|---|---|---|
| `family_membership_gemma-2-2b-base.md` | 2B base | 17 |
| `family_membership_gemma-2-2b-it.md` | 2B instruction-tuned | 17 |
| `family_membership_gemma-2-9b-base.md` | 9B base | 28 |
| `family_membership_gemma-2-9b-it.md` | 9B instruction-tuned | 28 |
| `family_membership_gemma-2-27b-base.md` | 27B base, the anchor | 31 |
| `family_membership_gemma-2-27b-it.md` | 27B instruction-tuned | 31 |

## Reading notes

Cluster indices are ordered by each cluster's mean position on that cell's own leading component, so they are internal to a cell. Cross-cell correspondence exists only at the level of the partitions, which is what §4.2 of the paper measures, and never by label: cluster 5 in one cell is not cluster 5 in another, and the tables should not be read across rows. The partition is also soft, as §4.1 reports, so membership at the boundaries is one realization of a graded structure rather than a taxonomy. Cohesion and partition self-agreement per cell are in Table 1 of the paper.

## What the tables support

- The partition-agreement row of the paper's §4.2, where agreement between the partitions of two models (adjusted Rand index) runs from .53 to .81 across the nine comparisons.
- The statement in §7.4 that the family holding compassionate, grateful, kind, and loving never leads the displacement ranking. Those four emotions share one family in every model, which the tables show directly. The ranking itself is in `../analysisA_displacement_bootstrap.json`.

## Producer

`scripts/run_extracts.py`, step 11 of `scripts/run_grid_phase4.sh`. The partition is fit on the cleaned vectors listed by hash in the top-level README.
