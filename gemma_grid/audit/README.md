# The audit record

Every claim in the paper that carries a graded verdict passed an adversarial audit. Each audit was a fresh-context language-model session (Claude) with no access to the reasoning of the analysis it audited, instructed in writing to refute each claim by recomputing it from the artifacts. Most audits were given the claimed values as targets. A blind arm was given none and wrote down every number before opening the artifact to compare. A separate fresh-context session proposed each grade, and the final rulings were the author's.

The seven audits below cover every graded claim in the paper and the two unregistered analyses added in review. Each folder holds the prompt as issued and the report as returned, reproduced byte for byte. File paths inside them are the paths of the working sessions that issued and ran each audit, and do not match this release's layout.

| # | Folder | Date | What it covers |
|---|---|---|---|
| 1 | `01_stages_a_to_d/` | 2026-08-20 | The full pipeline after the corpus remediation, Stages A to D and the CPU analyses that follow them, recomputed stage by stage |
| 2 | `02_prepaper_batch/` | 2026-08-26 | Five charges: the base against tuned welfare-axis scan, the layer-band sweep of the late-layer probe, the colon-token probe, the valence asymmetry in readability onset, and three additional depth analyses (residual-norm growth, arousal on the second and third components, layer velocity) |
| 3 | `03_final_round/` | 2026-09-02 | Nine charges: the transport battery and its permutation floor, the frequency control, the displacement model, the bipolarity cosines, the displacement rank-stability bootstrap, the dependence-aware bootstrap, the spectrum measures, the seam and onset bootstrap, and one charge on files from a companion analysis that are not part of this release |
| 4 | `04_transport_battery_and_onset/` | 2026-09-03 | Seven charges on the transport battery, the onset curve, and the covariate transfer, with its input manifest |
| 5 | `05_9b_centered_profile/` | 2026-10-09 | The 9B centered activation-profile diagnostic, the one number previously disclosed as unaudited, with an independent recomputation (`scripts/audit_T3_9b_centered_diagnostic.py`) |
| 6 | `06_readout_counts_and_section7_nulls/` | 2026-10-14 | The §6.1 readout counts from the blind grading, the layer 18 bootstrap of §5.3, and the layer-permutation null of §7.3 |
| 7 | `07_review_analyses/` | 2026-10-14 | The two unregistered analyses added in review: the static token-embedding-table transport of §4.3 (`pmap_embedding_table_transport.json`) and the dependence-aware bootstrap of §7.1. This folder also holds a correction notice filed beside the commission, which overstated when the commission was written. The commission itself is unedited |

Audit 6 refuted two procedure descriptions written inside artifacts, not the computations. The §7.3 layer-permutation file states its split range as `[2,L-2]`, where the code searches splits 2 through L minus 3. The layer 18 file stated its p value formula without the standardization by the bootstrap standard error that the code applies. Neither refutation reaches a number in the paper. The second description has been corrected in the artifact. The first artifact is kept unedited and the correction is recorded in the README.
