# P-map — Cross-cell linear transport of emotion vectors (FROZEN SPEC, 100T)

**Status: FROZEN before first execution.** Adopted by Adam 2026-08-27 as an unregistered,
pre-stated P-series addition (P-colon pattern: design frozen and hashed before any code runs;
graded against the pre-stated bar below; Phase-5 audited before paper contact; never described
as part of the registered grid). Drafted by the analysis-proposal chat, hardened and frozen by
the GGE1 100T Ledger Chat; executed by the GGE1 100T Data Coordinator's runner lane; graded by
the Ledger Chat; Adam ratifies. The sha256 of this file at commit is the freeze identity.

## Question
Are the emotion representations linearly transportable across the grid — can a linear map fit
on a subset of emotions locate a model's vector for an emotion it never saw fit on?

## Object
The 171 cleaned 100T emotion vectors per cell, each cell at its registered analysis layer
(2b → L17, 9b → L28, 27b → L31). Sources: anchor `data/vectors/emotion_vectors_171_100T.pt`
(sha `6de2e293…971b`, read-only); five cells `gemma_grid/data/<cell>/vectors/canonical_100T.pt`.
100T guard on every cleaned path (`assert "100T" in path`); anchor re-hashed intact at end.

## Preprocessing
Center each cell's 171 vectors by their own mean. (The cleaned vectors are already mean-centered
per cell — the B-ii degeneracy record — so this line is an idempotent safety step, not a
transformation.) No whitening, no row normalization.

## Map class
- **Primary: rectangular semi-orthogonal Procrustes with one global scale.** W (d_source →
  d_target) with orthonormal columns minimizing Frobenius error, closed-form via SVD of the
  cross-covariance, plus a single scalar s fit on the training folds. Zero free regularization
  parameters; geometry-preserving by construction.
- **Secondary (disclosed, descriptive only): unconstrained ridge**, λ chosen by nested 5-fold CV
  inside training folds on the frozen grid λ ∈ {10⁻³, 10⁻², 10⁻¹, 1, 10, 10², 10³} (features
  standardized within the training fold). Reported to show what an unconstrained map adds;
  carries no verdict.

## Pairs
- **Primary pair: 2b-base → 27b-base** (maximum scale gap, tuning held fixed).
- **Secondary: the nine registered G5.1 pairs**, same protocol, reported in one table — not
  individually graded. The reverse primary direction (27b-base → 2b-base) reports in the same
  table, descriptive.

## Split protocol
Five-fold cross-validation over the 171 emotions, **fold seed 0**, every emotion held out
exactly once; each fold's map is fit on the ~137 training emotions and evaluated on its
held-out set. Exploratory, labeled, ungraded: leave-one-family-out splits (the locked k=10
partition), which test extrapolation rather than interpolation.

## Endpoints
Per held-out emotion: (i) retrieval rank of the true target-cell vector among all 171 target
vectors under cosine; (ii) cosine to the true vector. Pooled across folds at the primary pair:
**median retrieval rank** (primary), top-1 and top-5 retrieval accuracy (descriptive), median
held-out cosine (descriptive, read against the ceiling).

## Floor
The identical fit-and-evaluate protocol with the emotion correspondence permuted between source
and target: **10,000 permutations, seed 0**, add-one estimator p = (1 + #{perm ≥ obs}) / 10,001,
on the retrieval statistic. This is the exact null for "the map carries emotion identity."

## Ceiling (branch pre-selected by artifact inventory, not by results)
- **Branch A — replicate vector sets exist on disk for BOTH primary-pair cells** (replicate or
  split-half 171×d matrices from independent story samples): run the identical protocol between
  replicate extractions of the same cell; held-out transport reads as a fraction of that
  replicate ceiling (the house's existing ceiling logic).
- **Branch B — they do not:** grading proceeds against the permutation floor alone; held-out
  cosines are reported beside the G5.1 within-cell RSA ceilings as context with the footing
  difference disclosed (RSA-scale, not retrieval-scale). 
The Coordinator answers the inventory question BEFORE execution; the answer selects the branch
and is recorded in the output. No ceiling construction is invented after results exist.

## Grading bar (pre-stated; the verdict rides on this and nothing else)
- **CONFIRMED:** median held-out retrieval rank ≤ 2 (of 171) at the primary pair AND permutation
  p < .001.
- **NULL:** retrieval resolved from the shuffled floor (p < .001) but median rank > 2.
- **DISCONFIRMED:** retrieval not resolved from the shuffled floor.
Top-1/top-5 accuracy and cosine values are descriptive and carry no verdict. Rationale for the
rank bar, recorded at freeze: the 171-emotion set contains documented synonym clusters
(afraid→scared .987), so top-1 retrieval would grade the vocabulary's cluster density along
with the map; median rank ≤ 2 encodes "the predicted location identifies the emotion up to its
immediate synonyms," which is the claim's meaning.

## Process
Executed on CPU from on-disk artifacts only; outputs to `geometry_out_100T/` as
`pmap_transport_100T.json` (+ per-pair table), SR.5-hashed; NOT added to the flip manifest
(post-flip P-series addition). Producer does not grade. The Ledger Chat grades against the bar
above; Adam ratifies; Phase-5 audit before paper contact. Surprise-gate: any output whose shape
contradicts this spec's structure returns to Adam before insertion. AX firewall (scope per the
2026-08-27 clarification): loving is not the focus of this analysis; it appears as one of 171.
