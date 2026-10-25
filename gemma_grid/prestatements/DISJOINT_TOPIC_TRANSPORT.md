# Pre-statement: disjoint-topic transport and a bag-of-words baseline

Study: GGE1 (Gemma 2 grid)
Status: FILED 2026-10-19, before any topic-split outcome metric exists. Cleared by the Ledger 2026-10-19 (cleared draft sha256 prefix 70278906).
Level: pre-stated and exploratory. It re-grades nothing registered or pre-stated in the transport battery.

## Question

Under one fixed partition of the story topics, does transport (§4.3) still meet its retrieval criterion when the source cell's emotion vectors and the target cell's come from disjoint sets of topics, so that no story and no topic is shared across a map?

## Data

- Each cell's per-story slice at its analysis layer (2B at L17; L18 is not used).
- Topic identity for every story from the corpus, through the gated topic join in the dispatch.
- Cleanup: every half-mean is cleaned with the cell's 100T cleanup basis, rebuilt from the cell's 100T neutral file (`<cell>__neutral_100T.pt`, or the project-root `neutral_100T.pt` for 27B base) under the main analysis's 50-percent-variance rule, and used only after it passes the dispatch's cleanup-basis gate against the stored `<cell>__cleanup_basis_100T.pt`. The slice files' neutral rows are never read.

## Partition

- Topic identity is the corpus `topic` field. All 171 emotions share the same 100 topics.
- One partition: numpy `RandomState(20261019).permutation(100)` applied to the sorted list of topic identities, first 50 to set A, the rest to set B. The same partition for every emotion and cell. The 50 set-A identities are written to the output.
- Random halves, for the matched control: for each emotion, a generator re-seeded as numpy `RandomState(20261019)` computes `permutation(1200)` over that emotion's stories in slice order, the first 600 positions go to half R1 and the rest to R2, ignoring topic. Re-seeding for each emotion draws the same positions in every emotion. Because each emotion's topics sit in a different order, the same positions do not select the same topics.
- Each half-vector is the mean over that emotion's 600 stories in the half, differenced against the half's grand mean and cleaned as above.

## Fits

All fits use the battery's map (semi-orthogonal Procrustes plus global scale), the battery's frozen five-fold assignment of emotions, and the battery's scoring against all 171 target vectors.

1. **Cross-cell, disjoint topics.** The battery's sixteen fits, source from set A and target from set B, then with the assignment swapped. 32 fits.
2. **Cross-cell, matched same-topic half-sample control.** The same sixteen fits, source from R1 and target from R2, then swapped. 32 fits. This matches part 1 in sample size and differs only in whether the halves share topics.
3. **Within-cell topic-split reference.** In each cell, set A to set B and set B to set A. 12 fits.

## Metrics

Per fit: top-1 rate, median rank, and median cosine between each mapped vector and its own target vector. Top-1 and median cosine are foregrounded. Median rank is kept for continuity with the battery's criterion.

## Criterion and reading

- Criterion, inherited from §4.3 for continuity: median rank at most 2 across all 171 predictions. No permutation test.
- If every part-1 fit meets the criterion, the result is stated as: all tested cross-cell fits meet the retrieval criterion under this fixed disjoint-topic partition. It is not stated as robustness across partitions.
- If any part-1 fit misses, the result is stated as: the criterion was not met in the listed fits under this partition. Every miss is listed by pair and assignment. It is not stated as a general failure of transport.
- Parts 2 and 3 are descriptive references. Part 2 separates topic disjointness from reduced sample size at matched size. Part 3 shows within-cell sensitivity to topic change and reduced sample size. Comparable cross-cell performance is reported as consistent with limited additional degradation across checkpoints, and the comparison is not read as identifying the cause of any degradation.
- No numeric thresholds beyond the inherited criterion. All other comparisons are reported descriptively.

## Bag-of-words baseline

- **Span.** The same span the models pool: tokens from the 50th onward under the Gemma 2 tokenizer (shared across the three sizes). The span is recovered by decoding each story's stored token IDs from the 50th onward with that tokenizer. Stories shorter than 50 tokens follow the main pipeline's short-story fallback, stated in the output.
- **Features.** Lowercased word counts on the decoded span, tokenized on runs of letters (regex `[a-z]+`), vocabulary from all 205,200 spans, words appearing in fewer than 5 spans dropped. Each span's count vector is L2-normalized. A span whose vector is empty after filtering is excluded and counted.
- **Variants, both reported.** (i) All words. (ii) Exact-label removal: every one of the 171 emotion names that is a single `[a-z]+` token is removed from the vocabulary. Hyphenated and multiword names are not split into components for removal, and morphological variants, synonyms, and related words are kept. The exact exclusion list is written to the output and released.
- **Half-vectors.** For each emotion, the set-A and set-B vectors are means over that emotion's set-A and set-B spans, centered on their half's grand mean. A centered vector with zero norm is reported and excluded from scoring.
- **Retrieval, two forms, both reported.** (a) Direct: rank the 171 set-B vectors by cosine to each set-A vector, and the reverse, with no map. (b) Fitted: the battery's map, five-fold assignment, and scoring, applied to the bag-of-words half-vectors, so the lexical comparator has the same fitting step the models have. Same metrics.
- **Reading.** High lexical-baseline retrieval would show that lexical information alone is sufficient for strong performance on this task. Lower lexical retrieval than the model vectors is reported descriptively and is not read as showing that the models carry information beyond lexical structure. Neither outcome re-grades anything.

## Excluded

- The embedding-table version of the disjoint-topic test. Per-story embedding-table means are not stored.
- Any permutation null and any second partition.

## Outputs

One file per part: cross-cell disjoint (part 1), matched half-sample (part 2), within-cell reference (part 3), bag-of-words (both variants, both retrieval forms). Each with per-fit metrics, the partition, the exclusion list where relevant, counts of excluded spans or vectors, and input hashes.
