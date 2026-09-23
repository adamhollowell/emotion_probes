# Emotion Vectors in a Base Model — Replication on Gemma-2-27B

Code for a protocol-matched replication of the representational core of Anthropic's
emotion-vector study (Sofroniew et al. 2026) on the **base pretrained** model
`google/gemma-2-27b`. Every disclosed parameter is inherited from the original; only
the subject model changes.

This repo: `https://github.com/adamhollowell/emotion_probes` · companion model repo:
`https://huggingface.co/adamhollowell/gemma-emotion-vectors`.

Paper: *Replicating the Geometry of Emotion Representations in a Base Open-Weights Model*, [arXiv:2609.22208](https://arxiv.org/abs/2609.22208).

---

## The two released repositories

The release is split by artifact type:

| Repo | Holds | Link |
|------|-------|------|
| **Code** (this repo) | the extraction/analysis pipeline and every figure producer; the **500 neutral dialogues**; the **300-story validation sample** with hand-codings; small analysis outputs | `https://github.com/adamhollowell/emotion_probes` |
| **Model** (Hugging Face) | the extracted emotion vectors (`emotion_vectors_171.pt`) + embedding-null vectors + logit-lens export recipe | `https://huggingface.co/adamhollowell/gemma-emotion-vectors` |

The **205,200-story stimulus corpus is not redistributed as a file** — it is a large volume of
Claude-generated text, released instead as a **generation recipe** (the verbatim elicitation
prompt and topic/emotion design in the paper's Appendix A, plus the generation scripts here),
from which an equivalent corpus regenerates. The Claude-generated data that *does* ship — the
neutral dialogues and the validation sample — carries a downstream-use notice: see
[`CLAUDE_DATA_NOTICE.md`](./CLAUDE_DATA_NOTICE.md).

### Getting the data this repo expects

The scripts read from `data/`:

- `data/vectors/emotion_vectors_171.pt` — download from the **model** repo `https://huggingface.co/adamhollowell/gemma-emotion-vectors`.
- `data/stories/neutral_dialogues.jsonl` — **ships in this repo** (already under `data/stories/`).
- `data/stories/stories_171_canonical_clean.jsonl` — the 205,200-story corpus is **not shipped**;
  regenerate it with the generation scripts (below) from the Appendix A prompt. Analyses that read
  the vector artifact do not need the raw stories.

---

## What ships as a recipe, not a file

Several inputs are **not redistributed** and are regenerated locally, for third-party-terms
or licensing reasons. This is deliberate; each has a builder or export script here.

- **Held-out evaluation corpus** (§3.6) — assembled from four third-party datasets
  (Common Corpus, `monology/pile-uncopyrighted`, LMSYS-Chat-1M, Isotonic), each under its own
  terms. We ship the seeded builder (`scripts/build_heldout_corpus_v2.py`), both construction
  manifests (`data/heldout/corpus_manifest_v2.json`, `token_manifest_v2.json`), the corpus
  SHA-256 (`RUN_MANIFEST.json`), and upstream pointers. Rebuild under those sources' terms and
  verify by hash. (One caveat: upstream revisions weren't pinned at build time, so a byte-exact
  rebuild depends on the sources being unchanged; the published hash is the check either way.)
- **Per-token projection artifact** (`pertoken.tar`, behind Figure 1 / §8) — the full tokenized
  text of the winning third-party documents. Not redistributed for the same reason; the render
  pipeline (`scripts/maxact_sweep_v2.py` + the Figure-1 builders) regenerates it directly from a
  rebuilt-and-hash-verified corpus. The **text-stripped** version (`pertoken_stripped.tar` — token
  ids, per-token projections, norms, doc/source labels, offsets; every readable-text field removed)
  is what the figures actually consume, and it *does* ship — but it is **549 MB, too large for
  GitHub, so it is hosted on Hugging Face** alongside the vectors:
  `https://huggingface.co/adamhollowell/gemma-emotion-vectors`. Fetch it into `data/heldout/maxact/`
  before running the Figure-1 / §8 analyses, then verify against the shipped
  `pertoken_stripped.tar.sha256` (`a30ce789…`, also in `RUN_MANIFEST.json`):

  ```python
  from huggingface_hub import hf_hub_download
  hf_hub_download("adamhollowell/gemma-emotion-vectors", "pertoken_stripped.tar",
                  local_dir="data/heldout/maxact")
  ```

  `data/heldout/maxact/maxactivating_v2.json` (short windows with third-party text) is not shipped;
  `pass1_results.json` ships the non-text metadata. The **P6.1 structural floor** (52.6% = 90/171)
  is classified from the windows' peak tokens, so we ship a derived
  [`geometry_out/peak_tokens.json`](geometry_out/peak_tokens.json) — per emotion, the top-20 windows'
  isolated **peak token + structural category** only, no window spans (built by
  `scripts/build_peak_tokens.py`). `scripts/analyze_maxact_peaks.py` reads it when the raw windows are
  absent, so the 52.6% rank-1 floor and 48.4% top-20 rate reproduce from shipped files.
- **Logit-lens ingredients** (`lens_ingredients.pt`) — essentially Gemma-2-27B's own unembedding
  matrix. Not redistributed (Gemma weights); regenerate with `scripts/export_lens_ingredients.py`
  from the base model, then run the lens scripts. *(Ships in the model repo alongside the vectors.)*
- **Third-party human-norm lexicons** (NRC-VAD, Warriner) — cite-don't-bundle (non-commercial);
  provenance + fetch scripts ship, not the lexicons. The **one exception shipped directly** is the
  45-emotion Russell–Mehrabian (1977) overlap table, `data/ratings/russell_mehrabian_1977.csv`.

---

## Reproducing the paper

Analyses run in paper order. Geometry and activation analyses run **from the released artifacts on
CPU**; extraction and any lens re-derivation need the base model.

### §3 — Methods: corpus, extraction, held-out corpus

- **Story corpus** (`data/stories/…`): generated by
  `run_batch_171_extension.py` → `cleanup_cells.py` → `run_batch_12_regenerate.py` →
  `cleanup_cell_12.py` → `assemble_171.py` → `strip_perspective_labels.py`; neutral dialogues via
  `run_neutral_generation.py`. (Uses the Anthropic API; set `ANTHROPIC_API_KEY` in `.env`.)
- **Corpus validation** (`emotion_validation/`): `draw_validation_set.py` (frozen sample, seed
  20260622), `check_emotion_words.py`; codings + summaries ship as the record.
- **Extraction** → `emotion_vectors_171.pt`: `python scripts/run_extraction_pipeline.py` (drives the
  `emotion_probes` package). GPU run — see `scripts/runpod_bootstrap.sh`. Regime: `bfloat16`,
  TransformerLens `TransformerBridge`, `tl_compatibility_mode=False` (raw weights), L31 `resid_post`.
- **Held-out corpus**: `build_heldout_corpus_v2.py` (recipe, above).

### Producer map (figures & tables → script)

Reproduce a figure/table by running its producer from the repo root (they read `data/vectors/…`
and write to the repo root or `geometry_out/`). Figures reference the Anthropic-parallel numbering.
See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full artifact→producer map, including
which appendix JSONs ship as certified outputs vs. regenerate from a shipped script.

| Result | Producer |
|--------|----------|
| Cosine-similarity matrix (cf. Fig 5) | `scripts/render_parallel_fig5_cosine_matrix.py` |
| UMAP + k-means clusters (cf. Fig 6) | `scripts/render_parallel_fig6_umap.py` → `render_parallel_fig6_interactive.py` (hand-tuned labels) |
| Cluster membership (Table 12 — ships as a LaTeX text table, no PNG) | `geometry_out/L31_kmeans_partition.csv` (built by `scripts/s4_clusters.py`) |
| Cluster robustness / cohesion | `scripts/analyze_cluster_robustness.py` |
| Per-emotion cohesion table (Appendix B) | `scripts/peremotion_purity.py` → `geometry_out/peremotion_purity.csv` (top-10 cosine-neighbor in-cluster purity + core stability; self-validates mean 0.756 / 46 mutual-NN pairs, exits nonzero on mismatch) |
| PC projection bars (cf. Fig 7) | `scripts/render_parallel_fig7_pc_projections.py` |
| PCA vs Russell–Mehrabian (cf. Fig 8) | `scripts/render_parallel_fig8_russell_mehrabian.py` (self-audits r=0.72/0.67) |
| Labeled circumplex (cf. Fig 57) | `scripts/render_parallel_fig57_circumplex.py` |
| Vector norms vs valence | `scripts/vector_norms.py` |
| RSA co-activation (r=0.907) | `scripts/analyze_rsa_coactivation.py` |
| Sun et al. reconciliation | `scripts/analyze_sun_choi_engagement.py` |
| PC1 valence-residual test | `scripts/analyze_pc1_valence_residual.py` — self-validates against the locked values (exits nonzero on mismatch); inputs `data/vectors/emotion_vectors_171.pt` + `data/ratings/nrc_vad_v2_1.csv` (NRC-VAD, gitignored — drop in per its provenance) |
| Cross-layer RSA heatmap (cf. Fig 9) | `scripts/render_parallel_fig9_cross_layer_rsa.py` |
| Layerwise geometry / affect alignment | `scripts/analyze_layerwise_internal_geometry.py`, `analyze_layerwise_external_affect_alignment.py`, `band_summary_table.py`, `analyze_depth_arousal.py`, `analyze_velocity.py` |
| Embedding-layer baseline / null (§6.6) | `scripts/extract_embedding_null.py` → `embedding_null_vectors.pt` |
| Direct-unembedding table (cf. Table 1) | `scripts/render_parallel_table1_unembedding.py` |
| Logit-lens RMSNorm methods point | `scripts/lens_readout_decision.py`, `lens_research_emotions.py`, `lens_recompute_test.py` |
| Lexicalization onset at the seam | `scripts/seam_sweep.py` |
| Max-activation panels (Figure 1) | `scripts/render_parallel_fig1_maxact.py` |
| Max-activation confound floor / sources | `scripts/maxact_sweep_v2.py`, `analyze_maxact_peaks.py`, `analyze_concept_word_corroboration.py`, `analyze_source_heterogeneity.py`, `analyze_thresholds.py`, `analyze_magnitude_diag.py` |

Precomputed analysis outputs are in `geometry_out/`, `activation_out/`, and
`data/heldout/maxact/`, so most figures reproduce without re-running the sweep. The study's own
numbers remain reproducible from the producers above plus `emotion_vectors_171.pt`.

### The logit-lens step (run the export first)

The lens scripts read `lens_ingredients.pt`, which is not shipped. Regenerate it once:

```bash
python scripts/export_lens_ingredients.py --out data/vectors/lens_ingredients.pt
```

It reads only Gemma-2-27B's embedding + final-norm tensors from the base model's safetensors
(~9 GB, CPU-friendly; you need Hugging Face Gemma access). Then run the scripts that consume it — `render_parallel_table1_unembedding.py` (Table 1) and the `lens_*` scripts (`lens_readout_decision.py`, `lens_research_emotions.py`, `lens_recompute_test.py`).

---

## Repository layout

```
emotion_probes/            the extraction/analysis library (imported by the scripts)
scripts/                   producers: extraction, corpus build, and every figure/table
emotion_validation/        corpus hand-coding: draw script, check, 300-story validation sample, codings (ships)
data/
  vectors/                 emotion_vectors_171.pt (from the model repo) + regenerated lens_ingredients.pt
  stories/                 neutral_dialogues.jsonl (ships); the 205,200-story corpus is recipe-only (regenerate)
  ratings/                 russell_mehrabian_1977.csv (shipped); NRC-VAD/Warriner fetched
  heldout/                 held-out-corpus manifests + hashes; maxact/ sweep outputs (no corpus)
geometry_out/              analysis outputs behind the geometry/clustering/layer figures
activation_out/            residual-norm stats
RUN_MANIFEST.json          run provenance + corpus/vector SHA-256s
```

---

## Two authored scripts (not part of the original run)

Two producers were reconstructed and are flagged as such:

- **`scripts/export_lens_ingredients.py`** — a lightweight re-derivation of `lens_ingredients.pt`
  from the base model (the original exporter ran on an ephemeral GPU instance). Verified to
  reproduce the original `W_U`, final-norm weight, and vocab **bit-for-bit**.
- **`scripts/vector_norms.py`** — a reconstruction of the (lost) vector-norm producer. Verified to
  reproduce the shipped `vector_norms_*` outputs (L31 mean-norm 491.65, norm↔PC1 r=+0.406, etc.).

**Note on `emotion_probes/validation.py`:** its `logit_lens()` is an internal extraction-time
sanity check written under an earlier compatibility-mode assumption and is **not** the paper's
logit-lens. The paper's lens analysis (§7) is the `scripts/lens_*` producers, which apply Gemma-2's
`(1 + w)` RMSNorm explicitly.

---

## Environment

Python 3.10–3.12. Core: `torch`, `transformer_lens==3.3.0`, `transformers`, `numpy`, `pandas`,
`scipy`, `scikit-learn`, `matplotlib`, `pillow`, `adjustText`, `umap-learn`, `safetensors`, `huggingface_hub`
(and `anthropic` for corpus generation). Extraction ran on a single **80 GB H100** in `bfloat16`;
the geometry and activation analyses run from the released artifacts without a GPU. The logit-lens
export loads only two tensors from the base model and runs on CPU.

## Citation & license

Please cite the paper ([arXiv:2609.22208](https://arxiv.org/abs/2609.22208)):

```bibtex
@misc{hollowell2026replicating,
  title         = {Replicating the Geometry of Emotion Representations in a Base Open-Weights Model},
  author        = {Hollowell, Adam},
  year          = {2026},
  eprint        = {2609.22208},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.22208}
}
```
 Code license: **MIT** (see `LICENSE`). Third-party data are
released under their own terms as described above; Gemma-2-27B is subject to Google's Gemma Terms
of Use.
