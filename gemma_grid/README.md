# The Geometry of Emotion Representations Across Scale, Depth, and Instruction Tuning

This directory holds the code, results, audit record, and manifests for a six-model study of emotion geometry in the Gemma 2 family: three sizes (2B, 9B, 27B), each in its base and instruction-tuned form. The study applies the emotion-vector pipeline of Sofroniew et al. (2026) to every model and asks which parts of the geometry hold across size and tuning.

It sits inside the repository of the replication study it builds on (Hollowell 2026, arXiv:2609.22208), whose pipeline it inherits unchanged. Base 27B is the anchor model, the one the replication study measured, so the grid's anchor vectors are the replication's released vectors.

Paper: The Geometry of Emotion Representations Across Scale, Depth, and Instruction Tuning, arXiv:[ID ON POSTING].

## Where the files are

| Location | Holds |
|---|---|
| This directory (`gemma_grid/`) | code, every analysis output that is not a large binary, the audit record, configs, provenance, and the hash manifest |
| Hugging Face `adamhollowell/gemma-emotion-vectors`, folder `gemma_grid/` | per-model emotion vectors, neutral-dialogue activations, cleanup bases, bootstrap draws, per-story slices, embedding-table vectors, and the probe-mean and colon-token activations |
| The parent repository | the neutral-dialogue corpus, the story-generation scripts, and the scenario activations (the anchor model's vectors are at the root of the Hugging Face repository) |

Nothing in the companion Hugging Face repository is removed or replaced by this study. The grid's files sit in their own folder beside the replication study's.

## Identity is by hash

Every input below is named by its sha256. A file name alone is not enough to recover the paper's numbers. The Hugging Face folder `gemma_grid/` mirrors this directory, so each binary sits at the same path in both places.

The study's files were produced under working names that carry the suffix `_100T`, for example `scripts/table5_depth_thirds_100T.py` writing `geometry_out_100T/table5_depth_thirds_100T.json`. Every released file drops that suffix, so the same script ships as `scripts/table5_depth_thirds.py`. Ten pipeline step scripts also took new names, `scripts/run_<step>.py`. `provenance/SCRIPT_EDITS.md` maps every released script to the name it ran under, with both hashes. Manifests, result files and scripts written before the release still name the working paths inside them. Identity is carried by the hashes, which do not change with the name. A few released copies differ from the working files in one respect: absolute local paths, in environment fields and in three scripts' scratch-folder defaults, were made relative. `RELEASE_MANIFEST.md` lists each such file with its released hash, and no number changed.

| Object | sha256 | Where |
|---|---|---|
| Neutral-dialogue corpus (500 dialogues) | `3838a8f23e726d2433135336477ab0ee0e21f4ecb8ca06613963d26de8a3a04f` | parent repo, `data/stories/neutral_dialogues.jsonl` |
| Anchor vectors, base 27B | `6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b` | Hugging Face, `emotion_vectors_171.pt` (the replication study's file) |
| Vectors, base 2B | `4622ff9c90fa56c7890f0af45931fd5454f514b7e374f82da5bcbdfa130a0300` | Hugging Face, `gemma_grid/data/gemma-2-2b-base/vectors/canonical.pt` |
| Vectors, tuned 2B | `e404a92a73607d02565bc9d405bb8d3d7cf242e25e40b493baf55d5938b2882f` | Hugging Face, `gemma_grid/data/gemma-2-2b-it/vectors/canonical.pt` |
| Vectors, base 9B | `74b8f3f50dbfbe4b93683a7007482e7eb41d9d8c84b6ccc9ff13827d8fedbb5c` | Hugging Face, `gemma_grid/data/gemma-2-9b-base/vectors/canonical.pt` |
| Vectors, tuned 9B | `f99508c5ed0f71325cce804f971ceda1ae0cd707d8bf6e45174cecc15b89767a` | Hugging Face, `gemma_grid/data/gemma-2-9b-it/vectors/canonical.pt` |
| Vectors, tuned 27B | `10d1234df22ef4e335b0cefd8e264dee5d75f9a1f2e94f285b18199f21d4b2b5` | Hugging Face, `gemma_grid/data/gemma-2-27b-it/vectors/canonical.pt` |
| Neutral activations, base 27B | `04e1a814c586f4aa30ef96edf075956dddab8bec19b6f46f998d36f1c30a644b` | Hugging Face, `gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__neutral.pt` |
| Neutral activations, base 2B | `13b6c6b98c67308f57dda598d2d4723e3708d9cc189ca9e371492faa6ec6c965` | Hugging Face, `gemma_grid/data/gemma-2-2b-base/vectors/gemma-2-2b-base__neutral.pt` |
| Neutral activations, tuned 2B | `82282c77ec8b94f0ecfa9bf08b4c69c46ed85a17d72920f03d6874c407f67bac` | Hugging Face, `gemma_grid/data/gemma-2-2b-it/vectors/gemma-2-2b-it__neutral.pt` |
| Neutral activations, base 9B | `723360d2febc35707deed7f6079023810c1e8c114917065d64f21affcee90340` | Hugging Face, `gemma_grid/data/gemma-2-9b-base/vectors/gemma-2-9b-base__neutral.pt` |
| Neutral activations, tuned 9B | `fd557d875961e6f6f12662a9319c99eb3b8233b077dc46362723990339c9ef74` | Hugging Face, `gemma_grid/data/gemma-2-9b-it/vectors/gemma-2-9b-it__neutral.pt` |
| Neutral activations, tuned 27B | `0b6c5abde2e395fd2cecbe2522566f57759ec6bf635678681295e9465fa7d5bd` | Hugging Face, `gemma_grid/data/gemma-2-27b-it/vectors/gemma-2-27b-it__neutral.pt` |
| Per-story slices, base 2B, without neutral rows | `ca3d11270ad3a4838a18cbb2e36f5d152eea746ad456c3c71feb45769324ebb6` | Hugging Face, `gemma_grid/data/gemma-2-2b-base/vectors/gemma-2-2b-base__per_story_slice.pt` |
| Per-story slices, tuned 2B, without neutral rows | `0b577fbda09fe811d48a852d32837ef2f01ead1c594b8832f6aba828522d9ff5` | Hugging Face, `gemma_grid/data/gemma-2-2b-it/vectors/gemma-2-2b-it__per_story_slice.pt` |
| Per-story slices, base 9B, without neutral rows | `6010cf0683540e1d872cadcdcbee4f2b7f1c48139ff32ae73b53f97d8c31a812` | Hugging Face, `gemma_grid/data/gemma-2-9b-base/vectors/gemma-2-9b-base__per_story_slice.pt` |
| Per-story slices, tuned 9B, without neutral rows | `3205a3a2197643db74765b9c9d2b6f738beaf15b6f76e3c744fa060a08b54842` | Hugging Face, `gemma_grid/data/gemma-2-9b-it/vectors/gemma-2-9b-it__per_story_slice.pt` |
| Per-story slices, base 27B, without neutral rows | `bfcaa65cb967e40c81a4d858a0deee08805115d2bd56ed5191cc2bb6ed52c73d` | Hugging Face, `gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__per_story_slice.pt` |
| Per-story slices, tuned 27B, without neutral rows | `a2e57cfd35cacd1be7bd217de8df3fcc6a667dd651085ecc6ddc18882d6249e7` | Hugging Face, `gemma_grid/data/gemma-2-27b-it/vectors/gemma-2-27b-it__per_story_slice.pt` |
| Cleanup basis, base 2B | `68d11f44f22c88d7e4548a87b18cdfffb30ba77a20cc14db1c0c53d0a914199d` | Hugging Face, `gemma_grid/data/gemma-2-2b-base/vectors/gemma-2-2b-base__cleanup_basis.pt` |
| Cleanup basis, tuned 2B | `aba94b203aa0361cd6cd87108e1791562d8410a6786f2986ab49d237183fff8c` | Hugging Face, `gemma_grid/data/gemma-2-2b-it/vectors/gemma-2-2b-it__cleanup_basis.pt` |
| Cleanup basis, base 9B | `1c525816edb0981635c9d154be10f49679f7b01a175387a299bfbd2774e2459b` | Hugging Face, `gemma_grid/data/gemma-2-9b-base/vectors/gemma-2-9b-base__cleanup_basis.pt` |
| Cleanup basis, tuned 9B | `dc663f75042a84d32b3e031a9143834a98ace31d03c01a270519b49481088249` | Hugging Face, `gemma_grid/data/gemma-2-9b-it/vectors/gemma-2-9b-it__cleanup_basis.pt` |
| Cleanup basis, base 27B | `e461d13e5c5429887c685b05f963ff717bb0e93b9e982863eb6c902dd2fec736` | Hugging Face, `gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__cleanup_basis.pt` |
| Cleanup basis, tuned 27B | `6058f8f479992bc0b2c50926e3b6e1a5cb3cf657a35f21646fb136901f996715` | Hugging Face, `gemma_grid/data/gemma-2-27b-it/vectors/gemma-2-27b-it__cleanup_basis.pt` |
| Bootstrap draws, base 2B | `3dff080466cf0f9d0460355eb0ea327e20234eb26cb6f0e7c067ba854387c046` | Hugging Face, `gemma_grid/geometry_out/gemma-2-2b-base/bootstrap_B1000.npz` |
| Bootstrap draws, tuned 2B | `f3143bc810712f09ccc3fdd86927b6cc9d9745624cc543a62760b7e513ee81eb` | Hugging Face, `gemma_grid/geometry_out/gemma-2-2b-it/bootstrap_B1000.npz` |
| Bootstrap draws, base 9B | `0ac4d74d4644cb16ce8f0b065fb744cade6af58bfc0d5f1205ee0123c670fbad` | Hugging Face, `gemma_grid/geometry_out/gemma-2-9b-base/bootstrap_B1000.npz` |
| Bootstrap draws, tuned 9B | `5c8b1a4129cba88f4e457f55b791adbe7daeced006d79ccc78324b1b918548d5` | Hugging Face, `gemma_grid/geometry_out/gemma-2-9b-it/bootstrap_B1000.npz` |
| Bootstrap draws, base 27B | `8dda7cc7bf7ef5e12d207673632f191728a0e6c81828215705fa0616cc200e78` | Hugging Face, `gemma_grid/geometry_out/gemma-2-27b-base/bootstrap_B1000.npz` |
| Bootstrap draws, tuned 27B | `d9db8ede260f878874ea6aa3847ba502a509b2428397dcee69d016a104a146b5` | Hugging Face, `gemma_grid/geometry_out/gemma-2-27b-it/bootstrap_B1000.npz` |
| Embedding-table vectors, base 2B | `6d23da00388a5514088c0bff5fe9250dcc33b09b7ec8619d7f33d4d77e7460c6` | Hugging Face, `gemma_grid/geometry_out/gemma-2-2b-base/c3_vectors.pt` |
| Embedding-table vectors, tuned 2B | `21edd586fb0aa7fe1775d954fb9e615b445854f4b6c1369198dfb858c2af880d` | Hugging Face, `gemma_grid/geometry_out/gemma-2-2b-it/c3_vectors.pt` |
| Embedding-table vectors, base 9B | `1314eb023c104ea2b9aa3972427ae30f088d68601982f7112ecb11cccc26e407` | Hugging Face, `gemma_grid/geometry_out/gemma-2-9b-base/c3_vectors.pt` |
| Embedding-table vectors, tuned 9B | `270ac1001bdc2590236bf01f9126596e8814f220ead04163e81f4a980d86ef3c` | Hugging Face, `gemma_grid/geometry_out/gemma-2-9b-it/c3_vectors.pt` |
| Embedding-table vectors, base 27B | `5b701c50f765bd798259013716d55ea8d3cff17cc13dd91ada09b2dbeaa7e4ca` | Hugging Face, `gemma_grid/geometry_out/gemma-2-27b-base/c3_vectors.pt` |
| Embedding-table vectors, tuned 27B | `40ca301223e376bfe78bd6eac6973945c182f9908264970b9ec1507eee5329f2` | Hugging Face, `gemma_grid/geometry_out/gemma-2-27b-it/c3_vectors.pt` |
| Colon-token activations, tuned 27B | `769a917f5b652e950afb2d4d274eff1ad4e697c46b93e4434cccecd1792cb785` | Hugging Face, `gemma_grid/colon_it/colon_analysis_it.pt` |
| Probe means across the layer band, base 27B, seed 0 | `6240994dff6916210917a8e31b3678ef272e80ea0498b077151879903c0d9108` | Hugging Face, `gemma_grid/probe_band/probe_band_means_seed0.pt` |
| Probe means across the layer band, base 27B, seed 1 | `3fc5dd053278262b0c57614f358c9b23db5794e4b343f4f0a4afa487cad625ec` | Hugging Face, `gemma_grid/probe_band/probe_band_means_seed1.pt` |
| Probe means at layers 31, 42 and 45, base 27B, seed 0 | `30177fe30e4a39195d3bffaf51e263034e17fed3ed80c769920824faf3c667fa` | Hugging Face, `gemma_grid/probe_l45/probe_l45_means.pt` |
| Probe means at layers 31, 42 and 45, base 27B, seed 1 | `4bde4a233aeb09d2a9d1cdb8d6e7040b0b853bc1683f9a263759382c4e7845b0` | Hugging Face, `gemma_grid/probe_l45/probe_l45_means_seed1.pt` |
| Model revision register | in `handoff/hf_model_revisions_VERIFIED.json` | this directory |

The anchor vectors file carries an internal `variant` label, and the per-model vector files carry an internal `provenance` string, both written at extraction. The labels are stale. The same holds for notes stored inside other binaries: the neutral-activation files name their source corpus by its working file name, beside the corpus's sha256, and the embedding-table vectors and the layer-band probe means carry notes written during the analysis. The hashes above identify the files of record, and every analysis gates on them. The cleaned vectors at the layers the analyses read were checked against the cleanup basis built on the neutral corpus of record (audit 7).

Every shipped file, in this directory and in the Hugging Face folder, is listed in `SHA256SUMS.txt`, the binaries under the paths the download puts them at. After downloading the Hugging Face folder, from the repository root:

```
shasum -a 256 -c gemma_grid/SHA256SUMS.txt
```

`SHA256SUMS.txt` is built from the pipeline's working hash manifest, limited to files in this release and using this release's paths, with the Hugging Face binaries added from the upload plan. Its sidecar `SHA256SUMS.txt.provenance.json` gives the sha256 of the working manifest it was derived from, and lists which hashes were recorded during the analysis and which were taken when the release was assembled.

## The six models

All six are pinned to the revisions in `handoff/hf_model_revisions_VERIFIED.json`, verified at the level of the weight-shard hashes on each machine that ran an extraction. Verification ran on 2026-07-31, after the first extraction of 2026-07-23, against the same pinned revisions.

| Model | Revision |
|---|---|
| `google/gemma-2-2b` | `c5ebcd40d208330abc697524c919956e692655cf` |
| `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` |
| `google/gemma-2-9b` | `33c193028431c2fde6c6e51f29e6f17b60cbfac6` |
| `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| `google/gemma-2-27b` | `938270f5272feb02779b55c2bb2fffdd0f53ff0c` |
| `google/gemma-2-27b-it` | `aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0` |

## Pipeline invariants

Every model passes through the same pipeline, and four rules hold for all six.

- The analysis layer is derived from each model's depth as round(2/3 × number of layers): layer 17 at 2B, 28 at 9B, and 31 at 27B. Layer 18 at 2B is the preregistered sensitivity check.
- Each model's cleanup basis comes from that model's own neutral-dialogue activations, keeping the top principal components to 50 percent of variance. No basis is shared between models.
- Every model is loaded in the raw-weights regime of the anchor model (TransformerLens with `tl_compatibility_mode=False`, bfloat16, the residual stream after each block), so the vocabulary readout applies Gemma 2's (1 + w) final norm explicitly.
- Base and instruction-tuned models receive identical raw story text, with no chat template.

The neutral-dialogue corpus is identified by its hash, `3838a8f23e726d2433135336477ab0ee0e21f4ecb8ca06613963d26de8a3a04f`, not by its file name.

## What ships as a recipe

- The 205,200-story stimulus corpus is Claude-generated and is released as a generation recipe, following the replication study. The 171-emotion vocabulary and the generation prompts are in the replication study's appendix. `data/story_topics.json` gives the emotion and topic of every story in corpus order, with no story text, and `handoff/story_indices_all_cells.json` maps each per-story slice row to its story. Together they let the disjoint-topic fits rerun without the corpus. The bag-of-words baseline of that analysis reads the story text, so it does not reproduce from this release.
- The vocabulary readout of §6 needs the unembedding matrix and final-norm weight of each of the six models. These are read straight from the model weights, so under the Gemma Terms of Use they ship as a regeneration recipe run under your own Gemma access grant, not as files. `scripts/export_lens_ingredients_grid.py` regenerates them from the six pinned checkpoints and was checked against the stored ingredients: every tensor matches exactly (`geometry_out/lens_export_verification.json`). The vocabulary map is the same for all six checkpoints, whose tokenizers are byte-identical at the pinned revisions, and matches the base 27B model's stored map in all 256,000 entries (`geometry_out/anchor_vocab_tokenizer_check.json`, `geometry_out/vocab_all_six_check.json`).
- NRC-VAD v2.1 and the Warriner et al. (2013) norms may not be redistributed. Each ships as provenance plus a fetch script. The Russell and Mehrabian (1977) ratings for 45 emotions ship as a small derived table with citation.

## Reproducing the paper

Everything downstream of the extracted vectors recomputes on a CPU from the released files, with one exception: the disjoint-topic bag-of-words baseline needs the story text (see above). Extraction and the readout ingredients need the models.

1. Download the Hugging Face folder `gemma_grid/` over this directory, which puts every binary at the path the scripts read. Put the replication study's `emotion_vectors_171.pt` and `embedding_null_vectors.pt`, from the root of the same Hugging Face repository, in `data/vectors/`.
2. Run the pipeline. Its preflight checks the corpus at the public path against `3838a8f2…` and stops if the file is missing or its hash differs.

```
bash gemma_grid/scripts/run_grid_phase4.sh
```

3. Run the standalone producers listed in the map below.
4. Check every output against the manifest with the `shasum` command above.

The software environment is pinned in the release manifest (`RELEASE_MANIFEST.md`), separately for extraction and for analysis. The pipeline stops if NumPy or scikit-learn differs from its pin. In the pinned environment every computed value reproduces exactly, with two exceptions. Some values from `scripts/analysisC_g32_depaware.py`, `scripts/verify_tuning_fisher_pooled.py` and `scripts/verify_interaction_did_constructions.py` differ at about 10⁻¹⁶. Step 5's `warm_cluster_mean` values in `stageB_contrasts.json` vary at about 10⁻⁸ relative with `PYTHONHASHSEED`, because the script iterates over a set of emotion names, and they are neither graded nor printed. Files that record a run time or a script path differ in those fields only. On other machines they agree to the precision the paper's §8.4 reports for this pipeline, with residue in the last one or two digits from the order of floating-point summation.

## Producer map

### The pipeline, `scripts/run_grid_phase4.sh`

Each step calls a `scripts/run_<step>.py` entry point, which drives the algorithms in the `scripts/phase4_*.py` modules.

| Output in `geometry_out/` | Step |
|---|---|
| `<model>/stageA_summary.json`, `stageA_cross_cell_table.csv` | 3, Stage A geometry per model and the cross-model table (Table 1) |
| `stageB_contrasts.json`, `stageB_displacement__*.csv` | 5, Stage B contrasts and per-emotion displacement |
| `<model>/bootstrap_B1000.npz` (Hugging Face) | 6, bootstrap draws per model |
| `stageC_grading.json`, `stageC_grading.md` | 7, grading of the registered contrasts |
| `c3_summary.json`, `<model>/c3_embedding.json`, `<model>/c3_vectors.pt` (Hugging Face) | 8, the static token-embedding table |
| `c3_grading.json` | 9 |
| `<model>/lens_summary.json`, `lens_cross_cell.json` | 10, the vocabulary readout (§6) |
| `extracts_summary.json`, `consolidation_curves.json`, `fig84_depth_tuning.json`, `family_membership/family_membership_<model>.md` | 11, depth profiles and the per-model family membership tables |
| `layerstab_ci.json` | 12 |
| `seam_gap_metric_v1.json` | 13, the consolidation seams (§6.3) |
| `op1_spectrum_measures_v1.json` | 14, variance spectrum and intrinsic dimension (§5.2) |
| `norm_robustness_full_results.json` | 15, Table 6 (§8.1) and Appendix E |
| `fixed_probe_tuning_shifts_results.json` | 16, activation profiles (§7.2) and the tuning-shift layer structure (§7.3) |
| `zhang_parallels/` | 17 to 19 |

Steps 1 and 2 write into `data/`: the token ids of the neutral corpus (`data/c3_tokens_neutral.npz`, in this directory) and each model's `canonical.pt` and cleanup basis (Hugging Face). Step 2 ends with the anchor gate, which stops the run unless the base 27B re-clean matches the replication study's vectors at layer 31 to a relative difference of 1e-6. Step 1 reads the Gemma 2 tokenizer, so it needs a Hugging Face login with the Gemma license accepted.

### Standalone producers

| Output | Producer | Paper |
|---|---|---|
| `pmap_transport.json`, `pmap_transport_floor.json`, `pmap_battery.json` | `scripts/pmap_transport_driver.py` over `scripts/pmap_transport.py`, `scripts/pmap_floor_pathA_node.py`, `scripts/pmap_battery_serial.py` | §4.3, Figure 2, Appendix D |
| `item5_svd_provenance.json` | `scripts/item5_svd_provenance.py` | Appendix D, solver events |
| `analysisA_displacement_bootstrap.json`, `analysisA_ranks__*.csv` | `scripts/analysisA_displacement_bootstrap.py` | §7.4 |
| `analysisB_seam_onset_bootstrap.json`, `analysisB_drawlevel.csv` | `scripts/analysisB_onset_parallel_v2fullscope.py` over `scripts/analysisB_seam_onset_bootstrap_gesvdfix.py` | §6.3 |
| `analysisC_g32_depaware.json` | `scripts/analysisC_g32_depaware.py` | §5.3 |
| `job1_L18_emotion_bootstrap_sensitivity.json` | `scripts/job1_L18_emotion_bootstrap_sensitivity.py` | §5.3, layer 18 |
| `crosswalk_s53_L18_registered_p.json` | `scripts/crosswalk_s53_L18_registered_p.py` (reads the layer 18 file) | §5.3 |
| `c3_raw_cleaned_embedding.json` | `scripts/c3_raw_cleaned_embedding.py` | §5.1, Table E.1 |
| `p32_resid_norm_growth_family.json` | `scripts/phase4_p32_resid_norm.py` | §8.3 |
| `p34b_arousal_pc23_family.json` | `scripts/phase4_p34b_arousal_pc23.py` | §5.3 |
| `p37_velocity_family.json` | `scripts/phase4_p37_velocity.py` | Appendix H, the layer-to-layer rotation |
| `layerstab_ci_exclL45.json` | `scripts/phase4_layerstab_ci_exclL45.py` | §8.3 |
| `b1_onsetcurve.json`, `b1_onsetcurve_knee.json`, `b1_window_sensitivity.json` | `scripts/b1_onsetcurve.py`, `scripts/lc60_c1_b1_window_sensitivity.py` | §6.3 |
| `b2_covariate_transfer.json`, `per_emotion_ownword_ranks.json` | `scripts/b2_covariate_transfer.py` | §7.4 |
| `g54a_displacement_model.json` | `scripts/g54a_displacement_model.py`, with inputs from `scripts/nogpu_adapters.py` | §7.4 |
| `g66_frequency_control.json` | `scripts/g66_frequency_control.py`, with inputs from `scripts/nogpu_adapters.py` | Appendix F |
| `valence_asymmetry_onset.json` | `scripts/baseit_welfarepaper_scan/g6_6_valence_asymmetry.py` | Appendix F |
| `fair_emotions_v1.json` | `scripts/serialize_fair_emotions.py` | Appendix F |
| `lens_readability_ablations.json` | `scripts/lens_readability_ablations.py` (imports `scripts/phase4_lens.py`) | §6.2, Appendix E |
| `precision_story_summary.json` | `scripts/precision_story_extract.py` (its inputs are not released) | §8.4, Appendix J |
| `job2_layer_permutation_null.json` | `scripts/job2_layer_permutation_null.py` | §7.3 |
| `probe_l45/PROBE_REPORT.md`, `probe_l45/PROBE_REPORT_FOLLOWUP.md` | `probe_l45/analyze_probe_l45.py` (seed 0), then `probe_l45/analyze_followup_l45.py` (seed 1 and the decision rule) | §8.3, Appendix H |
| `parametric_search_artifact_null_SECOND.json` | `scripts/parametric_search_artifact_null_v2.py` | §7.3 |
| `interaction_did_constructions.json` | `scripts/verify_interaction_did_constructions.py` | §3, Appendix E |
| `tuning_fisher_pooled_verification.json` | `scripts/verify_tuning_fisher_pooled.py` | §7.1 |
| `diag_T3_9b_result.json` | `scripts/diag_T3_9b.py` | §7.2 and Appendix G, the 9B centered correlation |
| `g4_pcolon_grade.json` | `colon_it/grade_pcolon.py` | §7.2, Appendix G |
| `op3_band_sweep_analysis.json` | `probe_band/analyze_op3_band_sweep.py` | §8.3 |
| `g6_1_regrade/G6_1_verdicts.json` | `scripts/g6_1_regrade_extract.py` | §6.1 |
| `pmap_embedding_table_transport.json` | `scripts/pmap_embedding_table_transport.py` | §4.3, last paragraph |
| `tuning_dependence_aware_ci_extract.json` | `scripts/tuning_dependence_aware_ci.py`, then `scripts/tuning_dependence_aware_ci_extract.py` | §7.1, last paragraph |

`tuning_dependence_aware_ci_extract.json` carries the 17 rows the paper's §7.1 relies on, each as a point estimate and a 95 percent percentile interval: the twelve per-size tuning deltas, the four pooled deltas, and the arousal interaction. The analysis is unregistered and grades nothing. Its producer reports bootstrap standard errors as well, and the extract leaves them out because on three rows (2B PC2 with pleasure, pooled PC2 with pleasure, and the interaction) the per-draw re-signing of each axis, which flips a sign in 10 of 2,000 draws, inflates the standard error while the interval still describes the row faithfully. The re-signing rule was stated in the analysis's specification before it ran and is not part of the registration. The extract commits by sha256 to the full output, which the producer regenerates.

`lens_readability_ablations.json` holds the layer-0 checks of §6.2 and Appendix E. They read the cleaned vectors without the final normalization, read the uncleaned vectors through the full readout, and pass random directions through the readout at each of the three widths. The producer was written for the release on 2026-10-26 and imports `scripts/phase4_lens.py` unchanged, so it needs `LENS_ASSET_DIR` as that script does. Before it writes, its reference arm must reproduce each model's layer-0 median in `lens_cross_cell.json` exactly, and in the run of record it did.

`precision_story_summary.json` is the record of the bf16 against fp32 comparison in §8.4 and Appendix J. A spot-check re-extracted 40 stories per emotion at each size's analysis layer in both precisions. The file keeps the story measurements and names each size's worst vector, the story vector with the lowest cosine between the two precisions, with the share of its error that falls in its largest dimension. It also carries the two hardware comparisons that §8.4 and Appendix J cite, both on story vectors. The spot-check's activation files are not released, so `scripts/precision_story_extract.py` does not run from the release. Its manifest gives their sha256.

`job1`, `job2` and the raw against cleaned embedding file were first computed without a retained script. Their scripts were written afterwards from the procedure each artifact records, and each reproduces every stored value exactly in the pinned environment. Each of the three has a verify mode, which is its default. The second parametric null ran from code kept in a working memo, which ships unchanged (below).

The §7.3 split search iterates `range(2, L - 2)` in Python, so the split layer takes the values 2 through L minus 3. The procedure field inside `job2_layer_permutation_null.json` writes this range as `[2,L-2]`. The same field also leaves out one step the script runs. The script reseeds the generator with 20261007 for each model size, and the stored null's mean, median and 95th percentile at 9B and 27B depend on that reseeding. The file is kept unedited because it is the one audited artifact never regenerated. The code, the script docstring, and this note give the range and the reseeding correctly, and every number in §7.3 comes from the code.

The battery's driver is `scripts/pmap_battery_serial.py`. It runs the items one at a time with a checkpoint, and when NumPy's default SVD routine fails to converge it repeats that decomposition with SciPy's `gesvd` (Appendix D). `scripts/pmap_battery.py` holds the preparation code that this driver, the disjoint-topic scripts and `scripts/item5_svd_provenance.py` import. Its own entry point is the earlier driver, which has no such fallback, and it is superseded.

`scripts/baseit_welfarepaper_scan/p_lens_axis.py` holds the vocabulary-readout helpers that `g6_6_valence_asymmetry.py` imports. Its own output is not used in the paper and does not ship.

### Producers written after the run

Three results were computed by code that was not kept. Their producers were written on 2026-10-14 from the procedure each result file records, and each reproduces its stored values.

| Result | Producer | What was checked |
|---|---|---|
| `job1_L18_emotion_bootstrap_sensitivity.json` | `scripts/job1_L18_emotion_bootstrap_sensitivity.py` | Reproduces every stored value exactly. The released file is this script's output. Its procedure field was later corrected to state the standardization, with no result changed. |
| `job2_layer_permutation_null.json` | `scripts/job2_layer_permutation_null.py` | Reproduces every stored value exactly. The released file is the original output, unchanged. |
| `c3_raw_cleaned_embedding.json` | `scripts/c3_raw_cleaned_embedding.py` | The original's 24 values were reproduced to within 10⁻¹⁵. The released file is this script's output, and the script reproduces it exactly. |

Two more scripts come from the surviving record of the session that ran the cross-cell transport, the displacement regression and the frequency control. That session's driver and the inputs it built were kept only in a scratch folder that has since been cleared.

| Script | Origin | What was checked |
|---|---|---|
| `scripts/pmap_transport_driver.py` | The run's driver, recovered from the session's record. The released copy changes it only in its adapter-list argument and output path. | Reproduces every value of `pmap_transport.json` exactly. |
| `scripts/nogpu_adapters.py` | Written from the same record. It rebuilds the inputs those three runs read: each model's vectors at its analysis layer, the fair-emotion list under the key the frequency-control loader reads, the layer 0 ranks, and each emotion's neighbourhood purity. | Its purity file is byte-identical to the one the runs read. With its inputs, `pmap_transport.json`, `g54a_displacement_model.json` and `g66_frequency_control.json` reproduce every stored value exactly. |

The run commands:

```
A=<scratch folder>
python gemma_grid/scripts/nogpu_adapters.py --out-dir "$A"
OMP_NUM_THREADS=10 python gemma_grid/scripts/pmap_transport_driver.py --cells "$A/cells_adapters.json"
python gemma_grid/scripts/g66_frequency_control.py --fair-emotions-json "$A/g66_fair_adapter.json" --nrc-csv data/ratings/nrc_vad_v2_1.csv --ranks-json "$A/ownword_ranks.json" --out gemma_grid/geometry_out/g66_frequency_control.json
python gemma_grid/scripts/g54a_displacement_model.py --displacement-csvs gemma_grid/geometry_out/stageB_displacement__gemma-2-2b-base__vs__gemma-2-2b-it.csv gemma_grid/geometry_out/stageB_displacement__gemma-2-9b-base__vs__gemma-2-9b-it.csv gemma_grid/geometry_out/stageB_displacement__gemma-2-27b-base__vs__gemma-2-27b-it.csv --nrc-csv data/ratings/nrc_vad_v2_1.csv --purity-json "$A/purity.json" --l0-ranks-json "$A/g54a_l0scalar.json" --out gemma_grid/geometry_out/g54a_displacement_model.json
```

A further result, `parametric_search_artifact_null_SECOND.json`, ran from code kept in a working memo. That code was committed unchanged as `scripts/parametric_search_artifact_null_v2.py`, and the released file was regenerated with it.

### Extraction, on GPUs

These scripts produced the binaries on Hugging Face. They need the pinned checkpoints, a Gemma access grant and a GPU, so they were not rerun for the release. They compile and import in the analysis environment.

| Binary | Script |
|---|---|
| `data/<model>/vectors/canonical.pt` (raw vectors at every layer) and `<model>__per_story_slice.pt`, five models | `scripts/run_cell_extraction.py --cell gemma_grid/configs/<model>.yaml --mode A`, run in order by `scripts/run_grid_phase3.sh`. `scripts/validate_cell.py` checks each model's shapes and geometry after its run. Pipeline step 2 then cleans the raw vectors. |
| `data/gemma-2-27b-base/vectors/gemma-2-27b-base__per_story_slice.pt` | `scripts/run_cell_extraction.py --mode B`, a rerun of base 27B at its analysis layer |
| `data/<model>/vectors/<model>__neutral.pt`, five models | `scripts/extract_neutral.py --cell gemma_grid/configs/<model>.yaml` |
| `data/gemma-2-27b-base/vectors/gemma-2-27b-base__neutral.pt` | the replication study's own pipeline, which extracted base 27B's neutral activations |
| `colon_it/colon_analysis_it.pt` | `colon_it/run_it_colon.py` |
| `probe_band/probe_band_means_seed0.pt`, `_seed1.pt` | `probe_band/run_op3_band_sweep.py --seed 0`, `--seed 1` |
| `probe_l45/probe_l45_means.pt`, `_seed1.pt` | `probe_l45/run_probe_l45.py`, seed 0 and seed 1 (`probe_l45/_sampling.py`) |

Each model's configuration is in `configs/<model>.yaml`. `scripts/cell_config.py` reads it and checks the model's depth, width and analysis layer at load, and `scripts/dry_run_shapes.py` runs the per-model analysis on random tensors of each model's shape. Story extraction reads the 205,200-story corpus, which ships as a recipe. Mode B checked base 27B's rerun against the replication study's first-round anchors through a verifier and anchor files that this release does not hold, so run from the release it stops with a message unless given `--pilot`, a plumbing check that skips the gate.

### Figures and tables

Figures and tables are in `figures/` under the names their producers write. `figures/gge1_v3_tables_data.json` holds every table value the producer computed.

| Paper | Files in `figures/` | Producer |
|---|---|---|
| Figure 1 | `gge1_v3_fig1_design.{pdf,png}`, `gge1_v3_fig1_design_data.json` | `figures/scripts/fig1_design.py` |
| Figure 2 | `gge1_v3_fig2_transport.{pdf,png}`, `gge1_v3_fig2_transport_data.json` | `figures/scripts/fig2_transport.py` |
| Figure 3 | `gge1_v3_fig3_spread.{pdf,png}`, `gge1_v3_fig3_spread_data.json` | `figures/scripts/fig3_spread.py` |
| Figure 4 | `gge1_v3_fig4_depth_seams.{pdf,png}`, `gge1_v3_fig4_depth_seams_data.json` | `figures/scripts/fig4_depth_seams.py` |
| Table 1 | `gge1_v3_table1_crosscell.md` | `figures/scripts/build_tables.py` |
| Table 2 | in the paper only, rows from `geometry_out/g6_1_regrade/G6_1_verdicts.json` | `scripts/g6_1_regrade_extract.py`, which rebuilds the extract and checks the paper's counts |
| Table 3 | in the paper only, counts from `geometry_out/g6_1_regrade/G6_1_verdicts.json` | `scripts/g6_1_regrade_extract.py` |
| Table 4 | `gge1_v3_table2_tuning_battery.md` (the producer's working number) | `figures/scripts/build_tables.py` |
| Table 5 | `geometry_out/table5_depth_thirds.json` | `scripts/table5_depth_thirds.py`, from `geometry_out/fig84_depth_tuning.json` and `geometry_out/fixed_probe_tuning_shifts_results.json`, checked cell by cell against the printed table |
| Table 6 | `gge1_v3_table3_norm_robustness.md` (the producer's working number) | `figures/scripts/build_tables.py` |

## The audit record

Every claim in the paper that carries a graded verdict passed an adversarial audit: fresh-context sessions instructed to refute each claim by recomputing it from the artifacts. `audit/` holds the seven audits that cover the released results, each as the prompt and the report, indexed in `audit/README.md`. Audit 3 includes one charge on files from a companion analysis that are not part of this release. Its report is reproduced unedited.

The §6.1 grading was run as one blind pass over 144 interleaved rows. `geometry_out/g6_1_regrade/G6_1_verdicts.json` carries the 72 rows behind the paper's counts and commits by hash to the full retained run. That file also records one limit of the blinding: the shuffle seed sits in a manifest the grader could have read, and whether it did cannot be tested from the artifacts.

## Other documents

- `PREREGISTRATION.md`: the registration, unedited. Its sections were locked on 20 July 2026. The decision log continues to 23 July with two process entries, the extraction pilot and the extraction, which add no signal, prediction, or floor.
- `REGISTRATION_CONCORDANCE.md`: the registration concordance, listing every registered signal, floor, and procedure, where the paper reports it, and its status.
- `prestatements/`: the eight pre-stated filings and one dated amendment to the disjoint-topic filing, reproduced exactly as filed, with their hashes and the paper sections they govern.
- `geometry_out/DISJOINT_TOPIC_PARTITION.md` and `geometry_out/disjoint_topic_partition.json`: the corpus's 100 story topics and their split into the two 50-topic sets of the disjoint-topic analysis.
- `geometry_out/family_membership/`: the per-cell family membership tables, one per model, with reading notes in the folder's README.
- `RELEASE_MANIFEST.md`: the release manifest, with the pinned software of the extraction and analysis environments and the run-time checks.

## Citation and license

```bibtex
@misc{hollowell2026grid,
  title         = {The Geometry of Emotion Representations Across Scale, Depth, and Instruction Tuning},
  author        = {Hollowell, Adam},
  year          = {2026},
  eprint        = {[ID ON POSTING]},
  archivePrefix = {arXiv},
  primaryClass  = {[CLASS ON POSTING]}
}

@misc{hollowell2026replicating,
  title         = {Replicating the Geometry of Emotion Representations in a Base Open-Weights Model},
  author        = {Hollowell, Adam},
  year          = {2026},
  eprint        = {2609.22208},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

Code is under the MIT license (`LICENSE`). The Gemma 2 models are subject to Google's Gemma Terms of Use. The vector files, the embedding-table vector files, the neutral-dialogue activation files, the per-story slices and the bootstrap files are Gemma Model Derivatives. They are distributed under the Gemma Terms of Use with its use restrictions, which apply to anyone who receives them (paper Appendix I). Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms. Third-party norms are used under their own terms as described above. Claude-generated data carries the downstream-use notice in the parent repository's `CLAUDE_DATA_NOTICE.md`.
