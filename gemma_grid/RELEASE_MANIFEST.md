# Release manifest

This file records the software environment the study ran in. The hash manifest of every shipped file is `SHA256SUMS.txt`, described in the README.

## Inputs of record

| Input | sha256 | Where |
|---|---|---|
| Neutral-dialogue corpus, 500 dialogues | `3838a8f23e726d2433135336477ab0ee0e21f4ecb8ca06613963d26de8a3a04f` | parent repository, `data/stories/neutral_dialogues.jsonl` |

The model checkpoints are pinned by revision in `handoff/hf_model_revisions_VERIFIED.json`, and every vector and activation file is listed by hash in the README.

## Anchor gates

Base 27B is the grid's anchor because the replication study (Hollowell 2026) measured it. Three gates check that the grid's pipeline reproduces that study's numbers before any comparison across models is read.

1. **Extraction.** Rerun on the grid's extraction path, the anchor's raw emotion vectors match the replication study's locked extraction on 24 of 24 checked anchors, with differences of about 1e-9. The gate concerns raw vectors, before any cleanup. `PREREGISTRATION.md` records it in its decision-log entry of 2026-07-23.
2. **Cleanup.** The pipeline rebuilds the anchor's cleanup basis from the neutral-dialogue activations and re-cleans the raw vectors. The result reproduces the replication study's released vectors (`6de2e293…`) at relative difference 7.11e-07 at layer 31, with 28 cleanup components. `scripts/run_grid_phase4.sh` stops at step 2 if the difference exceeds 1e-6.
3. **Bootstrap.** Before a model's bootstrap draws are used, its per-model means rebuilt from the per-story slices must reproduce the stored vectors. Each `bootstrap_B1000.npz` records its own check in `meta.reproduction_gate`:

| Model | Raw | Cleaned | Cleanup components |
|---|---|---|---|
| base 2B | 1.59e-06 | 1.47e-06 | 11 |
| tuned 2B | 6.74e-07 | 6.22e-07 | 14 |
| base 9B | 7.01e-07 | 7.07e-07 | 11 |
| tuned 9B | 7.15e-07 | 5.67e-07 | 19 |
| base 27B | 2.18e-06 | 1.08e-06 | 28 |
| tuned 27B | 1.19e-06 | 7.02e-07 | 27 |

The six checkpoints tie the input embedding to the unembedding matrix, which the static token-embedding analysis of §5.1 depends on. `scripts/verify_tied_embeddings.py` checks this at the weight level. The anchor's grand mean over stories is computed at layer 31, its analysis layer. Its late-layer values in §8.3 come from re-extractions at layers 31, 42 and 45.

## Software environment

Extraction ran on GPU machines and produced the activations and vectors. Analysis ran on CPU machines and produced every number downstream of the vectors. Analysis used two Python environments on the same Mac, a main one and a second one for the transport battery, two exploratory analyses, and the disjoint-topic transport analyses, so the table has three environments.

| Asset | Version | Environment | License | Citation |
|---|---|---|---|---|
| PyTorch | 2.8.0 with CUDA 12.8 | extraction | BSD 3-Clause | Paszke et al., 2019 |
| PyTorch | 2.8.0, CPU | analysis, main | BSD 3-Clause | Paszke et al., 2019 |
| PyTorch | 2.12.0, CPU | analysis, transport battery | BSD 3-Clause | Paszke et al., 2019 |
| TransformerLens | 3.3.0 | extraction | MIT | Nanda and Bloom, 2022 |
| Transformers (Hugging Face) | 5.15.0 | extraction | Apache 2.0 | Wolf et al., 2020 |
| NumPy | 2.1.2 | extraction | BSD 3-Clause | Harris et al., 2020 |
| NumPy | 2.0.2 | analysis, main | BSD 3-Clause | Harris et al., 2020 |
| NumPy | 2.4.6 | analysis, transport battery | BSD 3-Clause | Harris et al., 2020 |
| SciPy | 1.13.1 | analysis, main | BSD 3-Clause | Virtanen et al., 2020 |
| SciPy | 1.17.1 | analysis, transport battery | BSD 3-Clause | Virtanen et al., 2020 |
| scikit-learn | 1.6.1 | analysis, main | BSD 3-Clause | Pedregosa et al., 2011 |

Extraction ran in `bfloat16` on NVIDIA H100 80 GB accelerators under Python 3.12.3, with TransformerLens in the raw-weights regime (`tl_compatibility_mode=False`). The full package lists of the two extraction machines behind the released vectors and neutral-dialogue activations are `provenance/B_env_freeze.json` and `provenance/P_env_freeze.json`.

The main analysis environment ran on a Mac under Python 3.9.6 (`provenance/analysis_env_freeze.txt`) and, for the transport permutation floor and the seam re-run, on CPU cloud machines under Python 3.10.12 with the same NumPy, SciPy and PyTorch versions (`provenance/ENV_FREEZE_floor.txt`, `provenance/ENV_FREEZE_seam_rerun.txt`).

The transport battery and the two exploratory onset and covariate analyses (`b1_*`, `b2_*`) ran on the same Mac in a separate environment, the project's `.venv` (Python 3.12.13, Apple Accelerate, single-threaded BLAS), which is also the replication study's environment. Its full package list is `provenance/transport_battery_env_freeze.txt`, read from the environment's package metadata. Every package in it was installed by 2026-08-25, before those runs, and none has changed since. The versions match the run record. The battery and the two exploratory runs import no scikit-learn. The disjoint-topic runs listed below import it (1.8.0) and also record `tokenizers` 0.22.2, both in the package list. The transport battery reproduces its stored values exactly in this environment. In the main analysis environment its median cosines can differ in the last binary digit.

## Disjoint-topic transport outputs

These are the outputs of the disjoint-topic analysis and the corrected split-half reference, governed by `prestatements/DISJOINT_TOPIC_TRANSPORT.md` and its Amendment 1. Each ships with its sibling manifest, `<file>.MANIFEST.json`, which records the script, the input hashes, and the environment. The files were produced under names ending in `_100T`. The release drops that suffix, the bytes are unchanged, and the hashes below identify them.

Run in the `.venv` environment (Python 3.12.13), producer `scripts/item3_corrected.py` for the corrected split-half and `scripts/disjoint_topic_transport.py` for the disjoint-topic files:

| File | sha256 | Manifest sha256 | What it is | Status |
|---|---|---|---|---|
| `geometry_out/item3_splithalf_corrected.json` | `267f2eefbf837fd628a3bf5d28a3e6729883ddfaa84720ce98444cb2bccd89a3` | `1c9adf07ea4f3817a24e0ecb13d0d53d7b83f6adf914528b5b3d0587669028e6` | Part A: the corrected within-model split-half reference | graded |
| `geometry_out/disjoint_topic_part1_crosscell_disjoint.json` | `bf4ae7d7b318884cbdb37712f302bab8c1a6f16a8dc2a0a806ebd8f8e5049c68` | `77dcc53405c450bcd6d1c28614110edb0c3e724827a73362ad106eccc44fcd51` | Part B1: cross-model transport between disjoint topic sets; carries the topic partition | graded |
| `geometry_out/disjoint_topic_part2_matched_halves.json` | `3888e2b2ed4ad79547b6fa9498f10930c8bc93dcf8ce5f14b672b04d341bcb96` | `21bebb906eaad8175de4f51bec807718fced07a74aacc5c8afe1e7b170bbfd2c` | Part B2: matched same-topic control | graded |
| `geometry_out/disjoint_topic_part3_withincell.json` | `8b6118ed5306e0a5f146157ea5b1b6ebc22ed9c54705f868cb7f1384b96754fd` | `9b72d3dea6e21b4103e2e9e208f9d18e94d3799eb5fc21c4e16d571395681890` | Part B3: within-model topic-split reference | graded |
| `geometry_out/disjoint_topic_bow.json` | `3a28c857ddec0a85ed024f747e993290de2f02490fe36d51211c5e01fa2297fc` | `f81b28a0558bf9de9e72892a9c442639303667d8dbed3efc311b6cd413f9bebe` | Part B: bag-of-words baseline, direct form only | graded |

Run in the main analysis environment (Python 3.9.6), producer `scripts/item3_disjoint_gates.py`:

| File | sha256 | Manifest sha256 | What it is | Status |
|---|---|---|---|---|
| `geometry_out/item3_disjoint_gates.json` | `c0c90f9ea80bcdcf33ca152c95d31bc74fd1466c92fe36357971e8a724f9c125` | `52d9a344cad8d2ca0bad266c7fbe47616c385fec2df19fcedd6dada7732496f0` | Diagnostic input checks read by the graded producers (gates 2 and 3 must pass); prints no paper number | run-time input |

`scripts/disjoint_topic_transport.py` requires the story corpus, which is not released; not reproducible from the release alone.

Five of these results and their manifests record the analysis environment's folder in `versions/sys_prefix`. In the released copies that field gives the folder relative to the project root (`.venv`), not the absolute path on the machine that ran them, and each manifest's `output_sha256` gives the released file's hash. No other value changed.

The pre-statement's fitted retrieval form of the bag-of-words baseline was withdrawn by Amendment 1 before it was run. It was never computed, and no output for it exists.

## The corrected split-half from the released slices

The released per-story slices omit the neutral rows. `scripts/item3_corrected.py` ran first with a check that read those rows, so the released copy adds one condition: when a slice has no neutral key, the script skips the replay of the stored battery values, that replay's exactness stop, and the equivalence check, and records "skipped: slice has no neutral key" in their fields. No reported value changes. The released script adds a guard that skips an input check absent from the released slices; it reproduces the graded output exactly.

| Script | sha256 |
|---|---|
| As run, `provenance/as_run/item3_corrected_100T.py` | `6a99571a5543059dbb8f9d12d7c022e07c20dd73adaefb92cdb640ede04ae92d` |
| With the guard, before renaming | `ab0df278b71277651a29ed1eb43b2a25ad4d71079fe0011369982ba0799e3901` |
| Released, `scripts/item3_corrected.py` (the guard, the release's names, and the base 27B neutral activations read at their released path) | `1e0282700bc631ce1a29af5602162369161e3dd88b6db53b33d395a18da9713d` |

The run with the guard, on the released slices in the `.venv` environment, reproduced every graded value of `item3_splithalf_corrected.json`: all 24 split-half values, the four range bounds, and every other per-model field. Only the replay fields (now "skipped") and the run time differ. Its output ships beside the graded file:

| File | sha256 |
|---|---|
| `geometry_out/item3_splithalf_corrected.sanitized_slices.json` | `dc54eec9bde79f4d90128eb19e29841ceaa2653345e6d4d59d3778a777fd9f0d` |
| `geometry_out/item3_splithalf_corrected.sanitized_slices.json.MANIFEST.json` | `a6a6bdc91d329c38c7d61a0824609bc9a48336cfc12aac54fee59437cfb6ab21` |

As written by the run, the output hashed `f1ece0f4c413a5c20ff2c8b6dc0f4454e3fa49fc582c79d4b80a36ae5cfbc1b5` and its manifest `6d239b930c53f9d5a61984aa78c96db14e91e59f50fd5de9686ed9eb84414849`. For the release the local paths in both were made relative: the output's `versions/sys_prefix` field, and in the manifest the same field and the six per-story slice paths, which now name the released files. No value changed.

Item 3 of the transport battery (`geometry_out/pmap_battery.json`, producer `pmap_battery_serial.py`) read the neutral rows that the released slices omit. Item 3 of this battery is superseded by `geometry_out/item3_splithalf_corrected.json` and is not reproducible from the released slices. Every other battery item reproduces from the release as it stands.

## Notes on individual results

`geometry_out/fixed_probe_tuning_shifts_results.json` (produced as `fixed_probe_tuning_shifts_100T_results.json`, producer `fixed_probe_tuning_shifts.py`). Each model's `fig84_analog` block stores `mean_abs_shift_by_third`, computed with thirds of floor(n_layers/3) layers. The early and middle thirds each take that many layers, and the late third takes the rest: 8, 8 and 10 layers at 2B, 14 each at 9B, and 15, 15 and 16 at 27B. Table 5 does not use these stored means. It recomputes the thirds, in `scripts/table5_depth_thirds.py`, from the same block's `per_layer_mean_abs_shift` under §7.3's rule, which gives leftover layers to the earliest thirds: 9, 9 and 8 at 2B, 14 each at 9B, and 16, 15 and 15 at 27B. The two rules agree at 9B and differ at 2B and 27B.

## Scripts edited for release

`provenance/SCRIPT_EDITS.md` lists every released script with the name it ran under, the sha256 of the copy that ran, the sha256 of the released copy, and the released lines that differ. The differences are of three kinds. Names drop the working suffix `_100T`. Scratch-folder paths are read from `BOOT_SCRATCH` instead of a fixed folder. Declared edits point scripts at the release's own files, rewrite guards keyed to working file names, add fallbacks for inputs the release does not hold, and rewrite comments, docstrings, printed messages and local names that carried working labels. No declared edit alters a reported value.

The pipeline entry point, `scripts/run_grid_phase4.sh`, was rewritten for the release from the working orchestrator. It drops the working version's derivation phase and its comparison against an earlier working table, checks the release's inputs by hash before the first step, and keeps the anchor gate.

The extraction scripts (README, "Extraction, on GPUs") carry six kinds of declared edit. `scripts/extract_neutral.py` reads the replication study's neutral corpus at its public path, checks its sha256 before the model loads, and writes the released file name. The control mode that compared against an earlier corpus is removed. `colon_it/run_it_colon.py` reads the 12 scenario prompts from the replication study's released copy of the scenario matrix, which holds the same prompts in the same order, and checks the base-side scenario activations by their sha256. `scripts/run_grid_phase3.sh` takes its folders from the environment instead of the GPU machine's fixed paths. Mode B of `scripts/run_cell_extraction.py` stops with a message when its reproduction-gate inputs are absent. These scripts were not rerun, because they need GPUs.

`scripts/g6_1_regrade_extract.py` cannot run from the release. It reads the sealed label key of the §6.1 grading and five companion files, which are withheld by design because they cover the full blind run of 144 rows and only the 72 rows the paper reports are released. Run from the release, the script stops with a message saying so. `geometry_out/g6_1_regrade/G6_1_verdicts.json` ships as the record of the grading and commits to the six withheld files by their sha256.

The three library modules below are the ones whose scratch-folder paths were replaced.

Three library modules held fixed scratch-folder paths from the session that ran them. The released copies read the scratch folder from the `BOOT_SCRATCH` environment variable instead, which the pipeline sets for every step, and two of them add `os` to an existing import line. No other line changed. The wrappers that import these modules set their own scratch folders, so no output depends on the change. Both hashes are listed.

| Script | As run (sha256) | Released (sha256) | Lines changed |
|---|---|---|---|
| `scripts/phase4_lens.py` | `7830ad8dcc3bf9db4535f8b628cace0824fd14c5f645f3653997eb8514e5f04b` | `18bb3aeec2f98523d32d4673ff30b4f856dbc015bda04d31aad930b3444cfaa6` | 41, 42 |
| `scripts/phase4_c3_embedding.py` | `4e5dd0b8d070f31ce7149900aa6106fe05690adceb18de065c8dc3f581b821e7` | `43742779deb5e13a23ead42635f952bfe6570ecf99073cc571b35640f139a3f4` | 33, 39, 50 |
| `scripts/phase4_stageC_bootstrap.py` | `89fd1a8c7f028aaddee0f0719313fc11b68ac7a3a106575dc9544c18fbd6f8c8` | `c8c55c072b23b8e2c994da9a41bcba10d640cc4ab29fd1a4ffd1f08f95619b0f` | 31, 36, 85 |

## Run-time checks

The pipeline (`scripts/run_grid_phase4.sh`) checks the NumPy and scikit-learn versions before its first step and stops if either differs from the main analysis pins above. Several standalone producers check their own pins the same way. SciPy and PyTorch versions are recorded in the package lists rather than checked at run time.

## The linear-algebra driver

Two analyses fall back from NumPy's default SVD driver (gesdd) to the alternative driver (gesvd) when gesdd fails to converge.

- In the transport battery, two of 108 decompositions fell back, one in each direction between 27B base and 27B tuned at k = 40, with all outputs bit-identical (`geometry_out/item5_svd_provenance.json`).
- In the seam and onset bootstrap, five of the 500 draws failed under gesdd in the environment that ran it and were computed under gesvd. On a draw that converges under both drivers, the two agree to 9.1 × 10⁻¹² across all 32 layers, with the same number of components removed and no effect on the integer seam or onset. An independent rerun in a second environment reproduced the failure on one of the five draws, and the gesvd result there matched the record (seam at layer 23). The other four converged there, so whether a draw fails under gesdd depends on the machine and its linear-algebra library.

The CPU environment for the seam re-run records that the alternative driver is available (`provenance/ENV_FREEZE_floor.txt`).
