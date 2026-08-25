# Verified license table — "Licenses for existing assets" appendix
**Checked:** 2026-07-15 · every field verified from the asset's own license page / card / LICENSE file, live.

## Models / weights

| Asset | Version/ID | License (verified) | Relevant condition | Verified at (URL, date) |
|---|---|---|---|---|
| Gemma 2 27B | google/gemma-2-27b (HF tag "License: gemma") | **Gemma Terms of Use** (last modified Apr 1, 2026) | §3.1 Distribution: may redistribute Gemma or "Model Derivatives" only if you (1) impose the §3.2 use restrictions downstream, (2) give recipients the Agreement, (3) mark modified files, (4) include a NOTICE reading "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms." Raw-weight-derived tensors are Model Derivatives (→ §3.1). §1.1(e)/§3.3: model **Outputs are NOT Model Derivatives**; Google claims no rights in Outputs. Not OSI-open; use bound by the Prohibited Use Policy. | huggingface.co/google/gemma-2-27b + ai.google.dev/gemma/terms — 2026-07-15 |
| Claude Sonnet 4.5 (API) | claude-sonnet-4-5-20250929 | **Anthropic Commercial Terms of Service** (eff. Jun 17, 2025) | §B: Customer "owns its Outputs" (Anthropic assigns its rights in Outputs to Customer) → Customer may release model-generated text it created. §D.4: may not use Services/Outputs to "build a competing product or service, including to train competing AI models." Bound by the Usage Policy. No non-commercial restriction on Outputs. | anthropic.com/legal/commercial-terms — 2026-07-15 |

## Held-out corpus sources (HF datasets; assembled, not redistributed — recipe + hashes)

| Asset | Version/ID | License (verified) | Relevant condition | Verified at (URL, date) |
|---|---|---|---|---|
| Pile Uncopyrighted | monology/pile-uncopyrighted | Card YAML `license: other` | A copy of The Pile with copyrighted subsets removed (Books3, BookCorpus2, OpenSubtitles, YTSubtitles, OWT2); keeps only training-permitted subsets. No standard license text beyond "other." | huggingface.co/datasets/monology/pile-uncopyrighted/raw/main/README.md — 2026-07-15 |
| LMSYS-Chat-1M | lmsys/lmsys-chat-1m (gated) | **LMSYS-Chat-1M Dataset License Agreement** (custom) | **Redistribution prohibited** — "Prohibited Transfers: You should not distribute, copy, disclose, assign, sublicense, embed, host, or otherwise transfer the dataset to any third party." Research + commercial use permitted; deletion may be required. | huggingface.co/datasets/lmsys/lmsys-chat-1m — 2026-07-15 |
| Common Corpus | PleIAs/common_corpus | No `license:` field in card YAML; **per-document license** in each row's `license` field | Card: "All data in Common Corpus are either uncopyrighted or freely licensed and may be used for both commercial and non-commercial purposes." Redistribution permitted; each doc carries its own license (public domain, CC, MIT, Licence Ouverte, …). | huggingface.co/datasets/PleIAs/common_corpus/raw/main/README.md — 2026-07-15 |
| Human-Assistant Conversation | Isotonic/human_assistant_conversation | Card YAML `license: afl-3.0` (**Academic Free License v3.0**) | Permissive; redistribution + commercial use allowed with attribution/notice; no non-commercial restriction. | huggingface.co/datasets/Isotonic/human_assistant_conversation/raw/main/README.md — 2026-07-15 |

## Human-norm sources

| Asset | Version/ID | License (verified) | Relevant condition | Verified at (URL, date) |
|---|---|---|---|---|
| Russell & Mehrabian (1977) | J. Research in Personality 11(3):273–294 | **UNRESOLVED** — no separate dataset license (data embedded in the Elsevier article; no open dataset/license page). Factual statement: no dataset license beyond publisher copyright in the article. | Ship the 45-emotion derived table as factual normative ratings with citation + provenance. | (no asset-owned license URL) — 2026-07-15 |
| Warriner, Kuperman & Brysbaert (2013) | 13,915-lemma VAD norms | **CC BY-NC-ND 3.0 Unported** | Non-commercial only; no derivatives; attribution required. | crr.ugent.be/archives/1003 — 2026-07-15 |
| NRC-VAD Lexicon | v2.1 (Mar 2025), S. Mohammad / NRC Canada | **NRC Terms — "Non-Commercial Use Only — Research or Educational"** (© NRC Canada) | Non-commercial research/education only; **"No Redistribution: Do not redistribute the data. Direct interested parties to the lexicon home page."** Cite associated papers; commercial use requires contacting the author. | saifmohammad.com/WebPages/nrc-vad.html — 2026-07-15 |

## Libraries

| Asset | Version/ID | License (verified) | Relevant condition | Verified at (URL, date) |
|---|---|---|---|---|
| TransformerLens | 3.3.0 | **MIT** (© 2022 TransformerLensOrg) | Permissive; retain copyright + permission notice. | raw.githubusercontent.com/TransformerLensOrg/TransformerLens/main/LICENSE — 2026-07-15 |
| scikit-learn | 1.8.0 | **BSD 3-Clause** (© 2007–2026 scikit-learn devs) | Permissive; retain notice; no-endorsement clause. | raw.githubusercontent.com/scikit-learn/scikit-learn/main/COPYING — 2026-07-15 |
| umap-learn | (current) | **BSD 3-Clause** (© 2017 Leland McInnes) | Permissive; retain notice; no-endorsement clause. | raw.githubusercontent.com/lmcinnes/umap/master/LICENSE.txt — 2026-07-15 |
| PyTorch | (current) | **BSD 3-Clause** ("modified BSD"; © 2016– Facebook, Inc. et al.) | Permissive; retain notice; no-endorsement clause. | raw.githubusercontent.com/pytorch/pytorch/main/LICENSE — 2026-07-15 |
| wordfreq | (Robyn Speer) | **Code: Apache-2.0**; **bundled data: CC BY-SA 4.0** | "This license applies to the code only." Data redistributable under CC BY-SA 4.0; SUBTLEX-derived data must credit SUBTLEX. | raw.githubusercontent.com/rspeer/wordfreq/master/LICENSE.txt + README — 2026-07-15 |
| Transformers (Hugging Face) | (current) | **Apache-2.0** (© 2018– The Hugging Face team) | Permissive; retain notice; NOTICE-file passthrough if present. | raw.githubusercontent.com/huggingface/transformers/main/LICENSE — file-verified 2026-07-28 |
| safetensors | (current) | **Apache-2.0** | Permissive; retain notice; NOTICE-file passthrough if present. | raw.githubusercontent.com/huggingface/safetensors/main/LICENSE — file-verified 2026-07-28 |
| huggingface_hub | (current) | **Apache-2.0** | Permissive; retain notice; NOTICE-file passthrough if present. | raw.githubusercontent.com/huggingface/huggingface_hub/main/LICENSE — file-verified 2026-07-28 |
| Pillow | 12.2.0 | **MIT-CMU** (© Secret Labs AB / F. Lundh; Pillow fork © J. A. Clark) | Permissive (MIT-like); retain copyright + permission notice; author's name not to be used in advertising without permission. | raw.githubusercontent.com/python-pillow/Pillow/main/LICENSE — file-verified 2026-07-28 |
| adjustText | 1.4.0 | **MIT** (© 2017 Ilya Flyamer) | Permissive; retain copyright + permission notice. | raw.githubusercontent.com/Phlya/adjustText/master/LICENSE — file-verified 2026-07-28 |
| NumPy | 2.4.6 | **BSD 3-Clause** (© 2005–2025 NumPy Developers) | Permissive; retain notice; no-endorsement clause. | raw.githubusercontent.com/numpy/numpy/main/LICENSE.txt — file-verified 2026-07-28 |
| pandas | 3.0.3 | **BSD 3-Clause** (© 2008– PyData/AQR + contributors) | Permissive; retain notice; no-endorsement clause. | raw.githubusercontent.com/pandas-dev/pandas/main/LICENSE — file-verified 2026-07-28 |
| SciPy | 1.17.1 | **BSD 3-Clause** (© 2001– Enthought/SciPy Developers) | Permissive; retain notice; no-endorsement clause. | raw.githubusercontent.com/scipy/scipy/main/LICENSE.txt — file-verified 2026-07-28 |
| Matplotlib | 3.10.9 | **Matplotlib License** (PSF-based, BSD-compatible; © 2012– Matplotlib Development Team) | Permissive; retain license + copyright; summarize changes in derivatives; no trademark grant. | raw.githubusercontent.com/matplotlib/matplotlib/main/LICENSE/LICENSE — file-verified 2026-07-28 |
| tqdm | 4.67.3 | **MPL-2.0 / MIT** (dual; MPL-2.0 © Casper da Costa-Luis, MIT © original authors) | Permissive; MPL-2.0 file-level copyleft + MIT; retain notices. | raw.githubusercontent.com/tqdm/tqdm/master/LICENCE — file-verified 2026-07-28 |

## Citations found (resolves bib VERIFY flags)

- **Common Corpus** — two distinct papers, do not conflate: the **dataset-description** paper *Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training* (Langlais et al., **arXiv:2506.01732**, ICLR 2026 oral) is the primary reference for the dataset itself; *Toxicity of the Commons* (Arnett et al., **arXiv:2410.22587**) is the toxicity-curation-pipeline companion. The HF card links 2410.22587; for a dataset-source citation the dataset paper (2506.01732) is the more accurate target. No BibTeX block on card.
- **wordfreq** — *Robyn Speer. (2022). rspeer/wordfreq: v3.0 (v3.0.2). Zenodo. https://doi.org/10.5281/zenodo.7199437* (BibTeX key `robyn_speer_2022_7199437`).
- **TransformerLens** — README "Citation": `@misc{nanda2022transformerlens, title={TransformerLens}, author={Neel Nanda and Joseph Bloom}, year={2022}, howpublished={\url{https://github.com/TransformerLensOrg/TransformerLens}}}` (CITATION.cff not present — see UNRESOLVED).

## UNRESOLVED
- **Russell & Mehrabian (1977) dataset license** — none exists; norms are inside the Elsevier article, no open dataset/license page. No asset-owned URL to verify. Best statement: no license beyond publisher copyright.
- **TransformerLens CITATION.cff** — not present at main / v3.3.0 / contents API (all returned empty). Citation taken from repo README instead.
- **LMSYS raw card YAML** — raw README returned empty; the full License Agreement was verified from the rendered dataset card instead.
