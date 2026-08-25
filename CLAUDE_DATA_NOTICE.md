# Notice — Claude-generated data in this repository

Some data files in this repository are **Outputs of Claude Sonnet 4.5**
(`claude-sonnet-4-5-20250929`), generated via the Anthropic API:

- `data/stories/neutral_dialogues.jsonl` — the 500 neutral Human/Assistant dialogues used
  as the confound-cleanup baseline. (Following Sofroniew et al. (2026), speaker labels were
  converted post-hoc from `Person:`/`AI:` to `Human:`/`Assistant:`.)
- `emotion_validation/validation_set.json`, `validation_set.md`, and
  `validation_results_*.csv` — the 300-story validation sample (10 stories × 30
  emotions), a subset of the study's stimulus corpus, with per-story hand-codings.

**Basis for release.** Under Anthropic's Commercial Terms of Service (§B), the
customer owns the Outputs it generates. These files are released on that basis, for
research reproducibility, with attribution to Claude Sonnet 4.5 as the generator.

**Downstream-use restriction (please pass this on).** Consistent with Anthropic's
Commercial Terms (§D.4) and Usage Policy, you may **not** use this data to develop or
train a competing AI model, or otherwise in violation of Anthropic's Usage Policy
(https://www.anthropic.com/legal/aup). If you redistribute this data, include this
notice.

**Not included.** The full **205,200-story stimulus corpus** is *not* redistributed
here. It is released as a **generation recipe** — the verbatim elicitation prompt and
topic/emotion design in the paper's Appendix A, plus the generation scripts in this
repo — from which an equivalent corpus can be regenerated. (Generation is stochastic,
so a regenerated corpus is not identical; the fixed reference for exact numerical
reproduction is the released emotion-vector artifact.)
