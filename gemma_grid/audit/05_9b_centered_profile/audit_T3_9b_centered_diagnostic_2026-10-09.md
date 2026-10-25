# Audit report — T.3 9B centered-correlation diagnostic

Date: 2026-10-09
Run by: GGE1 100T Ledger Chat, as checker (Adam's ruling, 2026-10-09)
Dispatch: GGE1 editing chat, 2026-10-09, "Phase-5 audit of the T.3 9B centered-correlation diagnostic"
Script: `scripts/audit_T3_9b_centered_diagnostic.py` (auditor's own code)
Machine-readable result: `geometry_out_100T/audit_T3_9b_centered_diagnostic_2026-10-09.json`

## Protocol compliance, stated plainly

Satisfied: independent recomputation from the input artifacts, the auditor's own code
written against the dispatch's method description alone, and no access to the producing
script. `research/diagnostics/diag_T3_9b.py` was not opened at any point in this audit.

NOT satisfied: fresh context. This checker is the standing Ledger chat and had already
read `diag_T3_9b_result.json` in the same session, so it knew the target values before
computing. Phase-5 as recorded at M.2 is three fresh-context sessions with no access to
the producing session's reasoning. This run is an independent recomputation by a
non-producing session. It is not a fresh-context Phase-5 session, and the paper must not
describe it with the same word M.2 uses for the study's graded claims.

Torch is unavailable on the verifying machine. Tensors were materialized from the `.pt`
zip members directly (`tools/ptload.py`): raw storage bytes read per the pickle's own
shape, dtype, offset and stride, with a restricted unpickler that honors only
`_rebuild_tensor_v2`, `OrderedDict` and `TorchVersion` and returns inert stubs for
everything else, so nothing from the pickle executes.

## Inputs, hashed at audit time (SR.5)

| role | path | sha256 |
|---|---|---|
| probes (`vectors_raw`, L28) | `data/gemma-2-9b-it/vectors/canonical_100T.pt` | `f99508c5ed0f71325cce804f971ceda1ae0cd707d8bf6e45174cecc15b89767a` |
| base activations (`acts`, L28) | `data/gemma-2-9b-base/vectors/gemma-2-9b-base__neutral_100T.pt` | `723360d2febc35707deed7f6079023810c1e8c114917065d64f21affcee90340` |
| tuned activations (`acts`, L28) | `data/gemma-2-9b-it/vectors/gemma-2-9b-it__neutral_100T.pt` | `fd557d875961e6f6f12662a9319c99eb3b8233b077dc46362723990339c9ef74` |

Both activation files declare `analysis_layer` 28, `d_model` 3584, 500 dialogues, and the
same source corpus `neutral_dialogues_100T_HA.jsonl` at
`3838a8f23e726d2433135336477ab0ee0e21f4ecb8ca06613963d26de8a3a04f`. That is the canonical
100T extraction input: `ha_recompute/neutral_100T_DONE.json` names it
`canonical_extraction_input` and records "Step-2 extraction must verify this stamp and
consume ONLY the _HA file," and its hash is the one carried by the run-phase4 preflight,
the C3 tokens manifest and runbook manifests S1, Q and B. Input provenance verified.

## Verdicts

| claim | asserted | recomputed | verdict |
|---|---|---|---|
| 1 centered r | −.1214 | −0.12135313737528626 | SUSTAINED |
| 1 uncentered r | .9993 | 0.9993392965070416 | SUSTAINED |
| 2 union size | 21 | 21 (19 of 20 shared between cells) | SUSTAINED |
| 3 r after union drop | .9595 | 0.9595265411089704 | SUSTAINED |
| 3 r after IT-top-20 drop | .9538 | 0.9538279131608255 | SUSTAINED |
| 4 reliability, base | .9754 | 0.9754270315581309 | SUSTAINED, one convention only |
| 4 reliability, tuned | .9825 | 0.9825038794766932 | SUSTAINED, one convention only |
| 5 dim 4140 not in union | true | true, and 4140 exceeds 9B's d_model of 3584 | SUSTAINED, trivially |

Two artifact sub-values not listed as claims also reproduce exactly: the per-cell top-5
variance dimensions (base 504/1235/2558/1279/3033, tuned 504/1235/3033/1132/979) and the
tuned top-5 variance share of 9.9 percent.

The union is dimensions 79, 270, 376, 504, 595, 979, 999, 1132, 1235, 1243, 1279, 1799,
2059, 2293, 2476, 2488, 2558, 2885, 3033, 3186, 3475.

## Finding 1 — claim 4's split-half convention is underdetermined (SR.4 serialization gap)

"Odd-even dialogue split after the union drop" has two readings and they do not agree.
Recomputing the centering within each half reproduces .9754 and .9825. Centering on all
500 dialogues and splitting afterwards gives −0.1818 and −0.1781. The reported values are
the within-half convention. The method text must say so wherever the reliability figures
appear, because a reproducer taking the other reading gets a sign flip rather than a near
miss.

## Finding 2 — claim 5 is true by dimensionality, not by measurement

9B's residual stream has 3584 dimensions, so dimension 4140 does not exist at this size.
The statement "dimension 4140 is not in the 9B union" is therefore true but carries no
measurement, and a reader meeting it beside the §8.4 massive-activation discussion will
read a cross-size comparison that the dimensionality forbids. The ledger's T.3 entry
already states the reason. Any promoted text must carry the reason with the claim or drop
the claim.

## Finding 3 — the ledger's own T.3 sentence misnames the .9595 object

T.3 read "dropping the top-20 variance dimensions restores the centered structure
−0.1214 → 0.9595." The .9595 belongs to the 21-dimension union of both cells' top-20
sets. A top-20 drop is a different object and gives .9538. Right number, wrong name.
Corrected in the ledger on 2026-10-09 in the same pass as this report.

## Optional extra 1 — random-21 drop control

200 draws of 21 dimensions uniformly at random, seed 0, dropped from both cells: mean r
−0.1257, sd 0.0474, range −0.2080 to +0.4668. None of the 200 reaches the observed
0.9595. The restoration is specific to these high-variance dimensions and is not what
dropping 21 dimensions does in general.

## Optional extra 2 — variance share

The 21 union dimensions carry 28.5 percent of the base cell's activation variance and
21.1 percent of the tuned cell's. Each cell's own top 20 carry 28.0 and 20.7 percent.

## Labeling consequence for the paper

M.2's committed vocabulary has two slots, audited and unaudited, and this number now sits
in neither. Promotion from a footnote was conditioned on a Phase-5 audit, and what exists
is an independent recomputation by a non-producing session outside that protocol. Dropping
"unaudited" would claim parity the run does not have. Keeping it would deny a check that
did happen. The accurate statement belongs in Appendix F, in a sentence of its own, and
carries no session count and no claim of an independent reviewer.
