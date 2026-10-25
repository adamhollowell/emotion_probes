# Phase-5 adversarial audit prompt — T.3 9B centered-correlation diagnostic

Dispatched 2026-10-09 by the GGE1 editing chat on Adam's ruling, per `MEMO_to_E1_adversarial_audit_protocol.md` and M.2 (fresh-context sessions, no access to the producing analysis's reasoning, instructed in writing to refute each claim by independent recomputation from artifacts). Prompt preserved per SR.10. Report: `gemma_grid/PHASE5_T3_9B_CENTERED_AUDIT_REPORT_100T.md`.

Why this audit exists: the T.3 9B diagnostic was filed T4 (footnote-or-cut) and unaudited on 2026-08-23, with a Phase-5 audit owed if it were promoted above a footnote. Adam ruled on 2026-10-09 that it is promoted into the body of §7.2 of the paper. M.2's committed wording holds disclosed exceptions at footnote strength, so promotion requires this audit to pass.

---

You are a fresh-context auditor with a refutation mandate. You have no access to the producing session's reasoning. A rubber-stamp review is a failed review: attack every claim below and try to refute it by independent recomputation from artifacts, using your own code. Do not open, read, import, or re-run the producing script. Verdict vocabulary per claim: SUSTAINED / REFUTED / WEAKENED / UPHELD-WITH-CAVEAT. You may modify NOTHING in the project except writing your report and its machine-readable companion to the named paths. Project root: /Users/adamhollowell/Library/CloudStorage/Dropbox/Claude x Adam/emotion_probes_project (resolve your session's mount path first).

**Files you must not open.** `gemma_grid/research/diagnostics/diag_T3_9b.py` (the producing script). `gemma_grid/geometry_out_100T/diag_T3_9b_result.json` (the producing result). `gemma_grid/scripts/audit_T3_9b_centered_diagnostic.py` and `gemma_grid/geometry_out_100T/audit_T3_9b_centered_diagnostic_2026-10-09.*` (a prior check by another session, which this audit must not see). The ledger entry T.3 in `gemma_grid/Gemma_Grid_Findings_Ledger FINAL.md`. The memo `MEMO_T3_9B_DIAGNOSTIC_NOTE_100T.md` and `SCOPE_T3_9B_CENTERED_DIAGNOSTIC.md`. Treat opening any of these as a failed audit and say so in the report.

**Environment note.** If torch is unavailable in your Python, the `.pt` files are zip archives containing a pickle (`<name>/data.pkl`) and raw storages (`<name>/data/<n>`). Read them with a restricted unpickler that maps `torch._utils._rebuild_tensor_v2` to a numpy strided view and `torch.FloatStorage` to float32 (if a storage class is `BFloat16Storage`, widen uint16 to uint32, shift left 16, view as float32). Do not use `tools/ptload.py` or `ptload.py` from the project tree. Write your own reader and record it in the report.

## Inputs (hash each at read time per SR.5 and print the hashes in the report)

- Probes: `gemma_grid/data/gemma-2-9b-it/vectors/canonical_100T.pt`, key `vectors_raw`, shape (171, 42, 3584). Use layer index 28. Key `emotions` gives the 171 names in order.
- Base activations: `gemma_grid/data/gemma-2-9b-base/vectors/gemma-2-9b-base__neutral_100T.pt`, key `acts`, 500 dialogues by 42 layers by 3584. Use layer index 28.
- Tuned activations: `gemma_grid/data/gemma-2-9b-it/vectors/gemma-2-9b-it__neutral_100T.pt`, key `acts`, same shape. Use layer index 28.
- Target values the claims are asserted against: `fixed_probe_tuning_shifts_100T_results.json` at project root, fields `9B.ITprobes_centered.r_structure_base_vs_it` and `9B.ITprobes_uncentered.r_structure_base_vs_it`.
- Provenance: both activation files should record the neutral corpus they consumed. Report which file and which hash, and whether the two cells consumed the same corpus.

## Method, as the paper describes it (this is all you are given about the procedure)

An emotion's probe is its raw difference-of-means vector (before cleanup) from the tuned 9B cell at layer 28. The activation profile of a cell is, for each of the 171 probes, the mean over the 500 dialogues of the cosine between the probe and the dialogue's pooled layer-28 activation. The centered profile subtracts the mean activation across the 500 dialogues from every dialogue's activation before the cosines are taken. The base-to-tuned correlation is the Pearson correlation across the 171 emotions between the base cell's profile and the tuned cell's profile. A dimension drop removes the named dimensions from both the probes and the activations before any normalization or centering.

## Claims to attack (each gets a verdict, with your number beside the claimed one)

1. The centered base-to-tuned correlation at 9B is −.1214, and the uncentered correlation is .9993. Both must also match the two target fields in `fixed_probe_tuning_shifts_100T_results.json`.
2. Rank the 3,584 dimensions of each cell's centered layer-28 activations by variance across the 500 dialogues. The union of the two cells' top-20 sets has 21 members.
3. Dropping those 21 dimensions raises the centered correlation to .9595. Dropping only the tuned cell's top 20 raises it to .9538.
4. After the 21-dimension drop, split the 500 dialogues odd against even, recompute the centered profile within each half, and correlate the two halves. The result is .9754 for the base cell and .9825 for the tuned cell. Also report what you get if you center on all 500 dialogues first and split afterwards, and say whether the two readings agree.
5. Attack the word "concentrated." Draw 200 random 21-dimension subsets (seed 0, uniform without replacement), drop each from both cells, and report the mean, standard deviation, minimum, and maximum of the centered correlation. State whether any random draw reaches the value in claim 3.
6. Report the share of each cell's centered activation variance carried by the 21 union dimensions, and by each cell's own top 20.
7. Provenance. The hashes you took at read time, the neutral corpus each activation file names, and confirmation that nothing in the project was modified by your run except the two report files.

## Report

Write `gemma_grid/PHASE5_T3_9B_CENTERED_AUDIT_REPORT_100T.md` with a verdict per claim, your independently computed numbers at full precision beside the claimed ones, a defects list, an overall verdict, and required corrections if any. Write the machine-readable numbers to `gemma_grid/geometry_out_100T/phase5_T3_9b_centered_audit_100T.json`. Include the full text of the reader and the audit code you wrote as an appendix to the report. Return a summary.
