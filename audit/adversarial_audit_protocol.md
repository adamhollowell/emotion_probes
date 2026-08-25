# Adversarial Audit — Protocol and Prompt (released basis 6de2e293)

**What this is.** The verbatim commissioning brief given to a fresh-context Claude Opus 4.8
session (the "auditor") instructed to *refute* every headline quantity in the paper by
recomputing it independently from the released artifacts on the canonical basis
(`emotion_vectors_171.pt`, sha256 `6de2e293…`). The auditor had no access to the analysis
sessions' reasoning — only the released repository tree and the findings record. Its verdict
report is the companion `adversarial_audit_report.md`.

**Independence conditions.**
- Fresh context; no producing session's reasoning in scope.
- Read-only on the repository; all recomputation in independently written code (numpy / scipy /
  scikit-learn), torch-free, loading `.pt` via the shipped `outputs/load_pt.py`.
- Every "L31" quantity recomputed by slicing the `(171, 46, 4608)` array at `[:, 31, :]`; no
  cross-layer flatten (positive control: all-46 Frobenius 0.4253 vs L31-slice 0.4108 are distinct
  and both reproduced, proving the layer axis is indexed).
- Each number recomputed from the rawest shipped input, then compared to both the precomputed
  `geometry_out/*.json` and the published value; a mismatch is a finding, not a failure.

---

## Verbatim auditor brief

You are running an INDEPENDENT, ADVERSARIAL end-to-end reproduction pass on a released ML research
repository. Your job is to recompute each numerical finding from the rawest shipped artifact and
confirm it matches the findings record — trying to REFUTE each number, not confirm it. Report
honestly; a mismatch you find is a success, not a failure.

[Environment & paths, hard rules, per-entry recompute list P1.1–P7.1, out-of-scope model/GPU-gated
items, and the required verdict-report format — as executed. The full brief is reproduced from the
commissioning record; the load-bearing constraints are the independence conditions and
the L31-slicing guardrail stated above.]

Scope covered (reproducible from shipped CPU artifacts): PCA variance and axes (P1.1), human-norm
correlations (P1.2/P1.6), norm–valence coupling (P1.4), RSA co-activation (P1.5), clustering and
ARI (P2.1/P2.3), neighbor cosines and purity (P2.2/P2.4), cross-layer stability and the full Table 2
(P3.1/P3.3/P3.3(b)/P3.4/P3.5/P3.6), the embedding-layer baseline (P3.9), the max-activation
structural floor and composition (P6.1), magnitude calibration (P6.4), and the loving masking
control (P7.1).

Out of scope (model/GPU or by-reference gated, disclosed in the report): the logit-lens family
(P4.1/P4.3/P4.5/P5.1, needs Gemma `W_U` / nonpublic `lens_ingredients.pt`), the token-text
own-word rate and I-4 random floor (need decoded token text / full `pertoken.tar`), raw
residual-norm growth (P3.2, raw activations not shipped), and k@L31 (nonpublic neutral projection).
