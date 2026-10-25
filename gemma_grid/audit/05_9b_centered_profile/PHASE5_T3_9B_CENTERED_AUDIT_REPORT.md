# Phase-5 adversarial audit — T.3 9B centered-correlation diagnostic (100T)

- **Date:** 2026-10-09
- **Auditor:** fresh-context Claude Code session (claude-opus-5-5). I had no access to the producing session's reasoning.
- **Prompt:** `gemma_grid/PHASE5_T3_9B_CENTERED_AUDIT_PROMPT_100T.md`
- **Machine-readable companion:** `gemma_grid/geometry_out_100T/phase5_T3_9b_centered_audit_100T.json`
- **Overall verdict:** **SUSTAINED, with caveats.** All six numeric claims reproduce exactly to the stated precision under an independent reader and independent code. Two claims (4 and 5) depend on choices the prompt's wording does not pin down, and one interpretation ("concentrated") is accurate but understated. Required corrections are listed in §5. None of them blocks promotion to the body of §7.2, provided the corrections go into the text.

## 0. Audit hygiene (forbidden files)

I did not open, read, import, or run any of the forbidden files. These are `diag_T3_9b.py`, `diag_T3_9b_result.json`, `audit_T3_9b_centered_diagnostic.py`, `audit_T3_9b_centered_diagnostic_2026-10-09.*`, ledger entry T.3 (I did not open the ledger), `MEMO_T3_9B_DIAGNOSTIC_NOTE_100T.md`, and `SCOPE_T3_9B_CENTERED_DIAGNOSTIC.md`. For full disclosure: one `ls` of `gemma_grid/geometry_out_100T/`, run to orient, printed some of those *filenames* in a directory listing. I did not open any of them. I did not use `tools/ptload.py` or any project `ptload.py`. I did not use torch: the reader is my own restricted unpickler (Appendix A).

Environment: `lens_env/bin/python` (Python 3.9.6, numpy 2.0.2). All arithmetic after loading is float64. The float32 path is reported as a robustness check. macOS Accelerate printed spurious matmul RuntimeWarnings. These are known-benign on this machine, and every output was checked finite.

## 1. Inputs and provenance (claim 7)

The reader reads each file into memory once, hashes those exact bytes, and then parses the zip from the same buffer. Each hash below is therefore the hash of the bytes actually parsed.

| Input | sha256 (taken at read time) | Key / shape used |
|---|---|---|
| `gemma_grid/data/gemma-2-9b-it/vectors/canonical_100T.pt` | `f99508c5ed0f71325cce804f971ceda1ae0cd707d8bf6e45174cecc15b89767a` | `vectors_raw` (171, 42, 3584) float32 → layer 28 |
| `gemma_grid/data/gemma-2-9b-base/vectors/gemma-2-9b-base__neutral_100T.pt` | `723360d2febc35707deed7f6079023810c1e8c114917065d64f21affcee90340` | `acts` (500, 42, 3584) float32 → layer 28 |
| `gemma_grid/data/gemma-2-9b-it/vectors/gemma-2-9b-it__neutral_100T.pt` | `fd557d875961e6f6f12662a9319c99eb3b8233b077dc46362723990339c9ef74` | `acts` (500, 42, 3584) float32 → layer 28 |
| `fixed_probe_tuning_shifts_100T_results.json` | `4d7fe9eacd20cee46257d43b8ea6d1246cd24555194dac605c595727b75b318a` | target fields |
| `data/stories/neutral_dialogues_100T_HA.jsonl` (corpus, provenance check only) | `3838a8f23e726d2433135336477ab0ee0e21f4ecb8ca06613963d26de8a3a04f` | 500 lines |
| `gemma_grid/data/gemma-2-9b-it/vectors/canonical_HA.pt` (follow-up check only) | `1a94c9c3afc5f10fd15c0eac5162d1d483114c5bf58d9bc9625d6092a261f774` | `vectors_raw` equality check |

**Neutral corpus consumed.** Both activation files record `src = neutral_dialogues_100T_HA.jsonl` and `src_sha256 = 3838a8f2…a04f`. The on-disk corpus at `data/stories/neutral_dialogues_100T_HA.jsonl` hashes to the same value. **The two cells consumed the same corpus.** Other metadata from the two files:

| | Base | IT |
|---|---|---|
| Model | `google/gemma-2-9b` | `google/gemma-2-9b-it` |
| Revision (resolved = expected) | `33c19302…` | `11c9b309…` |
| Extraction date (UTC) | 2026-08-13T15:31:52Z | 2026-08-13T15:33:15Z |

Settings common to both files: `tl_compatibility_mode=False`, bf16, `analysis_layer=28`, and pooling "mean over resid_post from token 50 (BOS-inclusive) with short-text fallback".

**Probe provenance.** The probe file's `provenance` string reads "re-cleaned 2026-07-31 against HA neutrals (M.4 re-lock); raw unchanged". This analysis uses `vectors_raw`, which the cleanup does not touch. I verified that `vectors_raw` in `canonical_100T.pt` is bit-identical, at all 42 layers, to `vectors_raw` in `canonical_HA.pt`, with identical emotion order. So the cleaning basis cannot affect any number in this audit. The `_HA` suffix on the corpus name and the HA wording in the probe provenance string are cosmetic for this diagnostic (defect D5).

**Modification check.** See §6.

## 2. Method as implemented (from the prompt's description only)

- **Probes.** `P` is (171, 3584): `vectors_raw[:, 28, :]` from the tuned 9B cell. Each row is unit-normalized.
- **Activations.** `A` is (500, 3584) per cell, layer 28.
- **Profile.** profile_i = mean over n of cos(p_i, a_n). Centered: a_n is replaced by a_n − mean_n(a_n) before the cosine.
- **Correlation.** Pearson r across the 171 emotions, base profile against tuned profile.
- **Dimension drop.** The listed columns are removed from both P and A *before* centering and normalization. Column removal commutes with mean subtraction, so only the normalization is affected.
- **Variance ranking.** Per-dimension variance across the 500 dialogues. This equals the variance of the centered activations, and ddof does not affect the ranking.

## 3. Claims, verdicts, numbers

### Claim 1: centered r = −.1214 and uncentered r = .9993, matching the target fields. **SUSTAINED**

| | Claimed | Target field | Recomputed (f64) | Recomputed (f32) |
|---|---|---|---|---|
| Centered | −.1214 | −0.1214 | **−0.12135313737528508** | −0.12135441183657965 |
| Uncentered | .9993 | 0.9993 | **0.9993392965070411** | 0.999339296861236 |

Both round to the claimed and target values. Spearman gives the same picture: −0.1351 centered, 0.9988 uncentered.

**Reliability attack.** Could the near-zero centered r just reflect noisy profiles? Within-half split reliability before any drop is 0.6207 for the base cell and 0.8941 for the tuned cell. Spearman–Brown stepped up to the full sample, that is ≈ 0.77 and ≈ 0.94, which puts the attenuation ceiling near 0.85. An observed −0.12 is far below that ceiling. The near-zero, slightly negative correlation is therefore not an unreliability artifact.

### Claim 2: the union of the two cells' top-20 variance dimensions has 21 members. **SUSTAINED**

The recomputed union has 21 members: 19 shared, base-only {2885}, tuned-only {2293}.

Union dims: `79, 270, 376, 504, 595, 979, 999, 1132, 1235, 1243, 1279, 1799, 2059, 2293, 2476, 2488, 2558, 2885, 3033, 3186, 3475`.

| Cell | Top-ranked dim (variance) | Rank 20 vs rank 21 variance | Median variance |
|---|---|---|---|
| Base | 504 (686.3) | 52.17 vs 48.74 | 1.22 |
| Tuned | 504 (501.2) | 71.42 vs 67.18 | 2.01 |

The cut at 20 is not knife-edge. The union size does depend on k, as expected: k=19 → 20, k=20 → 21, k=21 → 23, k=30 → 35.

### Claim 3: dropping the 21 union dims gives .9595; dropping the tuned top-20 only gives .9538. **SUSTAINED**

| Drop | Claimed | Recomputed |
|---|---|---|
| 21 union dims | .9595 | **0.9595265411089708** |
| Tuned top-20 only | .9538 | **0.953827913160826** |
| Base top-20 only (extra) | — | 0.9539188482381042 |

Further attacks:

- The Spearman r after the union drop is 0.9694.
- The uncentered r after the union drop is 0.9482 (down from 0.9993). This is not claimed, but it shows the drop also touches the uncentered reading.
- **Dose–response is non-monotone.** Dropping the union of each cell's top-k dims gives:

  | k (dims dropped) | Centered r |
  |---|---|
  | 1 (1) | 0.769 |
  | 2 (2) | 0.925 |
  | 3 (4) | 0.933 |
  | 5 (7) | 0.954 |
  | 8 (10) | 0.973 |
  | 10 (12) | 0.972 |
  | 15 (16) | 0.955 |
  | 20 (21) | 0.9595 |
  | 30 (35) | 0.954 |
  | 50 (54) | 0.867 |
  | 100 (103) | 0.889 |

  k=20 is a reasonable pre-specified choice, but it is neither the peak nor a plateau edge. The text should not imply that 21 is a natural boundary.

### Claim 4: after the drop, odd/even split-half r is .9754 (base) and .9825 (tuned). **UPHELD-WITH-CAVEAT**

| Cell | Claimed | Recomputed: center within each half | Recomputed: center on all 500, then split |
|---|---|---|---|
| Base | .9754 | **0.9754270315581315** | **−0.1818415539628331** |
| Tuned | .9825 | **0.9825038794766928** | **−0.1780927891581009** |

The split uses 0-based indices: odd = 1,3,…; even = 0,2,…. A 1-based labelling swaps the names but gives the same partition and the same r.

**The two readings do not agree.** The claimed numbers reproduce exactly under within-half centering. Under center-then-split, the half-profiles are *negatively* correlated in both cells.

The cause is structural. Under global centering, each half's centered activations have a residual mean of ±δ, with opposite signs in the two halves. As §4 shows, the centered profile is the probes' projection onto a single very small vector. That vector is dominated by ±δ, so the half-profiles anti-correlate. The within-half reading is the correct analogue of the full-sample procedure, because each half is then centered the way the full sample is. But the number is protocol-dependent, and the protocol must be stated (correction C2).

### Claim 5: "concentrated" versus 200 random 21-dim drops (seed 0). **UPHELD-WITH-CAVEAT**

| Generator (seed 0, 21 dims, no replacement) | Mean | SD | Min | Max | Draws ≥ .9595 |
|---|---|---|---|---|---|
| `np.random.default_rng(0).choice` | −0.12572 | 0.04744 | −0.20797 | 0.46679 | **0** |
| `np.random.RandomState(0).choice` (legacy) | −0.12422 | 0.06797 | −0.33740 | 0.76792 | **0** |

**No random draw reaches .9595 under either generator, so the concentration claim stands.** Three caveats:

1. **The SD and max depend on the generator.** "Seed 0" does not identify a draw sequence. The text must name the generator (correction C3).
2. **The SD and max are driven by single lucky draws.** The default_rng max (0.467) is one draw containing dim 1235. The legacy max (0.768) is one draw containing dim 504. Excluding draws that touch any union dim:
   - default_rng: n=176, mean −0.1299, SD 0.0131, range [−0.208, −0.083]
   - legacy: n=182, mean −0.1293, SD 0.0109, range [−0.164, −0.082]

   The honest null is therefore tight around −0.13, and the reported SD (~0.05–0.07) is inflated by union-dim hits.
3. **"Concentrated" understates the result.** Two dimensions carry most of the effect:
   - Dropping dim 504 alone gives r = 0.769.
   - Dropping dim 1235 alone gives r = 0.467.
   - Dropping both (the top-2 union) gives r = 0.925.
   - Every other single union dim, dropped alone, leaves r in [−0.30, +0.05].

   The 21-dim set is a sufficient envelope, not the locus.

### Claim 6: variance share of the union and of each cell's own top-20. **SUSTAINED** (no value was claimed; reported)

| Share of total centered variance (3,584 dims) | Base | Tuned |
|---|---|---|
| 21 union dims | **0.284655413008854** | **0.211207499812183** |
| Own top-20 | **0.28012718864463104** | **0.2068835952823555** |
| Top-1 (dim 504) | 0.0724 | 0.0386 |
| Top-3 | 0.1273 | 0.0799 |
| Uniform expectation for 21 dims | 0.005859 | 0.005859 |

The union carries 36–49× its uniform share of variance.

### Claim 7: provenance. **SUSTAINED**

The hashes, the corpus identity, and the shared-corpus confirmation are in §1. The no-modification confirmation is in §6.

## 4. Follow-up finding: what the centered profile is

The profile has an exact identity:

profile_i = û(p_i) · m̄, where m̄ = mean_n û(x_n)

Here û is unit-normalization and x_n is the (centered or uncentered) activation. I verified it numerically: maximum absolute error 4e-17 centered, 4e-16 uncentered. Each cell's 171-emotion profile is therefore the probe set's projection onto **one vector** per cell. The base-vs-tuned r measures how two vectors look through the probe set.

| m̄ | ‖m̄‖ base | ‖m̄‖ tuned | cos(m̄_base, m̄_tuned) |
|---|---|---|---|
| Uncentered | 0.942 | 0.930 | 0.979 |
| Centered, full | 0.0158 | 0.0233 | 0.321 |
| Centered, after union drop | 0.0332 | 0.0355 | 0.877 |

The centered m̄ would be exactly zero if all centered dialogues had equal norm. It is a second-order quantity, set by norm heterogeneity across dialogues. That explains why the centered profiles are tiny: SD 0.0017 (base) and 0.0024 (tuned), against ~0.149 uncentered. It also explains why the centered profiles are so sensitive to a few high-variance dims: the union carries 27% of the base centered m̄'s squared norm and 55% of the tuned cell's.

This does not refute anything. It is the mechanism behind claims 3–5, and the paper's reader should get one sentence on it (correction C4, recommended).

## 5. Defects and required corrections

**Defects**

| ID | Severity | Defect |
|---|---|---|
| D1 | Moderate | Claim 5's random-draw summary is not reproducible from "seed 0" alone. default_rng and RandomState give SD 0.047 vs 0.068 and max 0.467 vs 0.768. |
| D2 | Moderate | Claim 4 is protocol-dependent. Center-within-half gives .975/.983; center-all-then-split gives −.182/−.178. The text does not say which. |
| D3 | Minor, interpretive | "Concentrated in 21 dims" understates it. Dims 504 and 1235 alone take r from −.12 to .925, and the dose–response over k is non-monotone. |
| D4 | Minor, interpretive | The centered profile is a projection onto a small-norm vector (‖m̄‖ ≈ .016–.023). Its values are ~1/60 the scale of the uncentered profile. |
| D5 | Cosmetic | The corpus filename carries `_HA` and the probe provenance string references HA re-cleaning. Neither affects this raw-probe analysis: `vectors_raw` is bit-identical to `canonical_HA.pt`, and the corpus hash matches across both cells. |

**Required corrections**

- **C2 (required).** State that split-half reliability centers within each half. If space allows, add that centering on the full sample before splitting gives negative half-correlations (≈ −.18) by construction.
- **C3 (required).** Name the generator for the random null, e.g. `numpy.random.default_rng(0).choice(3584, 21, replace=False)`, and report the figures for that generator:

  | Generator | Mean | SD | Min | Max | Draws ≥ .9595 |
  |---|---|---|---|---|---|
  | default_rng | −.1257 | .0474 | −.2080 | .4668 | 0/200 |
  | legacy RandomState | −.1242 | .0680 | −.3374 | .7679 | 0/200 |

  Optionally also report the union-free null: mean ≈ −.130, SD ≈ .012.
- **C3b (recommended).** Strengthen "concentrated" with the two-dim result: dropping dim 504 alone gives .769, and dims 504 + 1235 give .925.
- **C4 (recommended).** One clause noting that the centered profile is small in magnitude, and why.

With C2 and C3 applied, every claim is fit for body-text strength.

## 6. Modification check

I wrote all scratch work to the session scratchpad, outside the project. Before the first read, I touched a timestamp marker. After writing both outputs, I ran `find <project> -newer <marker> -type f`. The result is reproduced below.

```
loving_vector/Loving_Vector_Findings_Ledger_FINAL.md
gemma_grid/PHASE5_T3_9B_CENTERED_AUDIT_REPORT_100T.md
gemma_grid/geometry_out_100T/phase5_T3_9b_centered_audit_100T.json
```

Two of these three files are this audit's named outputs. The third, `loving_vector/Loving_Vector_Findings_Ledger_FINAL.md` (mtime 2026-10-09 12:00:31), was **not written by this audit**. None of my code references that path. My only project writes were the two outputs, made through explicit paths. My only other filesystem actions were reads, `ls`, `unzip -l`, `shasum`, `wc`, and `find`. Its mtime falls during the run, so I attribute it to concurrent activity outside this session (another session, or Dropbox sync). I did not open it. Apart from that, the audit modified nothing in the project except the two named files. Marker time: 2026-10-09 11:56:37 EDT.

## Appendix A — reader (`p5reader.py`, verbatim)

```python
"""Independent torch-free .pt reader (Phase-5 T.3 audit). Written from scratch for this audit.

Reads the whole file into memory once, sha256-hashes those exact bytes, then parses the zip
from the same buffer, so the hash reported is the hash of the bytes actually parsed.
Restricted unpickler: only the globals listed in _ALLOWED are resolvable; anything else raises.
"""
import hashlib
import io
import pickle
import zipfile
from collections import OrderedDict

import numpy as np


class _Storage:
    def __init__(self, kind):
        self.kind = kind


class _FloatStorage:  # marker classes standing in for torch storage types
    kind = "float32"


class _BFloat16Storage:
    kind = "bfloat16"


class _TorchVersion(str):
    pass


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False,
                       backward_hooks=None, metadata=None):
    flat = storage  # numpy 1-D array (already decoded to float32)
    itemsize = flat.itemsize
    if len(size) == 0:
        return np.array(flat[storage_offset])
    view = np.lib.stride_tricks.as_strided(
        flat[storage_offset:], shape=tuple(size),
        strides=tuple(s * itemsize for s in stride), writeable=False)
    return np.array(view)  # materialize a contiguous copy


_ALLOWED = {
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch", "FloatStorage"): _FloatStorage,
    ("torch", "BFloat16Storage"): _BFloat16Storage,
    ("collections", "OrderedDict"): OrderedDict,
    ("torch.torch_version", "TorchVersion"): _TorchVersion,
}


def load_pt(path):
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = zf.namelist()
    pkl = [n for n in names if n.endswith("/data.pkl")]
    assert len(pkl) == 1, pkl
    prefix = pkl[0][: -len("data.pkl")]
    bo = zf.read(prefix + "byteorder").decode().strip() if (prefix + "byteorder") in names else "little"
    assert bo == "little", bo
    cache = {}

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            key = (module, name)
            if key not in _ALLOWED:
                raise pickle.UnpicklingError("blocked global %s.%s" % key)
            return _ALLOWED[key]

        def persistent_load(self, pid):
            assert pid[0] == "storage", pid
            cls, key, _loc, numel = pid[1], pid[2], pid[3], pid[4]
            if key not in cache:
                buf = zf.read(prefix + "data/" + key)
                if cls.kind == "float32":
                    arr = np.frombuffer(buf, dtype="<f4")
                elif cls.kind == "bfloat16":
                    u = np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16
                    arr = u.view(np.float32)
                else:
                    raise ValueError(cls)
                assert arr.size == numel, (arr.size, numel)
                cache[key] = arr
            return cache[key]

    obj = _U(io.BytesIO(zf.read(pkl[0]))).load()
    return obj, sha
```

## Appendix B — main audit code (`p5audit.py`, verbatim)

```python
"""Phase-5 T.3 9B centered-correlation audit: independent recomputation (written from scratch).

Usage: python p5audit.py <project_root> <out_json>
All arithmetic in float64 after loading float32 artifacts.
"""
import hashlib
import json
import sys

import numpy as np

import p5reader

ROOT = sys.argv[1].rstrip("/") + "/"
OUT = sys.argv[2]
L = 28
PROBES = "gemma_grid/data/gemma-2-9b-it/vectors/canonical_100T.pt"
BASE = "gemma_grid/data/gemma-2-9b-base/vectors/gemma-2-9b-base__neutral_100T.pt"
TUNED = "gemma_grid/data/gemma-2-9b-it/vectors/gemma-2-9b-it__neutral_100T.pt"
TARGET = "fixed_probe_tuning_shifts_100T_results.json"
CORPUS = "data/stories/neutral_dialogues_100T_HA.jsonl"

R = {"inputs": {}}


def sha_file(p):
    return hashlib.sha256(open(ROOT + p, "rb").read()).hexdigest()


# ---------------- load ----------------
po, psha = p5reader.load_pt(ROOT + PROBES)
bo, bsha = p5reader.load_pt(ROOT + BASE)
to, tsha = p5reader.load_pt(ROOT + TUNED)
tgt_raw = open(ROOT + TARGET, "rb").read()
tgt = json.loads(tgt_raw)
R["inputs"] = {
    PROBES: {"sha256": psha, "key": "vectors_raw", "shape": list(po["vectors_raw"].shape),
             "variant": po.get("variant"), "provenance": po.get("provenance")},
    BASE: {"sha256": bsha, "key": "acts", "shape": list(bo["acts"].shape), "src": bo["src"],
           "src_sha256": bo["src_sha256"], "model_name": bo["model_name"],
           "hf_revision": bo["hf_revision_resolved"], "date_utc": bo["provenance"]["date_utc"],
           "pooling": bo["provenance"]["pooling"], "config": bo["config"]},
    TUNED: {"sha256": tsha, "key": "acts", "shape": list(to["acts"].shape), "src": to["src"],
            "src_sha256": to["src_sha256"], "model_name": to["model_name"],
            "hf_revision": to["hf_revision_resolved"], "date_utc": to["provenance"]["date_utc"],
            "pooling": to["provenance"]["pooling"], "config": to["config"]},
    TARGET: {"sha256": hashlib.sha256(tgt_raw).hexdigest()},
    CORPUS: {"sha256": sha_file(CORPUS)},
}
R["provenance"] = {
    "same_corpus_name": bo["src"] == to["src"],
    "same_corpus_sha": bo["src_sha256"] == to["src_sha256"],
    "corpus_on_disk_sha_matches_recorded": sha_file(CORPUS) == bo["src_sha256"],
}

emotions = list(po["emotions"])
assert len(emotions) == 171
P = po["vectors_raw"][:, L, :].astype(np.float64)        # (171, 3584)
Ab = bo["acts"][:, L, :].astype(np.float64)              # (500, 3584)
At = to["acts"][:, L, :].astype(np.float64)
del po, bo, to
D = P.shape[1]
assert P.shape == (171, 3584) and Ab.shape == (500, 3584) and At.shape == (500, 3584)


# ---------------- primitives ----------------
def unit_rows(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def profile(A, Pm, centered, center_mean=None):
    """Mean over dialogues of cos(probe, activation). centered: subtract dialogue mean first
    (center_mean overrides the mean used, for the center-then-split variant)."""
    X = A
    if centered:
        mu = A.mean(axis=0) if center_mean is None else center_mean
        X = A - mu
    C = unit_rows(X) @ unit_rows(Pm).T                      # (n, 171)
    return C.mean(axis=0)


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / np.sqrt((x @ x) * (y @ y)))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return pearson(rx.astype(float), ry.astype(float))


def keep_mask(drop):
    m = np.ones(D, bool); m[list(drop)] = False; return m


def r_after_drop(drop, centered=True):
    m = keep_mask(drop)
    pb = profile(Ab[:, m], P[:, m], centered)
    pt = profile(At[:, m], P[:, m], centered)
    return pearson(pb, pt), pb, pt


# ---------------- claim 1 ----------------
r_c, pb_c, pt_c = r_after_drop([], True)
r_u, pb_u, pt_u = r_after_drop([], False)
# float32 path as robustness check
P32, Ab32, At32 = P.astype(np.float32), Ab.astype(np.float32), At.astype(np.float32)
def prof32(A, centered):
    X = A - A.mean(0) if centered else A
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Q = P32 / np.linalg.norm(P32, axis=1, keepdims=True)
    return (X @ Q.T).mean(0)
r_c32 = pearson(prof32(Ab32, True).astype(float), prof32(At32, True).astype(float))
r_u32 = pearson(prof32(Ab32, False).astype(float), prof32(At32, False).astype(float))
tc = tgt["9B"]["ITprobes_centered"]["r_structure_base_vs_it"]
tu = tgt["9B"]["ITprobes_uncentered"]["r_structure_base_vs_it"]
R["claim1"] = {
    "claimed_centered": -0.1214, "claimed_uncentered": 0.9993,
    "target_centered": tc, "target_uncentered": tu,
    "recomputed_centered": r_c, "recomputed_uncentered": r_u,
    "recomputed_centered_float32": r_c32, "recomputed_uncentered_float32": r_u32,
    "spearman_centered": spearman(pb_c, pt_c), "spearman_uncentered": spearman(pb_u, pt_u),
    "round4_match_centered": round(r_c, 4) == -0.1214 == tc,
    "round4_match_uncentered": round(r_u, 4) == 0.9993 == tu,
    "profile_stats": {
        "base_centered_mean": float(pb_c.mean()), "base_centered_sd": float(pb_c.std()),
        "tuned_centered_mean": float(pt_c.mean()), "tuned_centered_sd": float(pt_c.std()),
        "base_uncentered_mean": float(pb_u.mean()), "base_uncentered_sd": float(pb_u.std()),
        "tuned_uncentered_mean": float(pt_u.mean()), "tuned_uncentered_sd": float(pt_u.std()),
    },
}

# ---------------- claim 2 ----------------
var_b = Ab.var(axis=0)    # variance across dialogues == variance of centered acts; ddof irrelevant to rank
var_t = At.var(axis=0)
ord_b = np.argsort(-var_b, kind="stable"); ord_t = np.argsort(-var_t, kind="stable")
top_b = [int(i) for i in ord_b[:20]]; top_t = [int(i) for i in ord_t[:20]]
union = sorted(set(top_b) | set(top_t))
R["claim2"] = {
    "claimed_union_size": 21, "recomputed_union_size": len(union), "union_dims": union,
    "top20_base": top_b, "top20_tuned": top_t,
    "intersection_size": len(set(top_b) & set(top_t)),
    "base_only": sorted(set(top_b) - set(top_t)), "tuned_only": sorted(set(top_t) - set(top_b)),
    "rank20_rank21_var_base": [float(var_b[ord_b[19]]), float(var_b[ord_b[20]])],
    "rank20_rank21_var_tuned": [float(var_t[ord_t[19]]), float(var_t[ord_t[20]])],
    "top_var_base": [[int(i), float(var_b[i])] for i in ord_b[:25]],
    "top_var_tuned": [[int(i), float(var_t[i])] for i in ord_t[:25]],
    "median_var_base": float(np.median(var_b)), "median_var_tuned": float(np.median(var_t)),
    "union_size_by_k": {k: len(set(ord_b[:k].tolist()) | set(ord_t[:k].tolist()))
                        for k in (5, 10, 15, 19, 20, 21, 25, 30, 50)},
}

# ---------------- claim 3 ----------------
r_union, pb_x, pt_x = r_after_drop(union, True)
r_tonly, _, _ = r_after_drop(top_t, True)
r_bonly, _, _ = r_after_drop(top_b, True)
r_union_unc, _, _ = r_after_drop(union, False)
# dose-response: drop union of top-k for several k
curve = {}
for k in (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100):
    u = sorted(set(ord_b[:k].tolist()) | set(ord_t[:k].tolist()))
    curve[k] = {"n_dropped": len(u), "r_centered": r_after_drop(u, True)[0]}
R["claim3"] = {
    "claimed_union_drop": 0.9595, "recomputed_union_drop": r_union,
    "claimed_tuned_top20_drop": 0.9538, "recomputed_tuned_top20_drop": r_tonly,
    "base_top20_drop_extra": r_bonly,
    "union_drop_uncentered_extra": r_union_unc,
    "spearman_union_drop": spearman(pb_x, pt_x),
    "dose_response_union_topk": curve,
}

# ---------------- claim 4 ----------------
m = keep_mask(union)
Pk = P[:, m]
idx = np.arange(500)
odd, even = idx[1::2], idx[0::2]   # 0-based; partition identical under 1-based naming (labels swap)
split = {}
for name, A in (("base", Ab[:, m]), ("tuned", At[:, m])):
    within = pearson(profile(A[odd], Pk, True), profile(A[even], Pk, True))
    mu = A.mean(axis=0)
    global_first = pearson(profile(A[odd], Pk, True, center_mean=mu),
                           profile(A[even], Pk, True, center_mean=mu))
    split[name] = {"within_half_centering": within, "center_all_then_split": global_first}
# before-drop split-half reliability (extra)
split_nodrop = {}
for name, A in (("base", Ab), ("tuned", At)):
    split_nodrop[name] = pearson(profile(A[odd], P, True), profile(A[even], P, True))
R["claim4"] = {"claimed": {"base": 0.9754, "tuned": 0.9825}, "recomputed": split,
               "no_drop_within_half_extra": split_nodrop}

# ---------------- claim 5 ----------------
def random_draws(rng_kind):
    vals = []
    if rng_kind == "default_rng":
        g = np.random.default_rng(0)
        draws = [g.choice(D, size=21, replace=False) for _ in range(200)]
    else:
        g = np.random.RandomState(0)
        draws = [g.choice(D, size=21, replace=False) for _ in range(200)]
    for d in draws:
        vals.append(r_after_drop(d.tolist(), True)[0])
    v = np.array(vals)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()),
            "max": float(v.max()), "n": int(v.size),
            "n_reaching_claim3": int((v >= 0.9595).sum()),
            "n_reaching_recomputed_claim3": int((v >= r_union).sum()),
            "n_draws_hitting_any_union_dim": int(sum(len(set(d.tolist()) & set(union)) > 0 for d in draws)),
            "max_draw_overlap_with_union": int(max(len(set(d.tolist()) & set(union)) for d in draws)),
            "values": [float(x) for x in v]}
R["claim5"] = {"default_rng_seed0": random_draws("default_rng"),
               "legacy_RandomState_seed0": random_draws("legacy")}

# ---------------- claim 6 ----------------
def share(var, dims):
    return float(var[list(dims)].sum() / var.sum())
R["claim6"] = {
    "base_union21_share": share(var_b, union), "tuned_union21_share": share(var_t, union),
    "base_own_top20_share": share(var_b, top_b), "tuned_own_top20_share": share(var_t, top_t),
    "uniform_21_expected_share": 21 / D,
    "base_top1_share": share(var_b, top_b[:1]), "tuned_top1_share": share(var_t, top_t[:1]),
    "base_top3_share": share(var_b, top_b[:3]), "tuned_top3_share": share(var_t, top_t[:3]),
}

# ---------------- extra attacks ----------------
# (a) is the centered collapse driven by the probes' own large components on the same dims?
pn = np.linalg.norm(P, axis=1)
R["extra"] = {
    "probe_norm_share_on_union_mean": float(((P[:, union] ** 2).sum(1) / pn ** 2).mean()),
    "probe_norm_share_on_union_max": float(((P[:, union] ** 2).sum(1) / pn ** 2).max()),
    "mean_act_norm_share_on_union_base": float((Ab.mean(0)[union] ** 2).sum() / (Ab.mean(0) ** 2).sum()),
    "mean_act_norm_share_on_union_tuned": float((At.mean(0)[union] ** 2).sum() / (At.mean(0) ** 2).sum()),
}
# (b) drop the union from activations only / probes only (method says both; check sensitivity)
pb_ao = profile(Ab[:, m], P[:, m], True)  # same as both (column removal) -- check nontrivial variant:
# zero the dims in activations only but keep probe full-dim (probe norm includes dropped dims)
Abz = Ab.copy(); Abz[:, union] = 0; Atz = At.copy(); Atz[:, union] = 0
R["extra"]["zero_acts_only_keep_probe_full_r"] = pearson(profile(Abz, P, True), profile(Atz, P, True))
Pz = P.copy(); Pz[:, union] = 0
R["extra"]["zero_probes_only_keep_acts_full_r"] = pearson(profile(Ab, Pz, True), profile(At, Pz, True))
# (c) top-20 by variance of acts at one cell, using variance after cosine-normalization? (alt ranking)
R["emotions"] = emotions

json.dump(R, open(OUT, "w"), indent=1)

# console digest
for k in ("claim1", "claim2", "claim3", "claim4", "claim6", "extra", "provenance"):
    d = dict(R[k]); print(k, json.dumps(d, default=str)[:3000])
for k, v in R["claim5"].items():
    print("claim5", k, {kk: vv for kk, vv in v.items() if kk != "values"})
```

## Appendix C — follow-up attacks (`p5extra.py`, verbatim)

```python
"""Phase-5 T.3 audit: follow-up attacks (written from scratch). Usage: python p5extra.py <root> <main_json> <out_json>"""
import json
import sys

import numpy as np

import p5reader

ROOT = sys.argv[1].rstrip("/") + "/"
M = json.load(open(sys.argv[2]))
OUT = sys.argv[3]
L = 28
po, _ = p5reader.load_pt(ROOT + "gemma_grid/data/gemma-2-9b-it/vectors/canonical_100T.pt")
ph, phsha = p5reader.load_pt(ROOT + "gemma_grid/data/gemma-2-9b-it/vectors/canonical_HA.pt")
Ab = p5reader.load_pt(ROOT + "gemma_grid/data/gemma-2-9b-base/vectors/gemma-2-9b-base__neutral_100T.pt")[0]["acts"][:, L, :].astype(np.float64)
At = p5reader.load_pt(ROOT + "gemma_grid/data/gemma-2-9b-it/vectors/gemma-2-9b-it__neutral_100T.pt")[0]["acts"][:, L, :].astype(np.float64)
P = po["vectors_raw"][:, L, :].astype(np.float64)
X = {}
X["canonical_HA_sha256"] = phsha
X["vectors_raw_100T_equals_HA_all_layers"] = bool(np.array_equal(po["vectors_raw"], ph["vectors_raw"]))
X["emotions_equal_100T_HA"] = list(po["emotions"]) == list(ph["emotions"])
del ph
union = M["claim2"]["union_dims"]
D = P.shape[1]


def unit(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


# Identity: profile_i = unit(p_i) . mbar, where mbar = mean_n unit(x_n). So each profile is the probes'
# projection onto ONE direction per cell; the base-vs-tuned r compares two directions through the probe set.
def mbar(A, centered, keep=None):
    Z = A if keep is None else A[:, keep]
    if centered:
        Z = Z - Z.mean(0)
    return unit(Z).mean(0)


for tag, cen in (("centered", True), ("uncentered", False)):
    mb, mt = mbar(Ab, cen), mbar(At, cen)
    Q = unit(P)
    pb_direct = (unit(Ab - Ab.mean(0) if cen else Ab) @ Q.T).mean(0)
    X["identity_maxabs_err_" + tag] = float(np.abs(Q @ mb - pb_direct).max())
    X["norm_mbar_base_" + tag] = float(np.linalg.norm(mb))
    X["norm_mbar_tuned_" + tag] = float(np.linalg.norm(mt))
    X["cos_mbar_base_tuned_" + tag] = float(mb @ mt / np.linalg.norm(mb) / np.linalg.norm(mt))
keep = np.ones(D, bool); keep[union] = False
mb, mt = mbar(Ab, True, keep), mbar(At, True, keep)
X["norm_mbar_base_centered_after_drop"] = float(np.linalg.norm(mb))
X["norm_mbar_tuned_centered_after_drop"] = float(np.linalg.norm(mt))
X["cos_mbar_base_tuned_centered_after_drop"] = float(mb @ mt / np.linalg.norm(mb) / np.linalg.norm(mt))
mbf, mtf = mbar(Ab, True), mbar(At, True)
X["mbar_centered_share_on_union_base"] = float((mbf[union] ** 2).sum() / (mbf ** 2).sum())
X["mbar_centered_share_on_union_tuned"] = float((mtf[union] ** 2).sum() / (mtf ** 2).sum())
X["mbar_centered_top5_dims_base"] = [[int(i), float(mbf[i])] for i in np.argsort(-np.abs(mbf))[:5]]
X["mbar_centered_top5_dims_tuned"] = [[int(i), float(mtf[i])] for i in np.argsort(-np.abs(mtf))[:5]]

# Random-draw decomposition: which draws gave the max; distribution excluding draws touching the union
for kind in ("default_rng", "legacy"):
    g = np.random.default_rng(0) if kind == "default_rng" else np.random.RandomState(0)
    draws = [g.choice(D, size=21, replace=False) for _ in range(200)]
    vals = np.array(M["claim5"]["default_rng_seed0" if kind == "default_rng" else "legacy_RandomState_seed0"]["values"])
    touch = np.array([len(set(d.tolist()) & set(union)) > 0 for d in draws])
    clean = vals[~touch]
    top = np.argsort(-vals)[:5]
    X["random_" + kind] = {
        "n_clean": int((~touch).sum()),
        "clean_mean": float(clean.mean()), "clean_sd": float(clean.std(ddof=1)),
        "clean_min": float(clean.min()), "clean_max": float(clean.max()),
        "top5": [{"r": float(vals[i]), "union_dims_in_draw": sorted(set(draws[i].tolist()) & set(union))} for i in top],
    }

# Single-dim drops for each union member (which dims carry the collapse)
single = {}
for dim in union:
    k = np.ones(D, bool); k[dim] = False
    Q = unit(P[:, k])
    single[int(dim)] = pearson(Q @ mbar(Ab, True, k), Q @ mbar(At, True, k))
X["single_dim_drop_r"] = single

json.dump(X, open(OUT, "w"), indent=1)
print(json.dumps(X, indent=1))
```
