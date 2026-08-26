#!/usr/bin/env python3
"""
export_lens_ingredients.py  (lightweight, CPU-friendly)
=======================================================
Regenerate the logit-lens ingredients (`lens_ingredients.pt`) from the base
model `google/gemma-2-27b`, instead of shipping the 4.4 GB file.

Why a recipe and not the file: essentially all of `lens_ingredients.pt` is
`W_U`, the 4608 x 256000 unembedding matrix -- 1.18 B parameters, ~4.4 % of
Gemma-2-27b, and a *verbatim* slab of Gemma's own weights (Gemma-2 ties its
input/output embeddings, so the unembedding IS the embedding table). Shipping
it is redistributing Gemma weights under Google's Gemma Terms. It is fully
re-derivable from the public (gated) base model, so we ship this exporter.
(Reproducibility ledger: M.5.)

Lightweight method: `W_U` is a single tensor. Rather than load the 27B model
(~54 GB, GPU territory -- how it was originally exported on the H100), this
reads ONLY the two tensors we need directly from the model's safetensors:

    model.embed_tokens.weight   [256000, 4608]  ->  W_U = its transpose
    model.norm.weight           [4608]          ->  ln_final_weight

It downloads only the shard(s) holding those tensors (a few GB, not 54) and
never instantiates the model, so it runs on a Mac CPU. The result is validated
against the existing `lens_ingredients.pt` via --verify-against; if `W_U`,
`ln_final_weight`, and `vocab` match, the recipe faithfully reproduces the
original H100/transformer_lens export.

Output schema (matches the original, ledger M.5):
    W_U               [4608, 256000] float32   (d_model, vocab_size)
    ln_final_weight   [4608]         float32   (raw Gemma final-RMSNorm weight)
    ln_final_eps      1e-6
    vocab             list[str] length vocab_size  (id -> token string)
    vocab_size        256000
    d_model           4608
    gemma_rmsnorm_note, ln_final_type, tl_compatibility_mode, provenance

Dependencies (all light; install into lens_env):
    pip install huggingface_hub safetensors        # required
    pip install transformers                        # for the vocab field
You must be logged in to Hugging Face with Gemma access accepted:
    huggingface-cli login        (and accept the license on the model page)

Usage
-----
    # inspect which shards are needed, download nothing heavy:
    python export_lens_ingredients.py --plan-only

    # regenerate + verify against the file you already have:
    python export_lens_ingredients.py --out /tmp/regen.pt \
        --verify-against data/vectors/lens_ingredients.pt
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

MODEL_NAME = "google/gemma-2-27b"
D_MODEL = 4608
VOCAB_SIZE = 256000
LN_FINAL_EPS = 1e-6

EMBED_KEY = "model.embed_tokens.weight"
NORM_KEY = "model.norm.weight"

GEMMA_RMSNORM_NOTE = (
    "Gemma2 RMSNorm: out = x/rms(x) * (1.0 + weight), computed in float32. "
    "The (1+weight) is Gemma-specific."
)
PROVENANCE = (
    "regenerated from google/gemma-2-27b safetensors (embed_tokens.weight -> W_U, "
    "model.norm.weight -> ln_final_weight), rounded through bfloat16 to match the "
    "original bfloat16 extraction regime (draft s3.1 / ledger M.2, M.5); reproduces "
    "the original raw-weights (tl_compatibility_mode=False) export"
)

# Sanity range for the RAW final-RMSNorm weight (ledger M.5 / draft s7.2:
# min ~= -1.10, max ~= 81.50, mean ~= 13.90).
LNW_MIN_LO, LNW_MIN_HI = -1.5, -0.8
LNW_MAX_LO, LNW_MAX_HI = 75.0, 90.0


def log(msg: str) -> None:
    print(msg, flush=True)


def plan_shards(model: str, revision: str | None):
    """Download only the tiny index.json and report which shard(s) hold the two
    tensors we need. Returns (weight_map, needed_shard_names). No heavy download."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        idx = hf_hub_download(model, "model.safetensors.index.json", revision=revision)
        weight_map = json.load(open(idx))["weight_map"]
    except EntryNotFoundError:
        # single-file (unsharded) model
        weight_map = {EMBED_KEY: "model.safetensors", NORM_KEY: "model.safetensors"}

    for k in (EMBED_KEY, NORM_KEY):
        if k not in weight_map:
            raise KeyError(f"{k} not found in the model's weight map; keys look like: "
                           f"{list(weight_map)[:5]} ...")
    needed = sorted({weight_map[EMBED_KEY], weight_map[NORM_KEY]})
    log(f"[plan] {EMBED_KEY} -> {weight_map[EMBED_KEY]}")
    log(f"[plan] {NORM_KEY} -> {weight_map[NORM_KEY]}")
    log(f"[plan] shard(s) that will be downloaded: {needed}")
    return weight_map, needed


def download_shards(model: str, revision: str | None, needed) -> dict:
    """Download the identified shard file(s). This is the multi-GB step."""
    from huggingface_hub import hf_hub_download
    paths = {}
    for name in needed:
        log(f"[download] {name} ...")
        paths[name] = hf_hub_download(model, name, revision=revision)
    return paths


def read_tensor(weight_map, paths, key) -> torch.Tensor:
    from safetensors import safe_open
    shard = weight_map[key]
    with safe_open(paths[shard], framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def build_vocab(model: str, revision: str | None) -> list[str] | None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log("[vocab] transformers not installed -> skipping vocab field. "
            "Install it to produce a drop-in replacement (pip install transformers).")
        return None
    import re
    log("[vocab] loading tokenizer (small download) ...")
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    toks = tok.convert_ids_to_tokens(list(range(VOCAB_SIZE)))
    # Match the original export's surface form:
    #  - SentencePiece word-boundary marker U+2581 ('▁') -> leading space
    #    (see lens_recompute_test disp()'s lstrip(' ')).
    #  - byte-fallback tokens '<0xHH>' -> the raw byte character chr(0xHH).
    byte_re = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

    def surface(t: str) -> str:
        m = byte_re.match(t)
        if m:
            # decode the single byte as UTF-8 with replacement:
            # 0x00-0x7F -> the char, 0x80-0xFF -> U+FFFD (matches the original export).
            return bytes([int(m.group(1), 16)]).decode("utf-8", errors="replace")
        return t.replace("▁", " ")

    vocab = [surface(t) for t in toks]
    assert len(vocab) == VOCAB_SIZE, f"vocab length {len(vocab)} != {VOCAB_SIZE}"
    return vocab


def sanity_check(ln_w: torch.Tensor) -> None:
    lo, hi, mean = float(ln_w.min()), float(ln_w.max()), float(ln_w.mean())
    log(f"[check] ln_final_weight  min={lo:.3f}  max={hi:.3f}  mean={mean:.3f}  "
        f"(expected ~ -1.10 / 81.50 / 13.90)")
    if not (LNW_MIN_LO <= lo <= LNW_MIN_HI and LNW_MAX_LO <= hi <= LNW_MAX_HI):
        log("[check] WARNING: final-norm range is off; confirm model.norm.weight "
            "is the right tensor before trusting this export.")


def _as_load_dtype(t: torch.Tensor, load_dtype: str) -> torch.Tensor:
    """Mimic how the model was loaded at extraction time, then return float32.
    The original export loaded Gemma in bfloat16 (draft s3.1 / ledger M.2), so
    rounding the f32 safetensors through bf16 reproduces the stored W_U exactly."""
    cast = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return t.to(cast[load_dtype]).float()


def export(out_path: str, model: str, revision: str | None,
           load_dtype: str = "bfloat16") -> dict:
    weight_map, needed = plan_shards(model, revision)
    paths = download_shards(model, revision, needed)

    import time
    log(f"[read] {EMBED_KEY}  (load_dtype={load_dtype}) ...")
    t0 = time.time()
    embed = read_tensor(weight_map, paths, EMBED_KEY)  # [vocab, d_model], f32 ~4.7GB
    assert tuple(embed.shape) == (VOCAB_SIZE, D_MODEL), (
        f"embed shape {tuple(embed.shape)} != ({VOCAB_SIZE}, {D_MODEL})")
    log(f"[read]   loaded {tuple(embed.shape)} in {time.time()-t0:.0f}s; "
        f"casting to {load_dtype} + transposing (silent step, ~30-120s) ...")
    if load_dtype != "float32":
        embed = embed.to({"bfloat16": torch.bfloat16,
                          "float16": torch.float16}[load_dtype])  # frees the f32 copy
    W_U = embed.t().contiguous().float()               # [d_model, vocab], float32
    del embed
    assert tuple(W_U.shape) == (D_MODEL, VOCAB_SIZE)
    log(f"[read]   W_U ready {tuple(W_U.shape)} in {time.time()-t0:.0f}s")

    log(f"[read] {NORM_KEY} ...")
    ln_w = _as_load_dtype(read_tensor(weight_map, paths, NORM_KEY), load_dtype).contiguous()
    assert tuple(ln_w.shape) == (D_MODEL,), f"norm shape {tuple(ln_w.shape)} != ({D_MODEL},)"
    sanity_check(ln_w)

    vocab = build_vocab(model, revision)

    ingredients = {
        "W_U": W_U,
        "ln_final_weight": ln_w,
        "ln_final_eps": LN_FINAL_EPS,
        "vocab": vocab,
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "gemma_rmsnorm_note": GEMMA_RMSNORM_NOTE,
        "ln_final_type": "RMSNorm(raw google/gemma-2-27b model.norm.weight)",
        "tl_compatibility_mode": False,
        "provenance": PROVENANCE,
    }
    log(f"[save] -> {out_path}")
    torch.save(ingredients, out_path)
    log("[save] done.")
    return ingredients


def verify_against(regen: dict, ref_path: str) -> bool:
    """PASS/FAIL hinges on the numeric tensors + vocab + shape scalars.
    Metadata strings (provenance, ln_final_type) are reported but not gating,
    since the lightweight method legitimately records a different provenance."""
    log(f"[verify] loading reference {ref_path} ...")
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    ok = True

    def norm_WU(t):
        t = t.float()
        return t.t().contiguous() if tuple(t.shape) == (VOCAB_SIZE, D_MODEL) else t

    a, b = norm_WU(regen["W_U"]), norm_WU(ref["W_U"])
    if a.shape != b.shape:
        log(f"[verify] W_U shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        ok = False
    else:
        # chunked over columns to avoid a full ~4.7GB temporary
        m = 0.0
        for i in range(0, a.shape[1], 8192):
            m = max(m, (a[:, i:i+8192].float() - b[:, i:i+8192].float()).abs().max().item())
        log(f"[verify] W_U  max|diff|={m:.3e}  exact={m == 0.0}")
        ok &= m < 1e-3

    a, b = regen["ln_final_weight"].float(), ref["ln_final_weight"].float()
    d = (a - b).abs().max().item()
    log(f"[verify] ln_final_weight  max|diff|={d:.3e}  exact={torch.equal(a, b)}")
    ok &= d < 1e-3

    rv, fv = regen.get("vocab"), ref.get("vocab")
    if rv is None:
        log("[verify] vocab: SKIPPED (regenerated without transformers)")
    else:
        diffs = [i for i in range(min(len(rv), len(fv))) if rv[i] != fv[i]]
        same = (len(rv) == len(fv)) and not diffs
        log(f"[verify] vocab identical: {same}  "
            f"(len regen={len(rv)} ref={len(fv)}; diffs={len(diffs)})")
        for i in diffs[:5]:
            log(f"[verify]   e.g. id {i}: regen {rv[i]!r} vs ref {fv[i]!r}")
        ok &= same

    for k in ("ln_final_eps", "vocab_size", "d_model", "tl_compatibility_mode"):
        match = regen.get(k) == ref.get(k)
        log(f"[verify] {k}: {match}")
        ok &= match

    # informational only
    for k in ("provenance", "ln_final_type", "gemma_rmsnorm_note"):
        log(f"[verify] (info) {k} matches original: {regen.get(k) == ref.get(k)}")

    log(f"[verify] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="lens_ingredients.pt")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--revision", default=None, help="pin a model revision (optional)")
    ap.add_argument("--load-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="precision the model was loaded at (extraction used bfloat16; "
                         "the f32 safetensors are rounded through this before storing f32)")
    ap.add_argument("--verify-against", default=None,
                    help="existing lens_ingredients.pt to compare against")
    ap.add_argument("--verify-only", default=None,
                    help="skip export; load this file and verify vs --verify-against")
    ap.add_argument("--plan-only", action="store_true",
                    help="just report which shard(s) are needed, then exit")
    args = ap.parse_args()

    if args.plan_only:
        plan_shards(args.model, args.revision)
        return 0

    if args.verify_only:
        if not args.verify_against:
            log("--verify-only requires --verify-against")
            return 2
        regen = torch.load(args.verify_only, map_location="cpu", weights_only=False)
        return 0 if verify_against(regen, args.verify_against) else 1

    regen = export(args.out, args.model, args.revision, args.load_dtype)
    if args.verify_against:
        return 0 if verify_against(regen, args.verify_against) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
