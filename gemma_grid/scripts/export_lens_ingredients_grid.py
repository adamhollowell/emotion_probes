#!/usr/bin/env python3
"""Regenerate the vocabulary-readout ingredients of the five grid models from the
Gemma 2 weights, and verify them against the stored ingredients files.

The ingredients are tensors read straight from the weights, so under the Gemma
Terms of Use they ship as this recipe rather than as files. Base 27B, the anchor, is
checked here in the replication study's own schema with `--models gemma-2-27b-base`.

For each model this reads two tensors from the model's safetensors at the pinned
revision in `gemma_grid/handoff/hf_model_revisions_VERIFIED.json`:

    model.embed_tokens.weight  [256000, d_model]  ->  W_U = its transpose (Gemma 2 ties
                                                      its input and output embeddings)
    model.norm.weight          [d_model]          ->  ln_final_w, applied in the (1 + w)
                                                      form at readout

It fetches only the bytes of those two tensors with HTTP range requests, falling
back to downloading the whole shard if the server refuses a range. Both tensors are
cast to bfloat16, the precision the models were loaded in at extraction, and the
unembedding bias b_U is zero (Gemma 2 has none). The result is compared tensor by
tensor with the stored `<model>__lens_ingredients.pt`, and a report is written.

Needs: torch, safetensors, huggingface_hub, a Hugging Face login with the Gemma
license accepted for each model.

    python gemma_grid/scripts/export_lens_ingredients_grid.py --verify
    python gemma_grid/scripts/export_lens_ingredients_grid.py --models gemma-2-2b-base --out-dir regen/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
PINS = REPO / "gemma_grid/handoff/hf_model_revisions_VERIFIED.json"
REPORT = REPO / "gemma_grid/geometry_out/lens_export_verification.json"
MODELS = {
    "gemma-2-2b-base": "google/gemma-2-2b",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "gemma-2-9b-base": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-2-27b-it": "google/gemma-2-27b-it",
    "gemma-2-27b-base": "google/gemma-2-27b",
}
ANCHOR = "gemma-2-27b-base"
E1_EXPORTER = REPO / "scripts/export_lens_ingredients.py"  # the replication study's exporter, sha256 1ff18010...
EMBED, NORM = "model.embed_tokens.weight", "model.norm.weight"
VOCAB = 256000
DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}


def log(msg: str) -> None:
    print(msg, flush=True)


def stored_path(cell: str) -> Path:
    if cell == ANCHOR:
        return REPO / "data/vectors/lens_ingredients.pt"
    return REPO / f"gemma_grid/data/{cell}/vectors/{cell}__lens_ingredients.pt"


def regenerate_anchor(repo_id: str, rev: str) -> dict:
    """Base 27B in the replication study's schema: float32 tensors rounded through
    bfloat16, the load precision, plus the vocabulary map from that study's exporter."""
    cfg = config(repo_id, rev)
    d_model = cfg["hidden_size"]
    wm = weight_map(repo_id, rev)
    embed = read_tensor(repo_id, rev, wm, EMBED)
    assert tuple(embed.shape) == (VOCAB, d_model)
    bf = embed.to(torch.bfloat16)
    del embed
    W_U = bf.t().float().contiguous()
    del bf
    ln = read_tensor(repo_id, rev, wm, NORM).to(torch.bfloat16).float().contiguous()
    vocab = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("e1_export", E1_EXPORTER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        vocab = mod.build_vocab(repo_id, rev)
    except Exception as exc:
        log(f"  [vocab] not rebuilt ({exc})")
    return {"W_U": W_U, "ln_final_weight": ln, "ln_final_eps": 1e-6, "vocab": vocab,
            "vocab_size": VOCAB, "d_model": d_model}


def compare_anchor(regen: dict, ref_path: Path) -> dict:
    ref = torch.load(ref_path, map_location="cpu", weights_only=False, mmap=True)
    out = {"tensors": {}, "scalars": {}}
    ok = True
    for k in ("W_U", "ln_final_weight"):
        a, b = regen[k], ref[k]
        row = {"shape_regen": list(a.shape), "shape_stored": list(b.shape), "dtype_stored": str(b.dtype)}
        if tuple(a.shape) != tuple(b.shape):
            row.update(equal=False, max_abs_diff=None)
        else:
            row["equal"] = bool(torch.equal(a, b.to(a.dtype)))
            row["max_abs_diff"] = 0.0 if row["equal"] else max_abs_diff(a, b)
        ok &= row["equal"]
        out["tensors"][k] = row
    for k in ("ln_final_eps", "vocab_size", "d_model"):
        rv, sv = regen[k], ref.get(k)
        match = str(rv) == str(sv)
        out["scalars"][k] = {"regen": str(rv), "stored": str(sv), "match": bool(match)}
        ok &= match
    rv, fv = regen.get("vocab"), ref.get("vocab")
    if rv is None:
        out["vocab"] = {"checked": False}
    else:
        diffs = [i for i in range(VOCAB) if rv[i] != fv[i]]
        out["vocab"] = {"checked": True, "identical": not diffs, "n_diffs": len(diffs)}
        ok &= not diffs
    out["pass"] = bool(ok)
    return out


def weight_map(repo_id: str, rev: str) -> dict:
    from huggingface_hub import hf_hub_download
    try:
        idx = hf_hub_download(repo_id, "model.safetensors.index.json", revision=rev)
        return json.load(open(idx))["weight_map"]
    except Exception:
        return {EMBED: "model.safetensors", NORM: "model.safetensors"}


def config(repo_id: str, rev: str) -> dict:
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(repo_id, "config.json", revision=rev)))


def _get(url: str, headers: dict, start: int, end: int) -> bytearray:
    """Fetch bytes [start, end] of url. Works with the httpx session of
    huggingface_hub 1.x and the requests session of 0.x."""
    from huggingface_hub.utils import get_session
    h = dict(headers, Range=f"bytes={start}-{end}")
    sess = get_session()
    buf = bytearray()
    if hasattr(sess, "stream"):  # httpx
        with sess.stream("GET", url, headers=h, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            if r.status_code != 206:
                raise RuntimeError("server ignored the range request")
            for chunk in r.iter_bytes(chunk_size=1 << 24):
                buf.extend(chunk)
    else:  # requests
        r = sess.get(url, headers=h, allow_redirects=True, timeout=600, stream=True)
        r.raise_for_status()
        if r.status_code != 206:
            r.close()
            raise RuntimeError("server ignored the range request")
        for chunk in r.iter_content(chunk_size=1 << 24):
            buf.extend(chunk)
    want = end - start + 1
    if len(buf) != want:
        raise RuntimeError(f"short read: {len(buf)} of {want} bytes")
    return buf


def read_tensor_ranged(repo_id: str, rev: str, shard: str, key: str) -> torch.Tensor:
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    url = hf_hub_url(repo_id, shard, revision=rev)
    headers = build_hf_headers()
    n = int.from_bytes(_get(url, headers, 0, 7), "little")
    header = json.loads(bytes(_get(url, headers, 8, 8 + n - 1)))
    meta = header[key]
    s, e = meta["data_offsets"]
    base = 8 + n
    log(f"  [range] {key}: {meta['dtype']} {meta['shape']}, {(e - s) / 1e9:.2f} GB")
    raw = _get(url, headers, base + s, base + e - 1)
    t = torch.frombuffer(raw, dtype=DTYPES[meta["dtype"]])
    return t.reshape(meta["shape"])


def read_tensor_full(repo_id: str, rev: str, shard: str, key: str) -> torch.Tensor:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    log(f"  [download] whole shard {shard} for {key}")
    path = hf_hub_download(repo_id, shard, revision=rev)
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def read_tensor(repo_id: str, rev: str, wm: dict, key: str) -> torch.Tensor:
    try:
        return read_tensor_ranged(repo_id, rev, wm[key], key)
    except Exception as exc:
        log(f"  [range] unavailable ({exc}); falling back to a full shard download")
        return read_tensor_full(repo_id, rev, wm[key], key)


def regenerate(cell: str, repo_id: str, rev: str) -> dict:
    cfg = config(repo_id, rev)
    d_model, n_layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    wm = weight_map(repo_id, rev)
    embed = read_tensor(repo_id, rev, wm, EMBED)
    assert tuple(embed.shape) == (VOCAB, d_model), f"{cell}: embed {tuple(embed.shape)}"
    W_U = embed.to(torch.bfloat16).t().contiguous()
    del embed
    ln = read_tensor(repo_id, rev, wm, NORM).to(torch.bfloat16).contiguous()
    assert tuple(ln.shape) == (d_model,), f"{cell}: norm {tuple(ln.shape)}"
    return {
        "W_U": W_U,
        "b_U": torch.zeros(VOCAB, dtype=torch.bfloat16),
        "ln_final_w": ln,
        "dtype": torch.bfloat16,
        "n_layers": n_layers,
        "d_model": d_model,
        "note": "apply Gemma-2 final RMSNorm in (1+w) form at lens readout",
        "provenance": f"regenerated from {repo_id}@{rev} safetensors by export_lens_ingredients_grid.py",
    }


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    m = 0.0
    a2, b2 = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    for i in range(0, a2.shape[1], 8192):
        m = max(m, (a2[:, i:i + 8192].float() - b2[:, i:i + 8192].float()).abs().max().item())
    return m


def compare(regen: dict, ref_path: Path) -> dict:
    ref = torch.load(ref_path, map_location="cpu", weights_only=False, mmap=True)
    out = {"tensors": {}, "scalars": {}}
    ok = True
    for k in ("W_U", "b_U", "ln_final_w"):
        a, b = regen[k], ref[k]
        row = {"shape_regen": list(a.shape), "shape_stored": list(b.shape),
               "dtype_stored": str(b.dtype)}
        if tuple(a.shape) != tuple(b.shape):
            row.update(equal=False, max_abs_diff=None)
        else:
            row["equal"] = bool(torch.equal(a, b.to(a.dtype)))
            row["max_abs_diff"] = 0.0 if row["equal"] else max_abs_diff(a, b)
        ok &= row["equal"]
        out["tensors"][k] = row
    for k in ("d_model", "n_layers", "dtype"):
        rv, sv = regen[k], ref.get(k)
        match = str(rv) == str(sv)
        out["scalars"][k] = {"regen": str(rv), "stored": str(sv), "match": bool(match)}
        ok &= match
    out["pass"] = bool(ok)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[m for m in MODELS if m != ANCHOR], choices=list(MODELS))
    ap.add_argument("--verify", action="store_true",
                    help="compare against the stored ingredients and write the report")
    ap.add_argument("--out-dir", default=None, help="also save each regenerated file here")
    args = ap.parse_args()
    if not args.verify and not args.out_dir:
        ap.error("give --verify, --out-dir, or both")

    pins = json.load(open(PINS))
    report = json.loads(REPORT.read_text()) if (args.verify and REPORT.exists()) else {}
    report.setdefault("record", "Regenerated vocabulary-readout ingredients against the stored files, per model.")
    report.setdefault("pins_file_sha256_prefix", None)
    report.setdefault("results", {})
    import hashlib
    report["pins_file_sha256_prefix"] = hashlib.sha256(PINS.read_bytes()).hexdigest()[:16]

    all_ok = True
    for cell in args.models:
        repo_id = MODELS[cell]
        rev = pins[repo_id]["current_main_sha"]
        log(f"[{cell}] {repo_id} @ {rev}")
        t0 = time.time()
        regen = regenerate_anchor(repo_id, rev) if cell == ANCHOR else regenerate(cell, repo_id, rev)
        if args.out_dir:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(regen, out / f"{cell}__lens_ingredients.pt")
            log(f"  [save] {out / f'{cell}__lens_ingredients.pt'}")
        if args.verify:
            ref = stored_path(cell)
            res = compare_anchor(regen, ref) if cell == ANCHOR else compare(regen, ref)
            res.update(revision=rev, stored_file=str(ref.relative_to(REPO)),
                       seconds=round(time.time() - t0, 1),
                       run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            report["results"][cell] = res
            REPORT.write_text(json.dumps(report, indent=2))
            for k, v in res["tensors"].items():
                log(f"  {k}: equal={v['equal']} max|diff|={v['max_abs_diff']}")
            for k, v in res["scalars"].items():
                log(f"  {k}: match={v['match']}")
            if "vocab" in res:
                log(f"  vocab: {res['vocab']}")
            log(f"  {'PASS' if res['pass'] else 'FAIL'}")
            all_ok &= res["pass"]
        del regen
    if args.verify:
        log(f"[report] {REPORT}")
        log("ALL PASS" if all_ok else "AT LEAST ONE FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
