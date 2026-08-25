#!/usr/bin/env python3
"""
maxact_sweep_v2.py -- max-activation sweep over the rebuilt held-out
corpus. Conventions match
emotion_probes/validation.py::top_activating_examples and the M.2-verified
numerics (TransformerBridge, tl_compatibility_mode=False, bf16 weights,
resid_post cast to float32 before projection).

Three improvements over the original by design:
  1. Per-token projection arrays saved for every winning document (pass 2)
     -> continuous-shading Figure 1, any window width, GPU-free forever.
  2. doc_id + source first-class in every record; corpus is checksummed and
     durable -> windows resolve to source documents forever.
  3. Per-emotion activation histograms over ALL corpus tokens (pass 1)
     -> the paper's "90th percentile on the dataset" threshold, either
     per-emotion or pooled convention, derivable exactly.

Pass 1: stream all docs, project resid_post[L31] onto 171 cleaned vectors,
        keep top-K (doc_id, pos, value) per emotion in heaps, accumulate
        histograms. Checkpoints every --checkpoint-every docs; resumable.
Pass 2: re-run only the deduplicated winning docs; save per-doc npz with
        token ids, token strings, and full (seq x 171) float16 projections;
        emit maxactivating_v2.json (old-schema-compatible + new fields).

Usage (from project root, venv active):
    python scripts/maxact_sweep_v2.py --smoke          # 200 docs, sanity gate
    python scripts/maxact_sweep_v2.py                  # full run (resumable)
    python scripts/maxact_sweep_v2.py --pass2-only     # if pass 1 already done
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

# ---- fixed run parameters (Decisions locked 2026-07-06; do not edit casually) ----
MODEL_NAME = "google/gemma-2-27b"
LAYER = 31
TOPK = 20
CAP = 2048                 # Decision 3A (provisional; confirmed at token-manifest step)
SKIP_FIRST = 1             # BOS skip, matching validation.py default
HIST_LO, HIST_HI, HIST_BINS = -8000.0, 8000.0, 8000
VECTORS_SHA256_EXPECTED = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_corpus(path: Path, limit: int | None = None) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
            if limit and len(docs) >= limit:
                break
    return docs


def load_vectors(path: Path, verify_hash: bool = True):
    if verify_hash:
        digest = sha256_file(path)
        if digest != VECTORS_SHA256_EXPECTED:
            raise SystemExit(
                "FATAL: emotion_vectors_171.pt sha256 mismatch.\n  expected %s\n  got      %s\n"
                "This is not the canonical artifact. Stop and investigate."
                % (VECTORS_SHA256_EXPECTED, digest))
        print("emotion_vectors_171.pt sha256 verified: %s" % digest[:16])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    emotions = list(payload["emotions"])
    V = payload["vectors"][:, LAYER, :].float()  # cleaned set, (171, d_model)
    assert V.shape[0] == len(emotions) == 171, V.shape
    # UNIT-NORMALIZE (v2.1 fix): project onto v_hat = v/||v||, matching the
    # original-run units (verified against maxactivating_layer31.json scale).
    # Monotonic per emotion (same winners/positions); makes values comparable
    # across emotions, which pooled percentiles and Figure-1 shading require.
    V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return emotions, V


def boot_model(device="cuda", dtype=torch.bfloat16):
    from transformer_lens.model_bridge import TransformerBridge
    print("Booting %s (bf16, tl_compatibility_mode=False)..." % MODEL_NAME)
    t0 = time.time()
    bridge = TransformerBridge.boot_transformers(MODEL_NAME, device=device, dtype=dtype)
    # NOTE: compatibility mode deliberately NOT enabled (M.2 numerics regime).
    bridge.eval()
    print("  booted in %.0fs (n_layers=%d, d_model=%d)"
          % (time.time() - t0, bridge.cfg.n_layers, bridge.cfg.d_model))
    return bridge


_STOP_AT_LAYER_OK = None  # tri-state: None=untested, True/False after first doc


def resid_at_layer(model, tokens, hook_name):
    """Forward through the model, return resid_post[LAYER] as (seq, d_model) f32 cpu.
    Tries stop_at_layer (saves ~1/3 compute, mathematically identical for L31);
    falls back to full forward if the bridge doesn't support it."""
    global _STOP_AT_LAYER_OK
    with torch.no_grad():
        if _STOP_AT_LAYER_OK is not False:
            try:
                _, cache = model.run_with_cache(
                    tokens, names_filter=[hook_name], stop_at_layer=LAYER + 1)
                if _STOP_AT_LAYER_OK is None:
                    _STOP_AT_LAYER_OK = True
                    print("  stop_at_layer supported: running truncated forwards (disclosed)")
                return cache[hook_name][0].float().cpu()
            except TypeError:
                _STOP_AT_LAYER_OK = False
                print("  stop_at_layer unsupported: full forwards (slower, same numbers)")
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        return cache[hook_name][0].float().cpu()


def pass1(model, docs, emotions, V, out_dir: Path, checkpoint_every: int):
    hook_name = "blocks.%d.hook_resid_post" % LAYER
    ckpt_path = out_dir / "checkpoint_pass1.pt"
    n_emo = len(emotions)

    # ---- resume or init state ----
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert state["n_docs_total"] == len(docs), "checkpoint is for a different corpus"
        heaps = state["heaps"]
        hist = state["hist"]
        tokens_per_doc = state["tokens_per_doc"]
        start_idx = state["next_doc_idx"]
        tiebreak = itertools.count(state["tiebreak"])
        print("RESUME: pass 1 from doc %d / %d" % (start_idx, len(docs)))
    else:
        heaps = [[] for _ in range(n_emo)]  # entries: (value, tie, doc_id, pos)
        hist = torch.zeros(n_emo, HIST_BINS, dtype=torch.int64)
        tokens_per_doc = {}
        start_idx = 0
        tiebreak = itertools.count(0)

    bin_w = (HIST_HI - HIST_LO) / HIST_BINS
    t0 = time.time()
    done_since = 0

    for idx in range(start_idx, len(docs)):
        doc = docs[idx]
        tokens = model.to_tokens(doc["text"])[:, :CAP]
        resid = resid_at_layer(model, tokens, hook_name)          # (seq, d)
        seq_len = resid.shape[0]
        tokens_per_doc[doc["doc_id"]] = int(seq_len)
        start = min(SKIP_FIRST, max(0, seq_len - 1))
        proj = resid[start:] @ V.T                                 # (seq-1, 171)

        # histograms (clamped into range; overflow lands in edge bins, tracked via min/max)
        bidx = ((proj - HIST_LO) / bin_w).long().clamp_(0, HIST_BINS - 1)
        flat = (bidx + torch.arange(n_emo).view(1, -1) * HIST_BINS).flatten()
        hist += torch.bincount(flat, minlength=n_emo * HIST_BINS).view(n_emo, HIST_BINS)

        # heaps
        n_pick = min(TOPK, proj.shape[0])
        if n_pick > 0:
            top_v, top_p = proj.topk(n_pick, dim=0)                # (n_pick, 171)
            for e in range(n_emo):
                h = heaps[e]
                for r in range(n_pick):
                    val = float(top_v[r, e])
                    if len(h) < TOPK:
                        heapq.heappush(h, (val, next(tiebreak), doc["doc_id"], int(top_p[r, e]) + start))
                    elif val > h[0][0]:
                        heapq.heappushpop(h, (val, next(tiebreak), doc["doc_id"], int(top_p[r, e]) + start))
                    else:
                        break  # top_v is sorted descending per column

        done_since += 1
        if done_since % 100 == 0:
            rate = done_since / max(time.time() - t0, 1e-9)
            eta_h = (len(docs) - idx - 1) / max(rate, 1e-9) / 3600
            print("  doc %6d/%d  (%.2f docs/s, ETA %.1f h)" % (idx + 1, len(docs), rate, eta_h))
        if (idx + 1) % checkpoint_every == 0 or idx == len(docs) - 1:
            tmp = ckpt_path.with_suffix(".pt.tmp")
            torch.save({"heaps": heaps, "hist": hist, "tokens_per_doc": tokens_per_doc,
                        "next_doc_idx": idx + 1, "n_docs_total": len(docs),
                        "tiebreak": next(tiebreak)}, tmp)
            tmp.rename(ckpt_path)

    # ---- write pass-1 outputs ----
    by_emotion = {}
    for e, emo in enumerate(emotions):
        entries = sorted(heaps[e], key=lambda x: -x[0])
        by_emotion[emo] = [{"value": v, "doc_id": d, "pos": p} for (v, _t, d, p) in entries]

    results = {
        "layer": LAYER, "n_docs": len(docs), "topk": TOPK, "cap": CAP,
        "skip_first_n_positions": SKIP_FIRST,
        "model": MODEL_NAME, "numerics": "TransformerBridge bf16, tl_compatibility_mode=False, f32 projection",
        "stop_at_layer_used": bool(_STOP_AT_LAYER_OK),
        "vectors_sha256": VECTORS_SHA256_EXPECTED,
        "vector_set": "cleaned",
        "projection": "resid_post[L31] (f32) dot unit-normalized vector (v/||v||); June-comparable units (script v2.1)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "by_emotion": by_emotion,
        "tokens_per_doc_observed": tokens_per_doc,  # cross-check vs token_manifest_v2
    }
    p1 = out_dir / "pass1_results.json"
    tmp = p1.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=1))
    tmp.rename(p1)
    print("wrote %s" % p1)

    np.savez_compressed(out_dir / "histograms.npz",
                        counts=hist.numpy(), lo=HIST_LO, hi=HIST_HI, bins=HIST_BINS,
                        emotions=np.array(emotions, dtype=object))
    print("wrote %s" % (out_dir / "histograms.npz"))
    return by_emotion


def pass2(model, docs, emotions, V, by_emotion, out_dir: Path):
    hook_name = "blocks.%d.hook_resid_post" % LAYER
    doc_index = {d["doc_id"]: d for d in docs}
    winners = sorted({w["doc_id"] for wins in by_emotion.values() for w in wins})
    print("pass 2: %d unique winning documents" % len(winners))
    pt_dir = out_dir / "pertoken"
    pt_dir.mkdir(parents=True, exist_ok=True)

    for i, doc_id in enumerate(winners):
        npz_path = pt_dir / ("%s.npz" % doc_id)
        if npz_path.exists():
            continue
        doc = doc_index[doc_id]
        tokens = model.to_tokens(doc["text"])[:, :CAP]
        resid = resid_at_layer(model, tokens, hook_name)
        proj = (resid @ V.T).to(torch.float16).numpy()             # (seq, 171), incl. BOS row
        ids = tokens[0].cpu().numpy()
        toks = [model.tokenizer.decode([int(t)]) for t in ids]
        np.savez_compressed(npz_path, token_ids=ids,
                            tokens=np.array(toks, dtype=object),
                            proj=proj, source=doc["source"], doc_id=doc_id,
                            layer=LAYER, skip_first=SKIP_FIRST)
        if (i + 1) % 100 == 0:
            print("  %d/%d per-token files" % (i + 1, len(winners)))

    # ---- old-schema-compatible windows file, extended ----
    half = 6  # 13-token window, matching maxactivating_layer31.json
    by_emotion_windows = {}
    for emo, wins in by_emotion.items():
        rows = []
        for w in wins:
            z = np.load(pt_dir / ("%s.npz" % w["doc_id"]), allow_pickle=True)
            toks = list(z["tokens"])
            proj = z["proj"]
            e_idx = emotions.index(emo)
            pos = w["pos"]
            lo, hi = max(0, pos - half), min(len(toks), pos + half + 1)
            rows.append({
                "value": w["value"],
                "peak_pos": pos,
                "window": toks[lo:hi],
                "peak_token": toks[pos],
                "proj_window": [float(x) for x in proj[lo:hi, e_idx]],
                "doc_id": w["doc_id"],
                "source": str(z["source"]),
                "story_emotion": None,
                "topic": str(z["source"]),  # legacy field kept for side-by-side diffing
            })
        by_emotion_windows[emo] = rows

    out = {"layer": LAYER, "n_stories": len(docs), "topk": TOPK,
           "schema": "v2: old fields + doc_id/source/proj_window; see pass1_results.json for config",
           "by_emotion": by_emotion_windows}
    p2 = out_dir / "maxactivating_v2.json"
    tmp = p2.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.rename(p2)
    print("wrote %s" % p2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/heldout/heldout_corpus_v2.jsonl"))
    ap.add_argument("--vectors", type=Path, default=Path("data/vectors/emotion_vectors_171.pt"))
    ap.add_argument("--out", type=Path, default=Path("data/heldout/maxact"))
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true",
                    help="200 docs only, output to <out>_smoke; verifies schema end-to-end")
    ap.add_argument("--pass2-only", action="store_true")
    ap.add_argument("--no-hash-check", action="store_true",
                    help="skip vectors sha256 gate (never use on the real run)")
    args = ap.parse_args()

    out_dir = Path(str(args.out) + "_smoke") if args.smoke else args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    emotions, V = load_vectors(args.vectors, verify_hash=not args.no_hash_check)
    docs = load_corpus(args.corpus, limit=200 if args.smoke else None)
    print("corpus: %d docs%s" % (len(docs), " (SMOKE)" if args.smoke else ""))

    model = boot_model()

    if args.pass2_only:
        by_emotion = json.loads((out_dir / "pass1_results.json").read_text())["by_emotion"]
    else:
        by_emotion = pass1(model, docs, emotions, V, out_dir,
                           checkpoint_every=args.checkpoint_every)
    pass2(model, docs, emotions, V, by_emotion, out_dir)
    print("\nSweep complete. Next: sync and verify outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
