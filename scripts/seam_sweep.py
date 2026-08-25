#!/usr/bin/env python
"""
seam_sweep.py — per-layer logit-lens readability sweep for the Fig-9 two-band test.

Question: does emotion self-readability stay low/noisy through the early-mid band
and climb sharply across the seam (~22-24) into the late plateau? A rise across the
seam supports the re-encoding reading of the two-band cross-layer structure.

Two metrics, both applied uniformly across all 46 layers:
  - self_rank   : rank of the emotion's OWN word-form in the layer's logit ranking
                  (0 = top token). Continuous, judgment-free. Lower = more readable.
  - on_theme    : strict lexical/stem match — does top-k contain the emotion's word
                  family. Reproducible, no external lexicon. This reads BELOW the
                  manual 167/171 at L38 by design (it cannot credit semantic-only
                  congruence like compassionate->help); the L38 value is printed as
                  an anchor so you can see the gap.

Projection recipe matches the validated L38 gate exactly:
  x.float() -> RMSNorm(x; (1+w), eps) -> @ W_U   (no soft-cap; soft-cap is monotonic
  and does not change token ranking).
"""
# --- basis gate: refuse to run unless the vectors match the expected sha256.
# Identify the vectors by their hash, not the filename `emotion_vectors_171.pt`. ---
import hashlib as _hlg, pathlib as _plg
_VECG = _plg.Path("data/vectors/emotion_vectors_171.pt")
_EXPG = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
if _VECG.exists():
    _hg = _hlg.sha256(_VECG.read_bytes()).hexdigest()
    assert _hg == _EXPG, (
        "%s basis mismatch: %s != expected canonical %s -- refusing non-canonical basis"
        % (_VECG, _hg[:12], _EXPG[:12]))
# --- end basis gate ---

import argparse, csv, os, sys
import torch


def build_vocab_maps(vocab):
    """Return (id2str list-like, str2id dict). Accepts list or dict id->str."""
    if isinstance(vocab, dict):
        # keys may be int or str
        items = sorted(((int(k), val) for k, val in vocab.items()), key=lambda kv: kv[0])
        id2str = [val for _, val in items]
    else:
        id2str = list(vocab)
    str2id = {}
    for i, s in enumerate(id2str):
        # first occurrence wins
        if s not in str2id:
            str2id[s] = i
    return id2str, str2id


def lens_logits(resid, w, eps, W_U):
    """resid: (..., d_model). Returns logits (..., vocab)."""
    x = resid.float()
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    normed = (x / rms * (1.0 + w)).to(W_U.dtype)
    return normed @ W_U


def self_target_ids(word, str2id):
    """Vocab ids for surface forms of the emotion word (space-prefixed + caps)."""
    forms = set()
    for base in (word, word.capitalize()):
        forms.add(base)
        forms.add(" " + base)
    ids = [str2id[f] for f in forms if f in str2id]
    return ids


def congruent_stem(word):
    """Stem for strict on-theme matching: first <=5 chars, lowercased."""
    return word.lower()[:5]


def is_on_theme(topk_strs, stem):
    for s in topk_strs:
        t = s.strip().lower()
        if len(t) >= len(stem) and t.startswith(stem):
            return True
    return False


def run_sweep(v, g, variant, topk):
    emotions = list(v["emotions"])
    arr = v["vectors_raw"] if variant == "raw" else v["vectors"]
    n_emo, n_layers, d_model = arr.shape
    W_U = g["W_U"]
    if W_U.shape[0] != d_model:  # accept transposed storage
        W_U = W_U.t().contiguous()
    w = g["ln_final_weight"].float()
    eps = float(g["ln_final_eps"])
    id2str, str2id = build_vocab_maps(g["vocab"])

    targets = [self_target_ids(e, str2id) for e in emotions]
    stems = [congruent_stem(e) for e in emotions]
    n_self_defined = sum(1 for t in targets if t)

    rows = []
    for L in range(n_layers):
        logits = lens_logits(arr[:, L, :], w, eps, W_U)  # (n_emo, vocab)
        # self_rank per emotion = min over target ids of (#tokens scoring higher)
        self_ranks = []
        for i, tids in enumerate(targets):
            if not tids:
                self_ranks.append(float("nan"))
                continue
            li = logits[i]
            best = min(int((li > li[t]).sum().item()) for t in tids)
            self_ranks.append(best)
        # on-theme via top-k
        _, idx = logits.topk(topk, dim=-1)  # (n_emo, topk)
        on = 0
        for i in range(n_emo):
            strs = [id2str[j] for j in idx[i].tolist()]
            on += int(is_on_theme(strs, stems[i]))
        valid = [r for r in self_ranks if r == r]  # drop nan
        valid_sorted = sorted(valid)
        median = valid_sorted[len(valid_sorted) // 2] if valid_sorted else float("nan")
        frac_top10 = (sum(1 for r in valid if r < 10) / len(valid)) if valid else float("nan")
        rows.append({
            "layer": L,
            "median_self_rank": median,
            "frac_self_top10": round(frac_top10, 4),
            "on_theme_count": on,
            "on_theme_rate": round(on / n_emo, 4),
        })
    return rows, n_emo, n_self_defined


def synthetic_inputs():
    """Build tiny synthetic v/g with a PLANTED seam to validate the metric responds.
    Each planted emotion's OWN word token is in vocab, and vocab >> top-k so top-k is
    selective. Before the seam: noise. After: own word dominates -> both metrics move."""
    emotions = ["anger", "blissful", "afraid", "amazed", "atease"]
    own_words = [" anger", " blissful", " afraid", " amazed"]
    distractors = [f" d{i}" for i in range(200)]
    vocab = own_words + distractors  # V = 24, top-k default 12 stays selective
    str2id = {s: i for i, s in enumerate(vocab)}
    d_model, n_layers, V = 16, 46, len(vocab)
    torch.manual_seed(0)
    W_U = torch.randn(d_model, V)
    arr = torch.randn(len(emotions), n_layers, d_model) * 0.1
    # plant: each emotion's OWN word becomes dominant only from ~layer 24 onward,
    # ramping across 20..24 (the seam), absent before.
    own = {"anger": " anger", "blissful": " blissful", "afraid": " afraid", "amazed": " amazed"}
    for ei, e in enumerate(emotions):
        if e not in own:
            continue
        col = W_U[:, str2id[own[e]]]
        for L in range(n_layers):
            strength = max(0.0, min(1.0, (L - 20) / 4.0))  # 0 before 20, 1 by 24
            arr[ei, L] = arr[ei, L] + strength * 6.0 * col
    v = {"emotions": emotions, "vectors_raw": arr, "vectors": arr.clone(),
         "n_layers": n_layers, "d_model": d_model, "variant": "synthetic"}
    g = {"W_U": W_U, "ln_final_weight": torch.ones(d_model), "ln_final_eps": 1e-6,
         "ln_final_type": "synthetic", "vocab": vocab}
    return v, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors-dir", default="data/vectors")
    ap.add_argument("--variant", choices=["raw", "cleaned"], default="raw")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--out", default="seam_sweep.csv")
    ap.add_argument("--self-test", action="store_true",
                    help="run on planted-seam synthetic data and assert recovery")
    args = ap.parse_args()

    if args.self_test:
        v, g = synthetic_inputs()
        rows, n_emo, n_self = run_sweep(v, g, "raw", args.topk)
        early = [r for r in rows if r["layer"] < 18]
        late = [r for r in rows if r["layer"] >= 26]
        early_rate = sum(r["on_theme_rate"] for r in early) / len(early)
        late_rate = sum(r["on_theme_rate"] for r in late) / len(late)
        early_rank = sum(r["median_self_rank"] for r in early) / len(early)
        late_rank = sum(r["median_self_rank"] for r in late) / len(late)
        print(f"[self-test] early on_theme_rate={early_rate:.3f}  late={late_rate:.3f}")
        print(f"[self-test] early median_self_rank={early_rank:.1f}  late={late_rank:.1f}")
        ok = late_rate > early_rate + 0.5 and late_rank < early_rank - 2
        print("[self-test] seam recovered:", ok)
        # show the ramp around the planted seam
        for r in rows:
            if 18 <= r["layer"] <= 26:
                print("   ", r)
        sys.exit(0 if ok else 1)

    D = args.vectors_dir
    v = torch.load(os.path.join(D, "emotion_vectors_171.pt"), map_location="cpu")
    g = torch.load(os.path.join(D, "lens_ingredients.pt"), map_location="cpu")
    rows, n_emo, n_self = run_sweep(v, g, args.variant, args.topk)

    with open(args.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"variant={args.variant}  emotions={n_emo}  self_rank defined for {n_self}/{n_emo}")
    print(f"{'L':>3} {'med_self_rank':>13} {'frac_top10':>10} {'on_theme':>9} {'rate':>6}")
    for r in rows:
        print(f"{r['layer']:>3} {r['median_self_rank']:>13} "
              f"{r['frac_self_top10']:>10} {r['on_theme_count']:>9} {r['on_theme_rate']:>6}")
    l38 = next((r for r in rows if r["layer"] == 38), None)
    if l38:
        print(f"\nL38 on_theme_count={l38['on_theme_count']}/{n_emo} "
              f"(strict lexical; manual semantic count was 167 — gap expected)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
