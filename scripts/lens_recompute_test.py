#!/usr/bin/env python3
"""
lens_recompute_test.py  (v2, reconciled to the numerics gate)
-------------------------------------------------------------
Decides Table 1's layer. The hinge question: was the original "loving is junk
at L31" a MISNORMALIZATION artifact (raw v@W_U, no ln_final) or a GENUINE
property that survives correct numerics? Everything about the faithful-layer
choice follows from that.

Numerics (confirmed from lens_ingredients.pt / the recipe that built
lens_final_fixed.pt):
    rms    = sqrt(mean(v^2) + eps),  eps = 1e-6      # SCALAR per (emotion,layer)
    normed = (v / rms) * (1 + ln_w)                   # Gemma (1+w) gain, applied ONCE
    logits = normed @ W_U                             # W_U is RAW (ln_final NOT folded)

Note on 1/rms: it is a single positive scalar, so it does NOT change the
top/bottom-k token RANKING -- it only rescales logit VALUES. It is included
here so values reproduce lens_final_fixed.pt exactly (and so the self-check
below is meaningful), not because the token lists depend on it.

Three readouts, to disaggregate the cause of any junk:
    (i)   raw     :  v @ W_U                          # reproduces the OLD buggy lens
    (ii)  correct :  (v/rms)*(1+w) @ W_U              # the canonical readout
    (iii) cosine  :  [(v/rms)*(1+w) @ W_U] / ||W_U row||   # disclosed token-norm variant

Runs LOCALLY in lens_env (needs emotion_vectors_171.pt + the 4.7GB lens_ingredients.pt).
CPU fine.
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

import torch

# ---- edit these three paths ----
VEC_PATH  = "data/vectors/emotion_vectors_171.pt"
ING_PATH  = "data/vectors/lens_ingredients.pt"
LENS_PATH = "data/vectors/lens_final_fixed.pt"   # for the self-validation; set to None to skip
# --------------------------------
EPS    = 1e-6
TOPK   = 12
TWELVE = ['happy','inspired','loving','proud','calm','desperate',
          'angry','guilty','sad','afraid','nervous','surprised']
LOVING_LAYERS = [31, 38, 43]   # 31 = faithful (2/3 depth); 38 = current artifact; 43 = where loving reportedly resolves
ALL12_LAYER   = 31             # the Anthropic-faithful layer to test for whole-table legibility

def disp(t):
    return t.lstrip(' ') if t.strip() else t

# ------------------------------------------------------------------ load lens ----
ing = torch.load(ING_PATH, map_location="cpu", weights_only=False)
d_model = int(ing["d_model"]); Vsz = int(ing["vocab_size"])
W_U = ing["W_U"].float()
if W_U.shape == (d_model, Vsz):
    W_U = W_U.t().contiguous()
assert W_U.shape == (Vsz, d_model), f"W_U {tuple(W_U.shape)} != ({Vsz},{d_model})"
ln_w     = ing["ln_final_weight"].float()      # [d_model]
gain     = 1.0 + ln_w
tok_norm = W_U.norm(dim=1).clamp_min(1e-8)     # per-token unembedding norm
vocab    = ing["vocab"]
def tokstr(i): return vocab[i]

print("=== diagnostics ===")
print(f"W_U {tuple(W_U.shape)}  d_model={d_model}  vocab={Vsz}")
print(f"gain(1+w): min={gain.min():.3f} max={gain.max():.3f} mean={gain.mean():.3f}")
print(f"tok_norm pcts [.5,.9,.99,.999,1.0]: "
      + ", ".join(f"{torch.quantile(tok_norm,q):.2f}" for q in (.5,.9,.99,.999,1.0)))
print()

# --------------------------------------------------------------- load vectors ----
vb   = torch.load(VEC_PATH, map_location="cpu", weights_only=False)
emos = vb["emotions"]
def vec(emo, layer, kind="clean"):
    arr = vb["vectors" if kind == "clean" else "vectors_raw"]
    return arr[emos.index(emo), layer, :].float()

def readout(v, mode):
    if mode == "raw":
        g = v
    else:                                   # correct / cosine share the full RMSNorm
        rms = torch.sqrt(v.pow(2).mean() + EPS)
        g = (v / rms) * gain
    logits = W_U @ g
    if mode == "cosine":
        logits = logits / tok_norm
    return logits

def topbot(v, mode, k=TOPK):
    lg = readout(v, mode)
    top = [disp(tokstr(i)) for i in torch.topk(lg,  k).indices.tolist()]
    bot = [disp(tokstr(i)) for i in torch.topk(-lg, k).indices.tolist()]
    return top, bot

# ----------------------------------------------- SELF-VALIDATION (trust gate) ----
# Reproduce lens_final_fixed.pt's stored loving@L38 'correct' tokens. If this does
# not MATCH, the script's numerics are off -> do NOT trust the L31 output below.
if LENS_PATH:
    lf = torch.load(LENS_PATH, map_location="cpu", weights_only=False)
    stored = [disp(t) for t in lf["rows"]["loving"]["top"][:TOPK]]
    got, _ = topbot(vec("loving", lf["fixed_layer"]), "correct")
    ok = stored == got
    print(f"=== self-validation @ L{lf['fixed_layer']} (loving, correct readout) ===")
    print(f"  stored: {stored}")
    print(f"  recomp: {got}")
    print(f"  ==> {'MATCH - numerics reproduce the artifact; L31 output is trustworthy.' if ok else 'MISMATCH - STOP and debug numerics before trusting anything below.'}")
    print()

# -------------------------------------------- loving grid: the hinge experiment ----
print("=== LOVING: top-8, three readouts x layers (the misnormalization test) ===")
print("    raw=(i) buggy   correct=(ii) canonical   cosine=(iii) token-norm\n")
for layer in LOVING_LAYERS:
    for mode in ("raw", "correct", "cosine"):
        top, _ = topbot(vec("loving", layer), mode, k=8)
        print(f"  L{layer} {mode:7} | " + ", ".join(top))
    print()
print("Read L31: if raw=junk but correct=clean -> the old L31 junk was the "
      "normalization bug, rebuild Table 1 at faithful L31. If correct is STILL "
      "junk -> genuine; L31 faithful gives junk-topped loving (a finding), and "
      "you decide L38-with-disclosure vs L31-with-disclosure.\n")

# ------------------------------- all 12 at the faithful layer, correct numerics ----
print(f"=== ALL 12 @ L{ALL12_LAYER} (faithful depth), correct readout, top-5 ===")
for emo in TWELVE:
    if emo not in emos:
        print(f"  {emo}: MISSING"); continue
    top, _ = topbot(vec(emo, ALL12_LAYER), "correct", k=5)
    print(f"  {emo:10} | " + ", ".join(top))
print("\nEyeball, don't auto-score. If all 12 are on-theme at L31, the faithful "
      "single-layer table is both correct and legible. If L31 rescues nothing "
      "that L38 didn't, staying at L38 (disclosed) is the honest call.")
