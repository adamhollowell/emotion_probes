"""
lens_readout_decision.py  (merged)

Settles the lens-READOUT choice, framed correctly: NOT "which convention is
correct" (provenance answers that -- ln_final_weight is the RAW Gemma weight, so
1+w is the correct normalization). It IS "which readout gives the most
interpretable mid-layer table, given that applying final-layer RMSNorm to a
mid-layer DIFFERENCE vector is inherently approximate."

Two readouts compared (cosine and 'noplus' dropped):
  (A) raw   : v @ W_U                      -- plain logit lens, no final norm
  (B) onepw : (v/rms(v)) * (1+w) @ W_U     -- Gemma-convention RMSNorm, gain once
cosine: W_U rows ~unit-norm -> cosine == raw. Dead.
noplus: only correct if stored weight were the EFFECTIVE gain; provenance says it
        is the RAW weight, so 1+w is the convention and noplus tests a ruled-out case.

Run from anywhere (absolute paths). CPU fine.
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

ING  = "data/vectors/lens_ingredients.pt"
VEC  = "data/vectors/emotion_vectors_171.pt"
LENS = "data/vectors/lens_final_fixed.pt"

# Anthropic Table 1, exact set and order:
TWELVE = ['happy','inspired','loving','proud','calm','desperate',
          'angry','guilty','sad','afraid','nervous','surprised']

ing = torch.load(ING, map_location='cpu', weights_only=False)
d_model = int(ing['d_model']); Vsz = int(ing['vocab_size'])
W_U = ing['W_U'].float()
# --- FIX 3: orientation guard. Normalize to (d_model, vocab) so v @ W_U is valid. ---
if W_U.shape == (Vsz, d_model):
    W_U = W_U.t().contiguous()
assert W_U.shape == (d_model, Vsz), \
    f"W_U {tuple(W_U.shape)} is neither (d_model,vocab) nor (vocab,d_model) for d_model={d_model} vocab={Vsz}"
ln_w  = ing['ln_final_weight'].float()
eps   = float(ing.get('ln_final_eps', 1e-6))
vocab = ing['vocab']

print("="*70)
print("PROVENANCE / CONVENTION NOTE (authoritative on the +1 question)")
print("="*70)
for k in ('gemma_rmsnorm_note','ln_final_type','tl_compatibility_mode','provenance'):
    print(f"  {k}: {ing.get(k)}")
print(f"  ln_final_weight stats: min={ln_w.min():.3f} max={ln_w.max():.3f} mean={ln_w.mean():.3f}")
print(f"  (1+w) gain stats:      min={(1+ln_w).min():.3f} max={(1+ln_w).max():.3f} mean={(1+ln_w).mean():.3f}")
print("  W_U per-token-norm pctiles: " +
      ", ".join(f"{p}%={torch.quantile(W_U.norm(dim=0), p/100):.3f}" for p in (1,25,50,75,99)))
print("  -> note should say ln_final_weight is the RAW weight => 1+w(onepw) is the convention,")
print("     extreme gain is a real Gemma property, raw-vs-onepw is a DISCLOSURE choice.\n")

vd = torch.load(VEC, map_location='cpu', weights_only=False)
emotions = vd['emotions']
V = vd['vectors'].float()

def raw_readout(v, k=8):
    return [vocab[t] for t in (v.float() @ W_U).topk(k).indices.tolist()]
def onepw_readout(v, k=8):
    v = v.float(); rms = torch.sqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return [vocab[t] for t in (((v/rms)*(1.0+ln_w)) @ W_U).topk(k).indices.tolist()]

# --- FIX 2: self-validation gate. Reproduce lens_final_fixed.pt (built with onepw). ---
lf = torch.load(LENS, map_location='cpu', weights_only=False)
stored = [t.strip() for t in lf['rows']['loving']['top'][:12]]
got    = [t.strip() for t in onepw_readout(V[emotions.index('loving'), lf['fixed_layer']], 12)]
print("="*70)
print(f"SELF-VALIDATION @ L{lf['fixed_layer']} (loving, onepw): "
      f"{'MATCH - readout reproduces the artifact; trust what follows.' if stored==got else 'MISMATCH - STOP, debug before trusting anything below.'}")
print("="*70 + "\n")

print("LOVING across layers (raw vs onepw)")
for L in (31, 38, 43):
    i = emotions.index('loving')
    print(f"  loving L{L}")
    print(f"     raw  : " + ' '.join(repr(t.strip()) for t in raw_readout(V[i,L])))
    print(f"     onepw: " + ' '.join(repr(t.strip()) for t in onepw_readout(V[i,L])))
    print()

print("="*70)
print("ANTHROPIC's 12 @ faithful L31 (top-5) -- is either readout uniformly on-theme?")
print("="*70)
for e in TWELVE:
    if e not in emotions:
        print(f"  {e:10} MISSING from artifact"); continue
    v = V[emotions.index(e), 31]
    print(f"  {e:10} raw  : " + ' '.join(repr(t.strip()) for t in raw_readout(v,5)))
    print(f"  {e:10} onepw: " + ' '.join(repr(t.strip()) for t in onepw_readout(v,5)))
    print()

print("READ ORDER:")
print(" 1. Confirm the note says ln_final_weight is the RAW weight (=> onepw is the convention).")
print(" 2. SELF-VALIDATION must say MATCH, else stop.")
print(" 3. loving: junk under BOTH raw and onepw at L31 but clean at L43 => lateness is GENUINE.")
print(" 4. 12-grid at L31: whichever readout (if any) is uniformly on-theme is the candidate report readout.")
