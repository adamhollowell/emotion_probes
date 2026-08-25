"""
render_parallel_table1_unembedding.py

Produces the data for the REPLICATION paper's Table 1 parallel:
Anthropic's exact 12 emotions, layer 31 (2/3 depth, faithful), RAW readout
(v @ W_U, no final-norm gain -- the disclosed, more-legible choice).

Prints top-5 AND bottom-5 per emotion (Anthropic's panel shows both), plus a
self-validation gate so we know the readout reproduces the artifact before trusting.
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

TWELVE = ['happy','inspired','loving','proud','calm','desperate',
          'angry','guilty','sad','afraid','nervous','surprised']
LAYER = 31

ing = torch.load(ING, map_location='cpu', weights_only=False)
d_model = int(ing['d_model']); Vsz = int(ing['vocab_size'])
W_U = ing['W_U'].float()
if W_U.shape == (Vsz, d_model):
    W_U = W_U.t().contiguous()
assert W_U.shape == (d_model, Vsz), f"W_U {tuple(W_U.shape)} unexpected"
ln_w  = ing['ln_final_weight'].float()
eps   = float(ing.get('ln_final_eps', 1e-6))
vocab = ing['vocab']

vd = torch.load(VEC, map_location='cpu', weights_only=False)
emotions = vd['emotions']
V = vd['vectors'].float()

def raw_logits(v):           return v.float() @ W_U
def onepw_logits(v):
    v = v.float(); rms = torch.sqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return ((v / rms) * (1.0 + ln_w)) @ W_U
def top(logits, k):    return [vocab[t].strip() for t in logits.topk(k).indices.tolist()]
def bottom(logits, k): return [vocab[t].strip() for t in (-logits).topk(k).indices.tolist()]

# self-validation: onepw must reproduce lens_final_fixed.pt @ its fixed layer
lf = torch.load(LENS, map_location='cpu', weights_only=False)
stored = [t.strip() for t in lf['rows']['loving']['top'][:12]]
got    = top(onepw_logits(V[emotions.index('loving'), lf['fixed_layer']]), 12)
print(f"SELF-VALIDATION @ L{lf['fixed_layer']} (onepw): "
      f"{'MATCH' if stored==got else 'MISMATCH - STOP, debug before using output'}\n")

print(f"TABLE 1 DATA  --  layer {LAYER}, RAW readout (v @ W_U), top-5 / bottom-5\n")
for e in TWELVE:
    if e not in emotions:
        print(f"{e}: MISSING\n"); continue
    lg = raw_logits(V[emotions.index(e), LAYER])
    print(f"{e}")
    print(f"   up  : " + ', '.join(top(lg, 5)))
    print(f"   down: " + ', '.join(bottom(lg, 5)))
    print()
