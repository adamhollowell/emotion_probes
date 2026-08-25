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

import torch, re
ing = torch.load('data/vectors/lens_ingredients.pt', map_location='cpu', weights_only=False)
W_U = ing['W_U'].float(); ln_w = ing['ln_final_weight'].float(); eps = float(ing.get('ln_final_eps',1e-6))
vocab = ing['vocab']
vd = torch.load('data/vectors/emotion_vectors_171.pt', map_location='cpu', weights_only=False)
emotions = vd['emotions']; V = vd['vectors'].float(); N = V.shape[1]

def lens(vec, k=8):
    v = vec.float()
    rms = torch.sqrt(v.pow(2).mean(-1,keepdim=True)+eps)
    logits = ((v/rms)*(1.0+ln_w)) @ W_U
    return [vocab[t].strip() for t in logits.topk(k).indices.tolist()]

WARM = ['loving','grateful','kind','compassionate','thankful','empathetic','sympathetic']
DIRECTED = ['scornful','contemptuous','disdainful','self-confident','smug','defiant']
CONTROL = ['tired','sad','calm','proud','disgusted','angry']   # self-states, should NOT be 'you'-directed
PRON = {'you','your','yours','i','me','my','mine','we','us','our'}

def dump(group, title):
    print(f"\n########## {title} ##########")
    for e in group:
        if e not in emotions: 
            print(f"  (skip {e})"); continue
        i = emotions.index(e)
        print(f"\n--- {e} (all layers) ---")
        for l in range(N):
            toks = lens(V[i,l])
            pron = [t for t in toks if t.lower() in PRON]
            mark = '  <2nd/1st person: '+','.join(pron)+'>' if pron else ''
            print(f"  L{l:2d}: " + ' '.join(repr(t) for t in toks) + mark)

dump(WARM, "WARM-AFFILIATIVE (your core cluster)")
dump(DIRECTED, "DIRECTED-STANCE (object-directedness check)")
dump(CONTROL, "CONTROL self-states (pronouns here = artifact)")

# summary: pronoun specificity
print("\n########## PRONOUN-SPECIFICITY SUMMARY ##########")
print("(fraction of layers where a 1st/2nd-person pronoun appears in top-8)")
for grp,name in [(DIRECTED,'directed'),(CONTROL,'control'),(WARM,'warm')]:
    for e in grp:
        if e not in emotions: continue
        i = emotions.index(e)
        hits = sum(1 for l in range(N) if any(t.lower() in PRON for t in lens(V[i,l])))
        print(f"  {name:9s} {e:16s}: {hits}/{N} layers")
