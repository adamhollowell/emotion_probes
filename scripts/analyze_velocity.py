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
import numpy as np

# Load the master vector tensor onto the CPU
data = torch.load('data/vectors/emotion_vectors_171.pt', map_location='cpu')
emotions = data['emotions']
vectors_raw = data['vectors_raw'].float().numpy()  # shape: [171, 46, 4608]
vectors_clean = data['vectors'].float().numpy()    # shape: [171, 46, 4608]

target_emotions = ['afraid', 'angry', 'happy', 'sad', 'guilty', 'loving']

def compute_layer_velocity(vecs, emotion_idx):
    e_vecs = vecs[emotion_idx]
    norms = np.linalg.norm(e_vecs, axis=-1, keepdims=True)
    normed_vecs = e_vecs / np.maximum(norms, 1e-8)
    step_similarities = []
    for i in range(45):
        sim = np.dot(normed_vecs[i], normed_vecs[i+1])
        step_similarities.append(sim)
    return np.array(step_similarities)

print("======================================================================")
print("     EMOTION REPRESENTATIONAL VELOCITY TRACKING (Gemma 2 27B)")
print("======================================================================")
print("Tracking step similarity change (Layer N -> Layer N+1). Higher = More Stable.")
print("-" * 70)

for emo_name in target_emotions:
    if emo_name not in emotions:
        print(f"Emotion '{emo_name}' not found in dataset primitives.")
        continue
        
    idx = emotions.index(emo_name)
    raw_sims = compute_layer_velocity(vectors_raw, idx)
    clean_sims = compute_layer_velocity(vectors_clean, idx)
    
    print(f"\nTarget Emotion: {emo_name.upper()}")
    print(f"  Raw Vector Stability bounds:   Min={raw_sims.min():.4f} (Layer {raw_sims.argmin()}), Max={raw_sims.max():.4f} (Layer {raw_sims.argmax()})")
    print(f"  Clean Vector Stability bounds: Min={clean_sims.min():.4f} (Layer {clean_sims.argmin()}), Max={clean_sims.max():.4f} (Layer {clean_sims.argmax()})")
    
    print("  Architectural Checkpoints (Clean Vectors Step-Stability):")
    print(f"    Early Pass (L2->L3):    {clean_sims[2]:.4f}")
    print(f"    The Layer 9 Snap:      {clean_sims[8]:.4f}")
    print(f"    Mid-Late Peak (L30->L31): {clean_sims[30]:.4f}")
    print(f"    Late Readout (L37->L38):  {clean_sims[37]:.4f}")
    
    ascii_spark = ""
    for val in clean_sims[::2]:
        if val > 0.98: ascii_spark += "█"
        elif val > 0.95: ascii_spark += "▇"
        elif val > 0.90: ascii_spark += "▅"
        elif val > 0.80: ascii_spark += "▃"
        else: ascii_spark += "_"
    print(f"  Clean Stability Profile [L0->L45]: [{ascii_spark}]")

print("\n======================================================================")
