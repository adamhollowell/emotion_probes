#!/usr/bin/env python3
"""C3 (embedding baseline) over the 100T neutral basis.

Only the cleanup basis changes: the raw embedding-layer emotion vectors
(story-derived) are UNCHANGED and are reused from the earlier run's saved
c3_vectors.pt (vectors_raw). This script: (1) builds the 500 100T-neutral
bag-of-embedding vectors per cell (weights from c3_tokens_neutral.npz ×
the cell's W_E = W_U^T; anchor uses the tied read, validated rel 1.4e-3);
(2) refits the neutral-embedding basis (50% cumvar, canonical convention);
(3) re-cleans the raw vectors; (4) recomputes the registered C3 signals
against the 100T-tree Stage A partitions. Outputs to geometry_out/.
Resumable per cell."""
import json, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf
from phase4_stageA import CELLS
import phase4_c3_embedding as C3   # reuse clean_from_neutral + signals machinery

OUT = REPO / "gemma_grid" / "geometry_out"
RAW_SRC = REPO / "gemma_grid" / "geometry_out"   # raw c3 vectors are read from the stored files
import os as _os; SCR = Path(_os.environ["BOOT_SCRATCH"]) / "c3scratch_100T"
LENS = {c: REPO / f"gemma_grid/data/{c}/vectors/{c}__lens_ingredients.pt" for c in CELLS}
VOCAB_N = 256000


def build_neutral_weights():
    z = np.load(REPO / "gemma_grid/data/c3_tokens_neutral.npz", allow_pickle=True)
    ids, offsets = z["ids"], z["offsets"]
    n = len(offsets) - 1
    assert n == 500
    W = np.zeros((n, VOCAB_N), np.float64)
    for t in range(n):
        seg = ids[offsets[t]:offsets[t + 1]]
        L = len(seg)
        start = 50 if L > 51 else min(1, L - 1)
        pooled = seg[start:]
        np.add.at(W[t], pooled, 1.0 / len(pooled))
    return W


def neutral_embeddings(cell, Wt):
    if cell == "gemma-2-27b-base":
        # anchor: tied read from its f32 W_U via the converted f16 memmap
        import os as _os; mm = np.load(str(Path(_os.environ["BOOT_SCRATCH"]) / "anchor_WU_f16.npy"), mmap_mode="r")
        acc = np.zeros((Wt.shape[0], 4608), np.float64)
        for t0 in range(0, VOCAB_N, 20000):
            acc += Wt[:, t0:t0 + 20000] @ mm[t0:t0 + 20000].astype(np.float32)
        return acc
    l = torch.load(LENS[cell], map_location="cpu", weights_only=False, mmap=True)
    WU = l["W_U"]; d = WU.shape[0]
    acc = np.zeros((Wt.shape[0], d), np.float64)
    for v0 in range(0, VOCAB_N, 16000):
        acc += Wt[:, v0:v0 + 16000] @ WU[:, v0:v0 + 16000].float().numpy().T
    return acc


def main():
    t0 = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    st_p = SCR / "state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {"weights": False, "done": []}
    wp = SCR / "neutral_weights.npy"
    if not st["weights"]:
        np.save(wp, build_neutral_weights())
        st["weights"] = True; st_p.write_text(json.dumps(st))
        print(f"neutral weights built ({time.time()-t0:.0f}s)")
    Wt = np.load(wp, mmap_mode="r")

    # patch the signals helper to read this tree's partitions/vectors
    C3.OUT = OUT
    import phase4_stageA as A
    for c, spec in A.CELLS.items():
        spec["vec"] = (spec["vec"].replace("canonical.pt", "canonical.pt")
                       if c != "gemma-2-27b-base" else "data/vectors/emotion_vectors_171.pt")

    summary_p = OUT / "c3_summary.json"
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}
    for cell in CELLS:
        if cell in st["done"]: continue
        if time.time() - t0 > 10**9:
            print(f"PROGRESS c3: {st['done']}"); return
        pa = torch.load(RAW_SRC / cell / "c3_vectors.pt", map_location="cpu", weights_only=False)
        emotions = list(pa["emotions"])
        raw = pa["vectors_raw"].float().numpy()[:, 0, :]
        neu = neutral_embeddings(cell, np.asarray(Wt))
        cleaned, k = C3.clean_from_neutral(raw, neu)
        S = {"cell": cell, "source": "raw reused from PA c3 (story-derived, basis-independent); basis refit on HA neutral embeddings",
             "basis_k": k, **C3.signals(cell, emotions, raw, cleaned, CELLS[cell]["layer"])}
        (OUT / cell / "c3_embedding.json").write_text(json.dumps(S, indent=2))
        summary[cell] = {kk: S[kk] for kk in ("var_pc", "valence_r", "arousal_r",
                                              "rsa_to_analysis_layer", "ari_vs_locked_partition", "depth_gap")}
        summary_p.write_text(json.dumps(summary, indent=2))
        st["done"].append(cell); st_p.write_text(json.dumps(st))
        print(f"[{cell}] valence {S['valence_r']:.3f} arousal {S['arousal_r']:.3f} k={k}")
    print("DONE c3")


if __name__ == "__main__":
    main()
