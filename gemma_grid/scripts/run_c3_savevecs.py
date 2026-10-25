#!/usr/bin/env python3
"""Persist per-cell 100T-cleaned c3 embedding vectors into
geometry_out/<cell>/c3_vectors.pt.

run_c3.py (step 8) computes the 100T-cleaned embeddings (clean_from_neutral)
but only writes c3_embedding.json / c3_summary.json; it does NOT persist the cleaned
vectors tensor. An earlier generation persisted them via a dedicated savevecs step,
and this mirrors that for 100T so step 9 c3grade reads 100T-cleaned
vectors from the 100T tree.

On-disk contract:
  {"emotions": list, "vectors": (171,1,d) 100T-cleaned, "vectors_raw": (171,1,d) from the stored file
   (basis-invariant, carried through), "layer": "embed_tokens", "source": str}

Idempotent (skips cells whose c3_vectors.pt already exists). Runs under lens_env (torch).
Reuses the same neutral_embeddings step 8 used (run_c3.H) — anchor W_U from
BOOT_SCRATCH/anchor_WU_f16.npy, others from per-cell lens_ingredients — and the cached
neutral weights step 8 built (BOOT_SCRATCH/c3scratch_100T/neutral_weights.npy).
"""
import os, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
from phase4_stageA import CELLS
import phase4_c3_embedding as C3
import run_c3 as H            # same module/neutral_embeddings step 8 used

OUT = REPO / "gemma_grid" / "geometry_out"
RAW_SRC = REPO / "gemma_grid" / "geometry_out"


def main():
    t0 = time.time()
    wpath = Path(os.environ["BOOT_SCRATCH"]) / "c3scratch_100T" / "neutral_weights.npy"
    if not wpath.exists():
        sys.exit(f"FATAL: {wpath} missing — run step 8 (c3) first so the neutral weights are cached")
    Wt = np.load(wpath, mmap_mode="r")
    for cell in CELLS:
        outp = OUT / cell / "c3_vectors.pt"
        if outp.exists():
            print(f"[{cell}] c3_vectors.pt exists — skip")
            continue
        pa = torch.load(RAW_SRC / cell / "c3_vectors.pt", map_location="cpu", weights_only=False)
        raw = pa["vectors_raw"].float().numpy()[:, 0, :]
        neu = H.neutral_embeddings(cell, np.asarray(Wt))
        cleaned, k = C3.clean_from_neutral(raw, neu)
        outp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"emotions": list(pa["emotions"]),
                    "vectors": torch.tensor(cleaned, dtype=torch.float32).unsqueeze(1),
                    "vectors_raw": pa["vectors_raw"], "layer": "embed_tokens",
                    "source": f"100T neutral basis (k={k}); raw reused from PA (story-derived, basis-invariant)"},
                   outp)
        print(f"[{cell}] c3_vectors.pt saved (100T-cleaned, k={k}, {time.time()-t0:.0f}s)")
    print("DONE c3-100T-savevecs")


if __name__ == "__main__":
    main()
