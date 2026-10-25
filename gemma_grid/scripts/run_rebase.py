#!/usr/bin/env python3
"""Re-lock step 1: rebuild per-cell cleanup bases from the neutral
activations and re-clean vectors_raw into canonical.pt (all layers).

Conventions identical to emotion_probes/probes.py::project_out_top_pcs
(per-layer SVD of centered neutrals, k = smallest cumvar >= 0.50, project out)
— the same code path validated by the reproduction gates. Earlier artifacts are
NOT touched; outputs are new files:
  gemma_grid/data/<cell>/vectors/canonical.pt   (5 mode-A cells)
  data/vectors: the anchor's canonical vectors ALREADY exist (emotion_vectors_171.pt,
  E1-produced) — verified here against a fresh re-clean at L31 instead of
  being regenerated.
Also persists each cell's cleanup basis (all layers):
  gemma_grid/data/<cell>/vectors/<cell>__cleanup_basis.pt
(the CLAUDE.md save-the-basis rule, honored this time at creation.)

Resumable per cell. Usage: repeat `python gemma_grid/scripts/run_rebase.py`.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
from phase4_stageA import CELLS

TARGET = 0.5
NEUTRALS = {c: REPO / f"gemma_grid/data/{c}/vectors/{c}__neutral.pt" for c in CELLS}
NEUTRALS["gemma-2-27b-base"] = REPO / "gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__neutral.pt"
import os as _os; STATE = Path(_os.environ["BOOT_SCRATCH"]) / "rebase_state.json"


def clean_all_layers(Vraw, N):
    """Vraw (n,L,d) f64, N (500,L,d) f64 -> cleaned (n,L,d), bases list."""
    n, L, d = Vraw.shape
    out = np.empty_like(Vraw)
    bases, ks = [], []
    for l in range(L):
        X = N[:, l, :]
        Xc = X - X.mean(0, keepdims=True)
        _, S, Vh = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2
        ratio = var / var.sum()
        k = int(np.argmax(np.cumsum(ratio) >= TARGET)) + 1
        Vk = Vh[:k]
        out[:, l, :] = Vraw[:, l, :] - (Vraw[:, l, :] @ Vk.T) @ Vk
        bases.append(Vk.astype(np.float32)); ks.append(k)
    return out, bases, ks


def main():
    t0 = time.time()
    st = json.loads(STATE.read_text()) if STATE.exists() else {"done": []}
    for cell in CELLS:
        if cell in st["done"]: continue
        if time.time() - t0 > 10**9:
            print(f"PROGRESS rebase: {st['done']}"); return
        vec_p = REPO / (CELLS[cell]["vec"] if cell != "gemma-2-27b-base" else "data/vectors/emotion_vectors_171.pt")  # anchor raw vectors identical to the as-run vectors_171.pt
        p = torch.load(vec_p, map_location="cpu", weights_only=False, mmap=True)
        em = list(p["emotions"])
        Vraw = (p["vectors_raw"] if torch.is_tensor(p["vectors_raw"]) else torch.as_tensor(p["vectors_raw"])).float().numpy().astype(np.float64)
        nt = torch.load(NEUTRALS[cell], map_location="cpu", weights_only=False)
        N = nt["acts"].float().numpy().astype(np.float64)
        assert N.shape[0] == 500 and N.shape[1] == Vraw.shape[1] and N.shape[2] == Vraw.shape[2], f"{cell}: neutral shape {N.shape} vs raw {Vraw.shape}"
        cleaned, bases, ks = clean_all_layers(Vraw, N)
        if cell == "gemma-2-27b-base":
            # verify against E1's canonical vectors at L31 instead of regenerating
            e1 = torch.load(REPO / "data/vectors/emotion_vectors_171.pt", map_location="cpu", weights_only=False, mmap=True)
            L31 = e1["vectors"][:, 31, :].float().numpy()
            rel = np.linalg.norm(cleaned[:, 31, :] - L31) / np.linalg.norm(L31)
            print(f"[anchor] fresh re-clean vs E1's emotion_vectors_171 @L31: rel {rel:.2e} "
                  f"({'MATCH' if rel < 1e-6 else '**INVESTIGATE**'}) | k@L31={ks[31]}")
        else:
            outp = vec_p.parent / "canonical.pt"
            torch.save({"variant": "canonical_100T", "emotions": em,
                        "vectors": torch.tensor(cleaned, dtype=torch.float32),
                        "vectors_raw": p["vectors_raw"],
                        "n_layers": Vraw.shape[1], "d_model": Vraw.shape[2],
                        "provenance": "re-cleaned 2026-07-31 against HA neutrals (M.4 re-lock); raw unchanged"},
                       outp)
            print(f"[{cell}] canonical.pt written | k@analysis={ks[CELLS[cell]['layer']]}")
        torch.save({"cell": cell, "target": TARGET, "k_per_layer": ks,
                    "analysis_layer": CELLS[cell]["layer"],
                    "bases_per_layer": bases},
                   REPO / f"gemma_grid/data/{cell}/vectors/{cell}__cleanup_basis.pt")
        st["done"].append(cell); STATE.write_text(json.dumps(st))
    print("DONE rebase:", st["done"])


if __name__ == "__main__":
    main()
