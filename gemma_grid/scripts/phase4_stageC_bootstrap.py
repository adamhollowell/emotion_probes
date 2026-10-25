#!/usr/bin/env python3
"""Phase 4 / Stage C (part 1) — story-level bootstrap per cell (GATE 1 (e)).

Pre-registered procedure (PREREGISTRATION.md (e), tier-2): resample each
emotion's stories WITH replacement (B≈1,000), rebuild the 171×d diff-of-means
matrix, project out that cell's OWN OFFICIAL cleaned basis (held fixed; only
emotion-stories are resampled, neutrals never), recompute statistics. The
per-replicate outputs feed both the per-cell SEs (variance %, neighbor
cosines) and the C4 i/ii/iv floors (RSA / Procrustes / displacement), computed
in phase4_stageC_grade.py by pairing independent replicates across cells.

REPRODUCTION GATE (first, per cell — abort on failure): full-sample
reconstruction from the per-story slice (per-emotion mean → subtract
unweighted grand mean → project out basis) must reproduce the cell's official
raw AND cleaned vectors at the analysis layer. Conventions copied from
emotion_probes/probes.py (difference_of_means, project_out_top_pcs).

Basis: gemma_grid/handoff/<cell>__cleanup_basis.pt (official, mode-A cells);
the anchor has no stored file — recomputed from its own recaptured neutral
slice (the recovery validated at the pre-termination gate). Stored bases are
cross-checked against a recompute where both exist.

Loader contract honored: per_emotion keyed BY NAME (anchor dict is corpus
order); analysis layer located by VALUE in the slice's `layers` list.

RNG: SeedSequence([20260725, cell_index, emotion_index]) — deterministic,
independent across cells (required for paired-replicate floors).

RESUMABLE: designed for a 45 s execution window. Each invocation does up to
--budget seconds of work and checkpoints. Call repeatedly until it prints
DONE. State/scratch live under --scratch (NOT the project folder); the final
.npz lands in gemma_grid/geometry_out/<cell>/.

Usage: python phase4_stageC_bootstrap.py --cell <cell> [-B 1000] [--budget 33]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf
from phase4_stageA import CELLS

OUT = REPO / "gemma_grid" / "geometry_out"
SLICES = {c: (CELLS[c]["vec"].rsplit("/vectors/", 1)[0] + f"/vectors/{c}__per_story_slice.pt")
          if c != "gemma-2-27b-base" else "gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__per_story_slice.pt"
          for c in CELLS}
SLICES["gemma-2-27b-base"] = "gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__per_story_slice.pt"
ROOT_SEED = 20260725
VARIANCE_TARGET = 0.5


def get_basis(cell, slc, li):
    X = slc["neutral"][:, li, :].float()
    Xc = X - X.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(X.shape[0] - 1, 1)
    ratio = var / var.sum()
    k = int((torch.cumsum(ratio, 0) >= VARIANCE_TARGET).nonzero()[0].item()) + 1
    recomputed = Vh[:k].numpy().astype(np.float64)
    stored_path = REPO / "gemma_grid" / "handoff" / f"{cell}__cleanup_basis.pt"
    note = {"recomputed_k": k}
    if stored_path.exists():
        b = torch.load(stored_path, map_location="cpu", weights_only=False)
        stored = np.asarray(b["analysis_layer_basis_components"], dtype=np.float64)
        s = np.linalg.svd(stored @ recomputed.T, compute_uv=False)
        note.update(stored_k=int(stored.shape[0]),
                    subspace_overlap=float(np.mean(s)) if stored.shape[0] == k else -1.0)
        return stored, note
    note["anchor_recovery"] = True
    return recomputed, note


def clean(raw, basis):
    return raw - (raw @ basis.T) @ basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("-B", type=int, default=1000)
    ap.add_argument("--budget", type=float, default=33.0)
    ap.add_argument("--scratch", type=Path,
                    default=Path(os.environ.get("BOOT_SCRATCH", ".scratch")) / "bootscratch")
    a = ap.parse_args()
    t0 = time.time()
    cell, spec, B = a.cell, CELLS[a.cell], a.B
    sdir = a.scratch / cell; sdir.mkdir(parents=True, exist_ok=True)
    state_p = sdir / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"emo_done": 0, "rep_done": 0, "gate": None}

    em_canon, V = nf.load_vectors(REPO / spec["vec"])
    layer = spec["layer"]
    slc = torch.load(REPO / SLICES[cell], map_location="cpu", weights_only=False, mmap=True)
    li = list(slc["layers"]).index(layer)
    assert set(slc["per_emotion"].keys()) == set(em_canon)
    basis, basis_note = get_basis(cell, slc, li)
    d = basis.shape[1]; n_emo = len(em_canon)
    cell_idx = list(CELLS).index(cell)

    # ---------- REPRODUCTION GATE (once) ----------
    if state["gate"] is None:
        pfull = torch.load(REPO / spec["vec"], map_location="cpu", weights_only=False, mmap=True)
        Vraw_off = (pfull["vectors_raw"] if torch.is_tensor(pfull["vectors_raw"])
                    else torch.as_tensor(pfull["vectors_raw"])).float().numpy()[:, layer, :]
        mu = np.stack([slc["per_emotion"][e][:, li, :].numpy().astype(np.float64).mean(0) for e in em_canon])
        raw = mu - mu.mean(0, keepdims=True)
        rel_raw = float(np.linalg.norm(raw - Vraw_off) / np.linalg.norm(Vraw_off))
        cln = clean(raw, basis)
        rel_cln = float(np.linalg.norm(cln - V[:, layer, :]) / np.linalg.norm(V[:, layer, :]))
        print(f"  REPRODUCTION GATE: raw {rel_raw:.2e} cleaned {rel_cln:.2e} | basis {basis_note}")
        if rel_raw > 1e-4 or rel_cln > 1e-4:
            print("  GATE FAIL — bootstrap void for this cell."); sys.exit(1)
        state["gate"] = {"raw": rel_raw, "cleaned": rel_cln, "basis": basis_note}
        state_p.write_text(json.dumps(state))

    # ---------- PHASE 1: per-emotion resampled means -> scratch memmap ----------
    mm_p = sdir / f"means_B{B}.f32"
    mm = np.lib.format.open_memmap(mm_p, mode=("r+" if mm_p.exists() else "w+"),
                                   dtype=np.float32, shape=(B, n_emo, d))
    while state["emo_done"] < n_emo and time.time() - t0 < a.budget:
        ei = state["emo_done"]; e = em_canon[ei]
        X = slc["per_emotion"][e][:, li, :].numpy().astype(np.float32)
        ns = X.shape[0]
        rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, cell_idx, ei]))
        counts = rng.multinomial(ns, np.full(ns, 1.0 / ns), size=B).astype(np.float32)
        mm[:, ei, :] = (counts / ns) @ X
        state["emo_done"] = ei + 1
        state_p.write_text(json.dumps(state))
    if state["emo_done"] < n_emo:
        print(f"PROGRESS phase1 {state['emo_done']}/{n_emo} emotions ({time.time()-t0:.0f}s)"); return
    mm.flush()

    # ---------- PHASE 2: per-replicate statistics ----------
    iu = np.triu_indices(n_emo, 1)
    stats_p = sdir / f"stats_B{B}.npz.tmp.npy"  # dict of arrays via np.save
    if (sdir / "stats.npy").exists():
        st = np.load(sdir / "stats.npy", allow_pickle=True).item()
    else:
        SA = json.loads((OUT / cell / "stageA_summary.json").read_text())
        prs = [(pr, SA["neighbours_top5"][pr][0][0]) for pr in nf.PROBES if pr in SA["neighbours_top5"]]
        st = {"tri": np.empty((B, len(iu[0])), np.float32),
              "coords": np.empty((B, n_emo, 5), np.float32),
              "ratios": np.empty((B, 5), np.float32),
              "pair_cos": np.empty((B, len(prs)), np.float32),
              "rs": np.empty((B, 6), np.float32),
              "pairs": np.array([f"{x}->{y}" for x, y in prs])}
    pair_idx = [(em_canon.index(s.split("->")[0]), em_canon.index(s.split("->")[1])) for s in st["pairs"]]
    idx45, pl, ar = nf.load_ratings(Path(REPO / "data/ratings/russell_mehrabian_1977.csv"), em_canon)

    while state["rep_done"] < B and time.time() - t0 < a.budget:
        b = state["rep_done"]
        mu_b = mm[b].astype(np.float64)
        raw_b = mu_b - mu_b.mean(0, keepdims=True)
        cln_b = clean(raw_b, basis)
        C = nf.cos(cln_b)
        st["tri"][b] = C[iu]
        coords, ratio = nf.pca(cln_b, 5)
        if nf.amean(coords[:, 0], em_canon, nf.POS) < nf.amean(coords[:, 0], em_canon, nf.NEG): coords[:, 0] *= -1
        if nf.amean(coords[:, 1], em_canon, nf.HI) < nf.amean(coords[:, 1], em_canon, nf.LO): coords[:, 1] *= -1
        st["coords"][b] = coords; st["ratios"][b] = ratio
        st["pair_cos"][b] = [C[i, j] for i, j in pair_idx]
        c1 = coords[idx45, 0].copy(); c2 = coords[idx45, 1].copy()
        if np.corrcoef(c1, pl)[0, 1] < 0: c1 = -c1
        if np.corrcoef(c2, ar)[0, 1] < 0: c2 = -c2
        norms = np.linalg.norm(cln_b, axis=1)
        st["rs"][b] = [np.corrcoef(c1, pl)[0, 1], np.corrcoef(c2, ar)[0, 1],
                       np.corrcoef(c1, ar)[0, 1], np.corrcoef(c2, pl)[0, 1],
                       np.corrcoef(norms, coords[:, 0])[0, 1],
                       np.corrcoef(norms, np.abs(coords[:, 1]))[0, 1]]
        state["rep_done"] = b + 1
        if (b + 1) % 50 == 0:
            np.save(sdir / "stats.npy", st); state_p.write_text(json.dumps(state))
    np.save(sdir / "stats.npy", st); state_p.write_text(json.dumps(state))
    if state["rep_done"] < B:
        print(f"PROGRESS phase2 {state['rep_done']}/{B} replicates ({time.time()-t0:.0f}s)"); return

    # ---------- ASSEMBLE ----------
    np.savez_compressed(OUT / cell / f"bootstrap_B{B}.npz",
                        tri=st["tri"], coords=st["coords"], ratios=st["ratios"],
                        pair_cos=st["pair_cos"], rs=st["rs"], pairs=st["pairs"],
                        emotions=np.array(em_canon),
                        meta=json.dumps({"cell": cell, "B": B, "layer": layer,
                                         "root_seed": ROOT_SEED, "basis": state["gate"]["basis"],
                                         "reproduction_gate": {k: state["gate"][k] for k in ("raw", "cleaned")}}))
    print(f"DONE [{cell}] B={B} -> bootstrap_B{B}.npz | var_pc1 SD {st['ratios'][:,0].std()*100:.3f}pp | "
          f"valence_r SD {st['rs'][:,0].std():.4f} | arousal_r SD {st['rs'][:,1].std():.4f}")


if __name__ == "__main__":
    main()
