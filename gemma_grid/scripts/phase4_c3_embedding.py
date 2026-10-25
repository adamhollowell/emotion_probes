#!/usr/bin/env python3
"""C3 / signal 12 — embedding-layer baseline for all six grid cells.

Registered definition (PREREGISTRATION (c) #12): bag-of-embeddings null —
mean of static token embeddings over tokens[50:] per story, same
diff-of-means, cleanup basis refit on the neutral dialogues' OWN embeddings —
then: valence/arousal corr, PC variance, RSA to the analysis layer, ARI vs
the cell's locked k=10 partition, neighbor cosines.
Anchor reference (E1 §6.6): valence 0.509 / arousal 0.108.

Inputs:
  * gemma_grid/data/c3_tokens.npz — raw token IDs (desktop step,
    c3_tokenize_corpus.py; ONE tokenizer serves all cells).
  * Embedding tables:
      - ANCHOR: data/vectors/embedding_null_vectors.pt — round one's vectors
        from the TRUE embed_tokens table (no tie assumption for the anchor).
      - 5 grid cells: W_E = W_U^T from <cell>__lens_ingredients.pt (bf16 ->
        f32), per the approved tied-read policy (PHASE4_SCOPE §7.4).
  * TIED-READ VALIDATION (one-time, before any grid cell is read): the
    anchor's W_U (round-one lens_ingredients.pt, f32, streamed from the zip
    in row blocks — file is too large to mmap here) is used to rebuild the
    anchor's raw diff-of-means via the tied read; it must match
    embedding_null_vectors.pt's vectors_raw. That match validates BOTH the
    tie identity on real weights AND this script's tokenization/pooling in
    one shot.

Conventions copied from scripts/extract_embedding_null.py /
emotion_probes/probes.py: pooling start = 50 if n > 51 else min(1, n-1);
grand mean unweighted over the 171 emotion means; cleanup = project out
top-k neutral-embedding PCs at 50% cumulative variance (SVD of centered).

RESUMABLE (45 s window): run repeatedly until DONE. State in
$BOOT_SCRATCH/c3scratch/.

Usage: python gemma_grid/scripts/phase4_c3_embedding.py [--budget 33]
Output: gemma_grid/geometry_out/<cell>/c3_embedding.json (+ c3_vectors.pt)
        gemma_grid/geometry_out/c3_summary.json
"""
import argparse, json, os, struct, sys, time, zipfile
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gemma_grid" / "scripts"))
import noise_floor as nf
from phase4_stageA import CELLS

OUT = REPO / "gemma_grid" / "geometry_out"
TOKENS = REPO / "gemma_grid" / "data" / "c3_tokens.npz"
SCRATCH = Path(os.environ.get("BOOT_SCRATCH", ".scratch")) / "c3scratch"
SKIP = 50
TARGET = 0.5
VOCAB = 256000
LENS = {c: REPO / f"gemma_grid/data/{c}/vectors/{c}__lens_ingredients.pt" for c in CELLS}
ANCHOR_LENS = REPO / "data/vectors/lens_ingredients.pt"
ANCHOR_NULL = REPO / "data/vectors/embedding_null_vectors.pt"


def build_weights():
    """(171+500, VOCAB) f64 weight matrix W s.t. W @ W_E = [emotion means; neutral vectors]."""
    z = np.load(TOKENS, allow_pickle=True)
    ids, offsets, labels = z["ids"], z["offsets"], z["labels"]
    emotions = sorted(set(labels) - {"__neutral__"})
    assert len(emotions) == 171
    rows = {e: i for i, e in enumerate(emotions)}
    n_stories = int(z["n_stories"])
    counts = {e: int((labels[:n_stories] == e).sum()) for e in emotions}
    W = np.zeros((171 + int(z["n_neutral"]), VOCAB), np.float64)
    nrow = 171
    for t in range(len(offsets) - 1):
        seg = ids[offsets[t]:offsets[t + 1]]
        n = len(seg)
        start = SKIP if n > SKIP + 1 else min(1, n - 1)
        pooled = seg[start:]
        lab = labels[t]
        if lab == "__neutral__":
            np.add.at(W[nrow], pooled, 1.0 / len(pooled)); nrow += 1
        else:
            np.add.at(W[rows[lab]], pooled, 1.0 / (len(pooled) * counts[lab]))
    return emotions, W


def stream_anchor_WU_matmul(Wt, budget, t0, part_p, state, key="v_tok_done"):
    """Stream the anchor W_U storage out of the torch zip. VERIFIED from the
    pickle: W_U has size (4608, VOCAB) with stride (1, 4608) — i.e. the
    underlying storage is embed_tokens.weight laid out (VOCAB, 4608)
    row-major (W_U is a transposed VIEW). So we stream TOKEN-major blocks:
    acc += Wt[:, t0:t1] @ block(t_rows, 4608). Entry is ZIP_STORED, and
    zipfile's ExtFile supports seek on stored entries."""
    zf = zipfile.ZipFile(ANCHOR_LENS)
    name = [n for n in zf.namelist() if n.endswith("data/0")][0]
    assert zf.getinfo(name).compress_type == zipfile.ZIP_STORED
    acc = np.load(part_p) if part_p.exists() else np.zeros((Wt.shape[0], 4608), np.float64)
    row_bytes = 4608 * 4
    with zf.open(name) as f:
        f.seek(state.get(key, 0) * row_bytes)
        while state.get(key, 0) < VOCAB and time.time() - t0 < budget:
            t = state.get(key, 0)
            block = min(20000, VOCAB - t)
            buf = f.read(block * row_bytes)
            rows = np.frombuffer(buf, np.float32).reshape(block, 4608)
            acc += Wt[:, t:t + block] @ rows
            state[key] = t + block
    np.save(part_p, acc)
    return acc, state.get(key, 0) >= VOCAB


def cell_embed_matmul(cell, Wt):
    """(rows, d) = Wt @ W_E for a grid cell via mmap'd bf16 W_U (d, VOCAB)."""
    l = torch.load(LENS[cell], map_location="cpu", weights_only=False, mmap=True)
    WU = l["W_U"]
    d = WU.shape[0]
    acc = np.zeros((Wt.shape[0], d), np.float64)
    for v0 in range(0, VOCAB, 16000):
        sl = WU[:, v0:v0 + 16000].float().numpy()          # (d, v)
        acc += Wt[:, v0:v0 + 16000] @ sl.T
    return acc


def clean_from_neutral(raw, neutral):
    X = neutral - neutral.mean(0, keepdims=True)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / max(len(neutral) - 1, 1)
    ratio = var / var.sum()
    k = int(np.argmax(np.cumsum(ratio) >= TARGET)) + 1
    Vk = Vh[:k]
    return raw - (raw @ Vk.T) @ Vk, k


def signals(cell, em, Vemb_raw, Vemb, layer):
    canon_em, Vfull = nf.load_vectors(REPO / CELLS[cell]["vec"])
    assert canon_em == em
    Vl = Vfull[:, layer, :]
    S = {}
    coords, ratio = nf.pca(Vemb, 5)
    if nf.amean(coords[:, 0], em, nf.POS) < nf.amean(coords[:, 0], em, nf.NEG): coords[:, 0] *= -1
    if nf.amean(coords[:, 1], em, nf.HI) < nf.amean(coords[:, 1], em, nf.LO): coords[:, 1] *= -1
    S["var_pc"] = [float(100 * r) for r in ratio[:3]]
    idx, pl, ar = nf.load_ratings(REPO / "data/ratings/russell_mehrabian_1977.csv", em)
    c1, c2 = coords[idx, 0].copy(), coords[idx, 1].copy()
    if np.corrcoef(c1, pl)[0, 1] < 0: c1 = -c1
    if np.corrcoef(c2, ar)[0, 1] < 0: c2 = -c2
    S["valence_r"] = float(np.corrcoef(c1, pl)[0, 1])
    S["arousal_r"] = float(np.corrcoef(c2, ar)[0, 1])
    n = len(em); iu = np.triu_indices(n, 1)
    Ce, Cl = nf.cos(Vemb), nf.cos(Vl)
    S["rsa_to_analysis_layer"] = float(np.corrcoef(Ce[iu], Cl[iu])[0, 1])
    SA = json.loads((OUT / cell / "stageA_summary.json").read_text())
    lab = np.full(n, -1)
    for ci, cl_ in enumerate(SA["kmeans_partition"]["clusters"]):
        for m in cl_["members"]: lab[em.index(m)] = ci
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score as ari
    emb_lab = KMeans(n_clusters=10, n_init=10, random_state=0).fit(Vemb).labels_
    S["ari_vs_locked_partition"] = float(ari(lab, emb_lab))
    anchor_SA = json.loads((OUT / "gemma-2-27b-base" / "stageA_summary.json").read_text())
    S["neighbor_cos"] = {}
    for p_, nb in anchor_SA["neighbours_top5"].items():
        i, j = em.index(p_), em.index(nb[0][0])
        S["neighbor_cos"][f"{p_}->{nb[0][0]}"] = float(Ce[i, j])
    det = SA["deterministic"]
    S["depth_gap"] = {"valence": det["valence_r"] - S["valence_r"],
                      "arousal": det["arousal_r"] - S["arousal_r"]}
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=33.0)
    a = ap.parse_args(); t0 = time.time()
    if not TOKENS.exists():
        sys.exit(f"WAITING: {TOKENS} not present yet (desktop step c3_tokenize_corpus.py).")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    state_p = SCRATCH / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else \
        {"weights": False, "v_rows_done": 0, "validated": None, "cells_done": []}

    wp = SCRATCH / "weights.npy"; ep = SCRATCH / "emotions.json"
    if not state["weights"]:
        emotions, W = build_weights()
        np.save(wp, W); ep.write_text(json.dumps(emotions))
        state["weights"] = True; state_p.write_text(json.dumps(state))
        print(f"weights built ({time.time()-t0:.0f}s)")
        if time.time() - t0 > a.budget: return
    emotions = json.loads(ep.read_text()); W = np.load(wp, mmap_mode="r")

    # ---- one-time tied-read validation against the anchor's TRUE-embedding artifact
    if state["validated"] is None:
        acc, done = stream_anchor_WU_matmul(np.asarray(W), a.budget, t0, SCRATCH / "anchor_acc_tokmajor.npy", state)
        state_p.write_text(json.dumps(state))
        if not done:
            print(f"PROGRESS tied-validation {state['v_rows_done']}/4608 W_U rows"); return
        mu = acc[:171]; raw_tied = mu - mu.mean(0, keepdims=True)
        nullv = torch.load(ANCHOR_NULL, map_location="cpu", weights_only=False)
        em0 = list(nullv["emotions"]); order = [em0.index(e) for e in emotions]
        raw_true = nullv["vectors_raw"].float().numpy()[:, 0, :][order]
        rel = float(np.linalg.norm(raw_tied - raw_true) / np.linalg.norm(raw_true))
        state["validated"] = {"rel_diff_tied_vs_true": rel}
        state_p.write_text(json.dumps(state))
        print(f"TIED-READ VALIDATION: rel diff {rel:.2e} (tied W_U^T vs true embed_tokens)")
        if rel > 1e-2:
            sys.exit("VALIDATION FAIL — do not read grid cells via tied W_E. STOP.")

    # ---- per cell
    summary_p = OUT / "c3_summary.json"
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}
    for cell in CELLS:
        if cell in state["cells_done"]: continue
        if time.time() - t0 > a.budget:
            print(f"PROGRESS cells {state['cells_done']}"); return
        layer = CELLS[cell]["layer"]
        if cell == "gemma-2-27b-base":
            nullv = torch.load(ANCHOR_NULL, map_location="cpu", weights_only=False)
            em0 = list(nullv["emotions"]); order = [em0.index(e) for e in emotions]
            raw = nullv["vectors_raw"].float().numpy()[:, 0, :][order]
            cleaned = nullv["vectors"].float().numpy()[:, 0, :][order]
            k = -1; src = "embedding_null_vectors.pt (TRUE embed table, round one)"
        else:
            acc = cell_embed_matmul(cell, np.asarray(W))
            mu, neu = acc[:171], acc[171:]
            raw = mu - mu.mean(0, keepdims=True)
            cleaned, k = clean_from_neutral(raw, neu)
            src = "tied W_E = W_U^T from lens ingredients (validated vs anchor true table)"
        S = {"cell": cell, "source": src, "basis_k": k, **signals(cell, emotions, raw, cleaned, layer)}
        (OUT / cell / "c3_embedding.json").write_text(json.dumps(S, indent=2))
        torch.save({"emotions": emotions, "vectors": torch.tensor(cleaned, dtype=torch.float32).unsqueeze(1),
                    "vectors_raw": torch.tensor(raw, dtype=torch.float32).unsqueeze(1),
                    "layer": "embed_tokens", "source": src},
                   OUT / cell / "c3_vectors.pt")
        summary[cell] = {kk: S[kk] for kk in ("var_pc", "valence_r", "arousal_r",
                                              "rsa_to_analysis_layer", "ari_vs_locked_partition", "depth_gap")}
        state["cells_done"].append(cell); state_p.write_text(json.dumps(state))
        summary_p.write_text(json.dumps(summary, indent=2))
        print(f"[{cell}] valence {S['valence_r']:.3f} arousal {S['arousal_r']:.3f} "
              f"RSA {S['rsa_to_analysis_layer']:.3f} ARI {S['ari_vs_locked_partition']:.3f} "
              f"depth-gap aro {S['depth_gap']['arousal']:+.3f}")
    if len(state["cells_done"]) == len(CELLS):
        anch = summary["gemma-2-27b-base"]
        print(f"ANCHOR GATE: valence {anch['valence_r']:.3f} (E1: 0.509) "
              f"arousal {anch['arousal_r']:.3f} (E1: 0.108)")
        print("DONE — all six cells. Ready for grading + reviewer pass.")


if __name__ == "__main__":
    main()
