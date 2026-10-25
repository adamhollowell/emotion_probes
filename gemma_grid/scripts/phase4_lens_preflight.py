#!/usr/bin/env python3
"""Anchor lens-asset preflight — regenerates the four anchor lens assets that
lived only in dead pod scratch, into $BOOT_SCRATCH, BEFORE step 8 (c3, needs the
anchor W_U memmap) and step 10 (lens, needs all four). Basis-INDEPENDENT: model
weights + tokenizer vocab, no cleaned vectors touched. Tokenizer-free AND
torch-free: every asset comes from the repo's own lens_ingredients.pt (the pod's
committed anchor bundle) + the 100T anchor emotion list, so vocab and W_U are
byte-identical to what the pod used. The lens anchor gate (onset==L23) validates
the result downstream.

Emits into $BOOT_SCRATCH (or --out):
  anchor_vocab.json       list[str] len 256000, id-indexed display form (' word')
  emotion_token_ids.json  {emotion: token_id} for single-token ' word' forms
  anchor_ln_final.npy     (4608,) f32   <- lens_ingredients data/1 (ln_final_weight)
  anchor_WU_f16.npy       (256000,4608) f16 memmap  <- lens_ingredients data/0 (token-major)
"""
import argparse, json, os, sys, zipfile, pickle, io
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LENS_ING = REPO / "data/vectors/lens_ingredients.pt"
ANCHOR_VEC = REPO / "data/vectors/emotion_vectors_171.pt"
V_SIZE = 256000
D_MODEL = 4608


class _Skip:  # stand-in for torch storages/rebuilders we do not materialize
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return None
    def __setstate__(self, s): pass


def _plain(pt_path, key):
    """Unpickle a torch .pt zip, skipping tensor storages; return top dict[key]
    (valid only for pure-python values: lists/dicts of str/num)."""
    z = zipfile.ZipFile(pt_path)
    pkl = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))

    class U(pickle.Unpickler):
        def persistent_load(self, pid): return None
        def find_class(self, m, n):
            if m.startswith("torch") or "rebuild" in n or m == "collections":
                return _Skip
            try:
                return super().find_class(m, n)
            except Exception:
                return _Skip
    return U(io.BytesIO(pkl)).load()[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("BOOT_SCRATCH"),
                    help="output dir (default $BOOT_SCRATCH)")
    ap.add_argument("--skip-wu", action="store_true",
                    help="skip the 2.4 GB W_U memmap (asset self-test only)")
    a = ap.parse_args()
    if not a.out:
        sys.exit("FATAL: set $BOOT_SCRATCH or pass --out")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- vocab: id-indexed display-form list ----
    vdict = _plain(LENS_ING, "vocab")                    # {int id: str}
    if len(vdict) != V_SIZE:
        sys.exit(f"FATAL: vocab size {len(vdict)} != {V_SIZE}")
    vocab = [vdict[i] for i in range(V_SIZE)]
    (out / "anchor_vocab.json").write_text(json.dumps(vocab))
    inv = {}
    for i in range(V_SIZE):
        inv.setdefault(vocab[i], i)

    # ---- emotion_token_ids: single-token ' word' forms ----
    emotions = _plain(ANCHOR_VEC, "emotions")            # 171 names
    emo_tid = {e: inv[" " + e] for e in emotions if (" " + e) in inv}
    (out / "emotion_token_ids.json").write_text(json.dumps(emo_tid))

    # ---- anchor ln_final: data/1 = 4608 f32 ----
    z = zipfile.ZipFile(LENS_ING)
    ln_raw = z.read(next(n for n in z.namelist() if n.endswith("data/1")))
    ln = np.frombuffer(ln_raw, np.float32)
    if ln.shape[0] != D_MODEL:
        sys.exit(f"FATAL: ln_final len {ln.shape[0]} != {D_MODEL}")
    np.save(out / "anchor_ln_final.npy", ln.astype(np.float32))

    # ---- anchor W_U f16 memmap: data/0 token-major (V_SIZE, d) f32 ----
    wu = "SKIPPED"
    if not a.skip_wu:
        name = next(n for n in z.namelist() if n.endswith("data/0"))
        mm = np.lib.format.open_memmap(out / "anchor_WU_f16.npy", mode="w+",
                                       dtype=np.float16, shape=(V_SIZE, D_MODEL))
        row_b = D_MODEL * 4
        with z.open(name) as f:
            t = 0
            while t < V_SIZE:
                blk = min(20000, V_SIZE - t)
                rows = np.frombuffer(f.read(blk * row_b), np.float32).reshape(blk, D_MODEL)
                mm[t:t + blk] = rows.astype(np.float16)
                t += blk
        mm.flush()
        wu = f"({V_SIZE},{D_MODEL}) f16 written"

    print(f"[preflight] out={out}")
    print(f"  anchor_vocab.json      : {len(vocab)} tokens (id-indexed display form)")
    print(f"  emotion_token_ids.json : {len(emo_tid)}/{len(emotions)} single-token emotions")
    print(f"  anchor_ln_final.npy    : {tuple(ln.shape)} f32  (eps note: gemma rms, applied in lens)")
    print(f"  anchor_WU_f16.npy      : {wu}")
    for e in ("happy", "guilty", "loving", "proud"):
        if e in emo_tid:
            print(f"    tid[{e}]={emo_tid[e]}  vocab[tid]={vocab[emo_tid[e]]!r}")


if __name__ == "__main__":
    main()
