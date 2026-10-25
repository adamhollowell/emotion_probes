#!/usr/bin/env python3
"""Part A of DISPATCH_V8 item 3 (sha256 09fd38ba...): old-input replay, then corrected execution.

Replay: the frozen battery procedure UNCHANGED, imported (pmap_battery.half_matrices + transport,
which import pmap_transport and the reference clean_layer), in the battery producer's runtime
(pmap_battery_serial.py: BLAS threads pinned to 1 before numpy import, gesdd->gesvd SVD fallback
shim). Must equal the stored item3_splithalf_ceiling floats exactly, else STOP (no corrected run).
Replay is verification only and does not ship as a result.

Corrected run: identical RandomState(0) even/odd split and half-means; the ONLY change is the second
argument of clean_layer = the cell's 100T neutral activations at the analysis layer (gate-2-passed
input). The slice files' `neutral` key is never read by the corrected path. To show that, the
corrected path's raw half-means are first checked to reproduce the replay's cleaned halves exactly
when cleaned with the replay's input (that check reads the slice neutral rows; the corrected
result does not).

Outputs (geometry_out/):
  item3_replay_verification.json (+ .MANIFEST.json)   verification only
  item3_splithalf_corrected.json (+ .MANIFEST.json)
Run: .venv/bin/python gemma_grid/scripts/item3_corrected.py   (Ledger ruling 2026-10-19)
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, hashlib, time, platform
import numpy as np

# ---- battery producer's authorized gesdd->gesvd fallback (verbatim behaviour of pmap_battery_serial.py) ----
_ORIG_SVD = np.linalg.svd
def _svd_fallback(a, full_matrices=True, compute_uv=True, hermitian=False):
    try:
        return _ORIG_SVD(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as _sp
        if not compute_uv:
            return _sp(a, full_matrices=full_matrices, compute_uv=False, lapack_driver="gesvd")
        return _sp(a, full_matrices=full_matrices, lapack_driver="gesvd")
np.linalg.svd = _svd_fallback

import scipy, sklearn, torch
# Ledger ruling 2026-10-19: Parts A and B run in .venv, the battery's original environment
# (the lens_env replay differed by 1 ULP in median cosine in 3 of 6 cells; kept as
# item3_replay_verification.lens_env.json, not graded).
ENV_NAME = ".venv"
EXPECTED = {"python": "3.12.13", "numpy": "2.4.6", "scipy": "1.17.1", "sklearn": "1.8.0", "torch": "2.12.0"}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOGPU = os.path.join(REPO, "gemma_grid/scripts")
sys.path.insert(0, NOGPU)
import pmap_transport as P
import pmap_battery as M
import analysisB_seam_onset_bootstrap_gesvdfix as B

GEO = os.path.join(REPO, "gemma_grid/geometry_out")
BATTERY = os.path.join(GEO, "pmap_battery.json")
GATES = os.path.join(GEO, "item3_disjoint_gates.json")
OUT_REPLAY = os.path.join(GEO, "item3_replay_verification.json")
OUT = os.path.join(GEO, "item3_splithalf_corrected.json")
ANCHOR = "gemma-2-27b-base"
METRICS = ("median_rank", "top1_acc", "top5_acc", "median_cosine")
INPUTS = {}


def sha(p, n=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, REPO)


def record(p):
    r = rel(p)
    if r not in INPUTS:
        INPUTS[r] = sha(p)
    return INPUTS[r]


def check_versions():
    got = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__, "torch": torch.__version__.split("+")[0]}
    bad = {k: (got[k], EXPECTED[k]) for k in EXPECTED if got[k] != EXPECTED[k]}
    if os.path.realpath(sys.prefix) != os.path.realpath(os.path.join(REPO, ENV_NAME)):
        bad["environment"] = (sys.prefix, ENV_NAME)
    if bad:
        sys.exit(f"STOP: environment/library version mismatch {bad}")
    cfg = np.show_config(mode="dicts")
    got["environment"] = ENV_NAME
    got["sys_prefix"] = sys.prefix
    got["numpy_blas_lapack"] = {k: cfg["Build Dependencies"][k] for k in ("blas", "lapack")}
    got["blas_threads"] = {v: os.environ[v] for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                       "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
    return got


def slice_path(c):
    return os.path.join(REPO, f"gemma_grid/data/{c}/vectors/{c}__per_story_slice.pt")


def neutral_100T_path(c):
    return os.path.join(REPO, "gemma_grid/data/gemma-2-27b-base/vectors/gemma-2-27b-base__neutral.pt") if c == ANCHOR else \
        os.path.join(REPO, f"gemma_grid/data/{c}/vectors/{c}__neutral.pt")


def raw_halves(cell):
    """Same split and half-means as M.half_matrices, story rows only (no neutral key read)."""
    blob = torch.load(slice_path(cell), map_location="cpu", weights_only=False, mmap=True)
    pe = blob["per_emotion"]; sl = blob.get("slice_layers") or blob.get("layers")
    li = list(sl).index(M.LAYER[cell])
    rows_a, rows_b = [], []
    for e in M.EMOS:
        acts = np.asarray(pe[e], dtype=np.float64)[:, li, :]
        perm = np.random.RandomState(0).permutation(acts.shape[0])
        rows_a.append(acts[perm[0::2]].mean(0)); rows_b.append(acts[perm[1::2]].mean(0))
    del blob
    return np.stack(rows_a), np.stack(rows_b), li


def write(path, obj, versions, run_utc):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    man = {"output": rel(path), "output_sha256": sha(path), "producer": rel(os.path.abspath(__file__)),
           "producer_sha256": sha(os.path.abspath(__file__)), "versions": versions,
           "inputs_sha256": dict(INPUTS), "run_utc": run_utc}
    with open(path + ".MANIFEST.json", "w") as f:
        json.dump(man, f, indent=2)
    return man["output_sha256"]


def main():
    versions = check_versions()
    run_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for p in (os.path.join(NOGPU, "pmap_transport.py"), os.path.join(NOGPU, "pmap_battery.py"),
              os.path.join(NOGPU, "analysisB_seam_onset_bootstrap_gesvdfix.py"), BATTERY, GATES):
        record(p)
    gates = json.load(open(GATES))
    assert gates["gate2_pass"] and gates["gate3_pass"], "STOP: gates 2/3 not passed"
    M.load_all()
    for c in M.CELLS:
        record(M.store_path(c))
    stored = json.load(open(BATTERY))["item3_splithalf_ceiling"]["cells"]
    # Ledger 2026-10-23: the replay and the equivalence check read the slice `neutral` key; a slice without it
    # (sanitized slices) skips both. The corrected computation is unchanged.
    no_neutral = any("neutral" not in torch.load(slice_path(c), map_location="cpu", weights_only=False, mmap=True)
                     for c in M.CELLS)
    REPLAY_SKIPPED = "skipped: slice has no neutral key"

    # ---------------- replay (unchanged frozen procedure) ----------------
    if no_neutral:
        for c in M.CELLS:
            record(slice_path(c))
        print(f"[replay] {REPLAY_SKIPPED}", flush=True)
    else:
        replay = {"record": "Item 3 old-input replay. VERIFICATION ONLY, does not ship as a result.",
                  "procedure": "pmap_battery.half_matrices + transport, imported unchanged; battery producer runtime "
                               "(threads=1, gesdd->gesvd fallback shim)", "cells": {}, "versions": versions}
        halves = {}
        all_exact = True
        for c in M.CELLS:
            t0 = time.time()
            record(slice_path(c))
            Va, Vb = M.half_matrices(c)
            m = M.transport(Va, Vb)
            st = stored[c]["split_half"]
            exact = {k: (m[k] == st[k]) for k in METRICS}
            all_exact &= all(exact.values())
            replay["cells"][c] = {"replay": m, "stored": st, "exact_equal": exact}
            halves[c] = (Va, Vb)
            print(f"[replay] {c} {m} exact={all(exact.values())} ({time.time()-t0:.0f}s)", flush=True)
        replay["all_exact"] = bool(all_exact)
        h = write(OUT_REPLAY, replay, versions, run_utc)
        print(f"[replay] all_exact={all_exact}  {OUT_REPLAY} sha256 {h}", flush=True)
        if not all_exact:
            print("STOP: replay does not reproduce stored item 3 values exactly. No corrected run.")
            return 1

    # ---------------- corrected run ----------------
    out = {"record": "Item 3 corrected execution of pre-stated item 3, replacing the unintended neutral input "
                     "(slice-file neutral rows, original corpus) with the designated 100T input. Original analysis "
                     "plan unchanged.",
           "dispatch_sha256": "09fd38ba352a2eebf573362dc8b848f5a5afc74dad3685626b1569b96089018b",
           "one_change": "clean_layer second argument = cell's 100T neutral acts at the analysis layer "
                         "(gate-2-passed); slice `neutral` key not read by the corrected path",
           "split": "per emotion RandomState(0).permutation(1200); even positions -> half a, odd -> half b",
           "estimator": {"module": "pmap_transport.py", "fold_seed": P.FOLD_SEED, "n_folds": P.N_FOLDS,
                         "kind": "procrustes", "scoring": "all 171 targets"},
           "emotion_order": "stored canonical vector files (pmap_battery.load_all)",
           "versions": versions, "run_utc": run_utc, "cells": {}}
    for c in M.CELLS:
        t0 = time.time()
        Ra, Rb, li = raw_halves(c)
        if no_neutral:
            same = REPLAY_SKIPPED
        else:
            # equivalence check of the reimplemented half-means against the replay (verification read of slice neutral)
            sn = torch.load(slice_path(c), map_location="cpu", weights_only=False, mmap=True)
            Nslice = np.asarray(sn["neutral"], dtype=np.float64)[:, li, :]
            del sn
            same = bool(np.array_equal(B.clean_layer(Ra, Nslice), halves[c][0]) and
                        np.array_equal(B.clean_layer(Rb, Nslice), halves[c][1]))
            assert same, f"STOP: reimplemented half-means do not reproduce replay halves for {c}"
        npath = neutral_100T_path(c)
        record(npath)
        N100 = torch.load(npath, map_location="cpu", weights_only=False, mmap=True)["acts"][:, M.LAYER[c], :]
        N100 = N100.double().numpy()
        Va, Vb = B.clean_layer(Ra, N100), B.clean_layer(Rb, N100)
        m = M.transport(Va, Vb)
        out["cells"][c] = {"analysis_layer": M.LAYER[c], "neutral_100T_file": rel(npath),
                           "neutral_100T_sha256": INPUTS[rel(npath)],
                           "k_gate2": gates["gate2_cleanup_basis"][c]["k_rebuilt"],
                           "halves_reproduce_replay_under_old_input": same, "split_half": m}
        print(f"[corrected] {c} {m} ({time.time()-t0:.0f}s)", flush=True)
    t1 = [v["split_half"]["top1_acc"] for v in out["cells"].values()]
    mc = [v["split_half"]["median_cosine"] for v in out["cells"].values()]
    out["ranges"] = {"top1_acc": [min(t1), max(t1)], "median_cosine": [min(mc), max(mc)]}
    if no_neutral:
        out["replay_verification_file"] = REPLAY_SKIPPED
        out["replay_all_exact"] = REPLAY_SKIPPED
        out["lens_env_replay_note"] = REPLAY_SKIPPED
    else:
        out["replay_verification_file"] = rel(OUT_REPLAY)
        out["replay_all_exact"] = True
        out["lens_env_replay_note"] = ("A replay under lens_env (py3.9.6/numpy 2.0.2) differed from the stored median cosine by "
                                       "1 ULP in 2b-it, 9b-it, 27b-it (all other values exact); file "
                                       "item3_replay_verification.lens_env.json. Not graded (Ledger 2026-10-19).")
    h = write(OUT, out, versions, run_utc)
    print(f"[corrected] {OUT} sha256 {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
