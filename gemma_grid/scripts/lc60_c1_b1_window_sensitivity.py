#!/usr/bin/env python3
"""
LC-60 Commission 1 — B1 floor-window sensitivity (100T).

PURE ARITHMETIC on the already-shipped per-layer log-rank curve y(l)=log10(median_rank(l)).
Do NOT recompute onset curves, do NOT call the frozen estimator, do NOT touch neutrals/vectors.

Re-derives the knee from the STORED curve under two floor windows:
  W4 = [42,45]  (shipped 4-layer clamped window)
  W5 = [41,45]  (counterfactual: deepest FIVE available layers; honest analogue of nominal [42,46])

Self-consistency gate: under W4 must reproduce the shipped knees exactly, else STOP.

Read-only on all inputs; writes ONE new artifact + manifest.
"""
import json
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.normpath(os.path.join(HERE, "..", "geometry_out"))

CURVE_PATH = os.path.join(GEO, "b1_onsetcurve.json")
KNEE_PATH = os.path.join(GEO, "b1_onsetcurve_knee.json")
OUT_PATH = os.path.join(GEO, "b1_window_sensitivity.json")

# Expected input SHAs (from the dispatch, verified on disk before use)
EXPECT = {
    CURVE_PATH: "f7b4f593bd89d040db1631640ca89e8c5b34e079ca80d6a8bde581183269826d",
    KNEE_PATH: "713c494057137d670a19e892362102f03fa17aabd7c2e6a3bfc753575ebd9eff",
}

PLATEAU_WINDOW = (0, 4)   # UNCHANGED across conditions
WINDOWS = {
    "W4": (42, 45),       # shipped 4-layer clamped window
    "W5": (41, 45),       # counterfactual deepest-5 (honest analogue of nominal [42,46])
}

# Shipped W4 values to reproduce exactly (self-consistency gate)
SHIPPED = {
    "gemma-2-27b-base": {"frac": 25.668705160828218, "nearest": 26, "branch": "DEEPER"},
    "gemma-2-27b-it":   {"frac": 25.46715980109999,  "nearest": 25, "branch": "COINCIDENT"},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def branch_of(nearest):
    if nearest <= 21:
        return "SHALLOWER"
    if nearest >= 26:
        return "DEEPER"
    return "COINCIDENT"  # {22,23,24,25}


def window_mean(y, lo, hi):
    idx = [l for l in range(lo, hi + 1)]
    for l in idx:
        if l not in y:
            raise SystemExit(f"STOP: floor/plateau index {l} absent from curve (window [{lo},{hi}])")
    return sum(y[l] for l in idx) / len(idx)


def first_descent_crossing(y, layers, Y_M):
    """Linear interpolation in (l, y) between the two sampled layers bracketing
    the FIRST descent crossing of Y_M (y(l) >= Y_M > y(l+1))."""
    for i in range(len(layers) - 1):
        l0, l1 = layers[i], layers[i + 1]
        y0, y1 = y[l0], y[l1]
        if y0 >= Y_M > y1:
            frac = l0 + (y0 - Y_M) / (y0 - y1)
            return frac, l0, l1
    raise SystemExit("STOP: no descent crossing of Y_M found")


def nearest_sampled(frac, layers):
    return min(layers, key=lambda l: (abs(l - frac), l))


def steepest_descent(y, layers):
    best_lower, best_drop = None, None
    for i in range(len(layers) - 1):
        l0, l1 = layers[i], layers[i + 1]
        drop = -(y[l1] - y[l0])
        if best_drop is None or drop > best_drop:
            best_drop, best_lower = drop, l0
    return best_lower, best_lower + 0.5


def main():
    # ---- verify input SHAs (read-only gate) ----
    for path, exp in EXPECT.items():
        got = sha256_file(path)
        if got != exp:
            raise SystemExit(f"STOP: sha mismatch {path}\n  expected {exp}\n  got      {got}")

    with open(CURVE_PATH) as f:
        curve = json.load(f)
    with open(KNEE_PATH) as f:
        knee = json.load(f)

    results = {
        "record": "LC-60 C1 — B1 floor-window sensitivity: knee branch robustness under W4=[42,45] vs W5=[41,45] (100T).",
        "spec_sha256": knee["spec_sha256"],
        "authorizing": "MEMO LC-60, Commission 1 (dispatch RUNNER_DISPATCH_LC60_C1_B1_WINDOW_SENSITIVITY_100T.md)",
        "method": "Pure arithmetic on stored y(l)=log10(median_rank(l)); no onset recompute, no estimator call.",
        "plateau_window": list(PLATEAU_WINDOW),
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "W5_realization": "W5=[41,45] pinned to the DEEPEST FIVE available layers on the 46-layer 27B (L0..L45); "
                          "honest 5-real-layer analogue of the nominal [42,46]. If the Ledger intended a different "
                          "5-layer set, that is a one-line correction (flagged, not guessed wider).",
        "input_shas": {os.path.basename(p): EXPECT[p] for p in EXPECT},
        "cells": {},
        "self_consistency_W4": {},
        "flip_verdict": None,
    }

    all_pass = True
    any_flip = False

    for cell in ["gemma-2-27b-base", "gemma-2-27b-it"]:
        yj = knee["cells"][cell]["y_log10_rank"]
        y = {int(k): float(v) for k, v in yj.items()}
        layers = sorted(y.keys())

        Y_P = window_mean(y, *PLATEAU_WINDOW)

        cell_out = {"plateau_Y_P": Y_P, "windows": {}}
        for wname, (lo, hi) in WINDOWS.items():
            Y_F = window_mean(y, lo, hi)
            Y_M = (Y_P + Y_F) / 2.0
            frac, b0, b1 = first_descent_crossing(y, layers, Y_M)
            nearest = nearest_sampled(frac, layers)
            branch = branch_of(nearest)
            sd_lower, sd_mid = steepest_descent(y, layers)
            cell_out["windows"][wname] = {
                "floor_window": [lo, hi],
                "Y_F": Y_F,
                "Y_M": Y_M,
                "primary_knee_interp_frac": frac,
                "bracket": [b0, b1],
                "primary_knee_nearest_sampled": nearest,
                "branch": branch,
                "secondary_knee_lower_layer": sd_lower,
                "secondary_knee_frac_mid": sd_mid,
                "signed_frac_distance_from_seam": frac - 23,
            }
        results["cells"][cell] = cell_out

        # ---- self-consistency gate on W4 ----
        w4 = cell_out["windows"]["W4"]
        exp = SHIPPED[cell]
        frac_ok = abs(w4["primary_knee_interp_frac"] - exp["frac"]) < 1e-9
        near_ok = w4["primary_knee_nearest_sampled"] == exp["nearest"]
        br_ok = w4["branch"] == exp["branch"]
        cell_pass = frac_ok and near_ok and br_ok
        all_pass = all_pass and cell_pass
        results["self_consistency_W4"][cell] = {
            "shipped_frac": exp["frac"], "rederived_frac": w4["primary_knee_interp_frac"],
            "frac_match_1e-9": frac_ok,
            "shipped_nearest": exp["nearest"], "rederived_nearest": w4["primary_knee_nearest_sampled"],
            "nearest_match": near_ok,
            "shipped_branch": exp["branch"], "rederived_branch": w4["branch"], "branch_match": br_ok,
            "PASS": cell_pass,
        }

        # ---- flip check: W4 branch vs W5 branch ----
        if cell_out["windows"]["W4"]["branch"] != cell_out["windows"]["W5"]["branch"]:
            any_flip = True

    results["self_consistency_W4"]["ALL_PASS"] = all_pass
    if not all_pass:
        results["flip_verdict"] = "ABORT: W4 self-consistency FAILED — re-derivation is wrong; do not trust W5."
        print(json.dumps(results, indent=2))
        raise SystemExit("STOP: W4 self-consistency gate FAILED (see printed results). Not writing artifact.")

    results["flip_verdict"] = (
        "FLIP: a branch assignment changed under W5 — STOP, do not integrate, return to Adam."
        if any_flip else
        "NO FLIP: both cells keep their branch under W5 — assignments are window-robust; feeds Phase-5 unchanged."
    )
    results["any_flip"] = any_flip

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", OUT_PATH)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
