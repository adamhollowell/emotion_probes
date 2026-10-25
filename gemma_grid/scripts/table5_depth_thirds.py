"""Table 5 of the paper: mean perturbation and mean activation shift, by third of each model's layers.

Thirds follow the paper's rule (§7.3): three consecutive thirds of near-equal size, leftover layers
added one each to the earliest thirds (numpy.array_split). 2B 0-8, 9-17, 18-25. 9B 0-13, 14-27,
28-41. 27B 0-15, 16-30, 31-45.

Rows:
  Perturbation, 2B / 9B / 27B: mean of (1 - RSA) over each third, from the `curve` in
    geometry_out/fig84_depth_tuning.json (the base-vs-tuned table correlation per layer).
  Perturbation, 27B without layer 45 / without layer 44 / without layers 44 and 45: the same, with
    the named layers removed from the late third only.
  Activation shift, 2B / 9B / 27B: mean of `per_layer_mean_abs_shift` over each third, from
    fixed_probe_tuning_shifts_results.json. That file's own `mean_abs_shift_by_third` uses
    floor(n_layers/3) thirds with the remainder in the late third and is NOT used here.

Every cell is formatted as printed in the paper (4 decimals, no leading zero) and compared with the
printed table embedded below. Any mismatch stops the run. Writes
geometry_out/table5_depth_thirds.json and its .MANIFEST.json (input and output hashes).
Run from the project root:  .venv/bin/python gemma_grid/scripts/table5_depth_thirds.py
"""
import hashlib, json, os, platform, sys, time
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG84 = os.path.join(REPO, "gemma_grid/geometry_out/fig84_depth_tuning.json")
SHIFTS = os.path.join(REPO, "gemma_grid/geometry_out/fixed_probe_tuning_shifts_results.json")
OUT = os.path.join(REPO, "gemma_grid/geometry_out/table5_depth_thirds.json")

# Table 5 as printed in the final draft, GGE1_DRAFT_v10_100T.md (sha256 c2b6740a...), early / middle / late.
PRINTED = {
    ("Perturbation", "2B"): (".0042", ".0055", ".0075"),
    ("Perturbation", "9B"): (".0034", ".0055", ".0073"),
    ("Perturbation", "27B"): (".0100", ".0142", ".0213"),
    ("Perturbation", "27B without layer 45"): (".0100", ".0142", ".0103"),
    ("Perturbation", "27B without layer 44"): (".0100", ".0142", ".0165"),
    ("Perturbation", "27B without layers 44 and 45"): (".0100", ".0142", ".0043"),
    ("Activation shift", "2B"): (".0122", ".0131", ".0072"),
    ("Activation shift", "9B"): (".0098", ".0050", ".0056"),
    ("Activation shift", "27B"): (".0135", ".0106", ".0152"),
}
SIZES = {"2B": ("2b", "2B"), "9B": ("9b", "9B"), "27B": ("27b", "27B")}
DROPS = {"27B without layer 45": [45], "27B without layer 44": [44], "27B without layers 44 and 45": [44, 45]}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def fmt(x):
    s = f"{x:.4f}"
    return s[1:] if s.startswith("0.") else s


def thirds(n):
    return [list(map(int, ix)) for ix in np.array_split(np.arange(n), 3)]


def main():
    fig84 = json.load(open(FIG84)); shifts = json.load(open(SHIFTS))
    rows, cells_ok = [], True
    for label, (fk, sk) in SIZES.items():
        curve = fig84[fk]["curve"]
        n = fig84[fk]["n_layers"]
        assert [c["layer"] for c in curve] == list(range(n)), f"{label}: curve layers not 0..n-1"
        pert = np.array([1.0 - c["rsa"] for c in curve])
        T = thirds(n)
        rows.append(("Perturbation", label, T, [float(pert[ix].mean()) for ix in T]))
        if label == "27B":
            for name, drop in DROPS.items():
                Td = [T[0], T[1], [l for l in T[2] if l not in drop]]
                rows.append(("Perturbation", name, Td, [float(pert[ix].mean()) for ix in Td]))
        m = np.array(shifts[sk]["fig84_analog"]["per_layer_mean_abs_shift"], dtype=np.float64)
        assert len(m) == n, f"{label}: shift profile has {len(m)} layers, curve has {n}"
        rows.append(("Activation shift", label, T, [float(m[ix].mean()) for ix in T]))
    rows.sort(key=lambda r: list(PRINTED).index((r[0], r[1])))
    out_rows = []
    for meas, cell, T, vals in rows:
        got = tuple(fmt(v) for v in vals)
        want = PRINTED[(meas, cell)]
        ok = got == want
        cells_ok &= ok
        print(f"{meas:17s} {cell:30s} {' '.join(got)}   {'ok' if ok else 'MISMATCH, paper ' + ' '.join(want)}")
        out_rows.append({"measure": meas, "cell": cell, "layers": [[t[0], t[-1]] for t in T],
                         "n_layers_per_third": [len(t) for t in T], "values": vals,
                         "printed": list(got), "matches_paper": ok})
    if len(rows) != len(PRINTED):
        sys.exit(f"STOP: built {len(rows)} rows, paper prints {len(PRINTED)}")
    if not cells_ok:
        sys.exit("STOP: at least one cell does not match the printed table")
    rel = lambda p: os.path.relpath(p, REPO)
    res = {"record": "Table 5, recomputed from stored per-layer values under the paper's thirds rule (§7.3).",
           "rule": "numpy.array_split(arange(n_layers), 3): consecutive thirds, leftover layers to the earliest thirds",
           "rows": out_rows, "all_27_cells_match_paper": True}
    json.dump(res, open(OUT, "w"), indent=1)
    man = {"script": rel(os.path.abspath(__file__)), "script_sha256": sha(os.path.abspath(__file__)),
           "inputs": {rel(FIG84): sha(FIG84), rel(SHIFTS): sha(SHIFTS)},
           "output": rel(OUT), "output_sha256": sha(OUT),
           "versions": {"python": platform.python_version(), "numpy": np.__version__},
           "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(man, open(OUT + ".MANIFEST.json", "w"), indent=1)
    print(f"all 27 cells match the paper. wrote {rel(OUT)} sha256 {man['output_sha256']}")


if __name__ == "__main__":
    main()
