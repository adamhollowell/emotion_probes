#!/usr/bin/env python3
"""Story-only extract of the bf16 against fp32 precision spot-check (§8.4 and Appendix J).

Written for the release on 2026-10-26. The spot-check re-extracted 40 stories for each of the 171
emotions at each size's analysis layer (2B layer 17, 9B layer 28, 27B layer 31) in bfloat16 and in
float32, and a summary recorded the agreement. This extract keeps the story measurements only. They
are raw story activations, so no cleanup basis enters them.

Per size it records the summary's story statistics as stored (cosine mean and minimum, relative
error mean and maximum over the 6,840 story vectors) and identifies two vectors:

  worst vector             the story vector with the lowest bf16 against fp32 cosine.
  largest relative error   the story vector with the largest ||bf16 - fp32|| / ||fp32||.

For each it records the emotion, the story's slot among that emotion's 40 sampled stories, the story's
position among that emotion's 1,200 stories in corpus order, and the top dimension of the difference
with its share, max_d (bf16_d - fp32_d)^2 / sum_d (bf16_d - fp32_d)^2.
The worst vector's share is the summary's worst_vec_topdim_share, and the script checks that it
reproduces it, and the stored statistics, to within 1e-6 before writing.

Two hardware records are added, story fields only. For 9B and 27B, cross_environment copies the story
fields of the spot-check file's stored truth_gate: the spot-check's bf16 story vectors (H200) against
the stored extraction (H100), as relative difference and cosine. For 9B, twin_run parses the twin-run
result line: the same 6,840 story vectors extracted twice on one accelerator, once with the forward
pass stopped at the layer read and once completed, compared bit for bit.

The spot-check activation files are not part of the release. Their sha256 values are in the manifest.

    python gemma_grid/scripts/precision_story_extract.py --spotcheck-dir <folder> --summary <summary json> \
        --twin-result <twin_9b_result.txt>
"""
import argparse, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "gemma_grid" / "geometry_out" / "precision_story_summary.json"
SIZES = ["2b", "9b", "27b"]
STORED = ["stories_cos_mean", "stories_cos_min", "stories_rel_mean", "stories_rel_max"]
TOL = 1e-6


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def describe(i, p, diff, k):
    e, slot = divmod(i, k)
    emo = p["emotions"][e]
    d2 = diff[i] ** 2
    return {"emotion": emo, "story_slot": slot, "story_index": int(p["story_indices"][emo][slot]),
            "topdim": int(d2.argmax()), "topdim_share": float(d2.max() / d2.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spotcheck-dir", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--twin-result", required=True)
    a = ap.parse_args()
    sd, summ = Path(a.spotcheck_dir), json.loads(Path(a.summary).read_text())
    out = {"analysis": "bf16 against fp32 precision spot-check, story vectors only",
           "definitions": {
               "worst_vector": "the story vector with the lowest bf16 against fp32 cosine",
               "largest_relative_error_vector": "the story vector with the largest ||bf16 - fp32|| / ||fp32||",
               "topdim_share": "max_d (bf16_d - fp32_d)^2 / sum_d (bf16_d - fp32_d)^2 for that vector",
               "stored_statistics": "copied from the spot-check summary as stored, over all story vectors"},
           "sizes": {}}
    inputs = {}
    for s in SIZES:
        f = sd / f"precision_spotcheck_{s}.pt"
        inputs[f.name] = sha(f)
        p = torch.load(f, map_location="cpu", weights_only=False)
        B, F = p["stories_bf16"], p["stories_fp32"]
        n_e, k, d = B.shape
        b, fl = B.reshape(-1, d), F.reshape(-1, d)
        diff = (b - fl).numpy()
        cos = torch.nn.functional.cosine_similarity(b, fl, dim=1)
        rel = (b - fl).norm(dim=1) / fl.norm(dim=1)
        rec = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
               "rel_mean": float(rel.mean()), "rel_max": float(rel.max())}
        S = summ[s]
        for key in STORED:
            if abs(rec[key.replace("stories_", "")] - S[key]) > TOL:
                sys.exit(f"STOP: {s} {key} recomputes to {rec[key.replace('stories_', '')]}, stored {S[key]}")
        worst = describe(int(cos.argmin()), p, diff, k)
        worst["cosine"] = float(cos.min())
        if abs(worst["topdim_share"] - S["worst_vec_topdim_share"]) > TOL:
            sys.exit(f"STOP: {s} worst-vector share {worst['topdim_share']}, stored {S['worst_vec_topdim_share']}")
        big = describe(int(rel.argmax()), p, diff, k)
        big["relative_error"] = float(rel.max())
        out["sizes"][s] = {"layer": p["layers"][0], "n_story_vectors": n_e * k, "width": d,
                           **{key: S[key] for key in STORED},
                           "worst_vec_topdim_share": S["worst_vec_topdim_share"],
                           "worst_vector": worst, "largest_relative_error_vector": big}
        if "truth_gate" in p:
            g = p["truth_gate"]
            if g["stories_rel_diff"] != S["truth_gate"]["stories_rel_diff"]:
                sys.exit(f"STOP: {s} truth_gate stories_rel_diff differs between the activation file and the summary")
            out["sizes"][s]["cross_environment"] = {
                "comparison": "spot-check bf16 story vectors (H200) against the stored extraction (H100), same stories",
                "stories_rel_diff": g["stories_rel_diff"], "stories_cos": g["stories_cos"],
                "stories_outliers_below_0.999": g["stories_outliers_below_0.999"]}
        print(f"{s}: worst vector {worst['emotion']} slot {worst['story_slot']} dim {worst['topdim']} share {worst['topdim_share']:.4f}; "
              f"largest relative error {big['emotion']} slot {big['story_slot']} dim {big['topdim']} share {big['topdim_share']:.4f}")
    line = Path(a.twin_result).read_text().strip()
    num = lambda pat: float(re.search(pat, line).group(1))
    bits = re.search(r"bit-differing vectors=(\d+)/(\d+)", line)
    out["sizes"]["9b"]["twin_run"] = {
        "comparison": "bf16 story vectors extracted twice on one H200, forward pass stopped at the layer read against completed, same rows",
        "n_vectors": int(bits.group(2)), "bit_differing_vectors": int(bits.group(1)),
        "rel_diff_max": num(r"rel_diff max=([0-9.e+-]+)"), "rel_diff_mean": num(r"rel_diff max=[0-9.e+-]+ mean=([0-9.e+-]+)"),
        "cos_min": num(r"cos min=([0-9.e+-]+)"), "cos_mean": num(r"cos min=[0-9.e+-]+ mean=([0-9.e+-]+)"),
        "record_line": line}
    if out["sizes"]["9b"]["twin_run"]["n_vectors"] != out["sizes"]["9b"]["n_story_vectors"]:
        sys.exit("STOP: twin-run vector count differs from the spot-check's")
    OUT.write_text(json.dumps(out, indent=1))
    man = {"output": str(OUT.relative_to(REPO)), "output_sha256": sha(OUT),
           "script": str(Path(__file__).resolve().relative_to(REPO)), "script_sha256": sha(Path(__file__).resolve()),
           "inputs": {**inputs, "summary": sha(a.summary), "twin_9b_result.txt": sha(a.twin_result)}, "tolerance_against_stored": TOL,
           "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    OUT.with_name(OUT.stem + "_manifest.json").write_text(json.dumps(man, indent=1))
    print("wrote", OUT, man["output_sha256"])


if __name__ == "__main__":
    main()
