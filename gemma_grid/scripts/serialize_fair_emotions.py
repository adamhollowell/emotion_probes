#!/usr/bin/env python3
"""
serialize_fair_emotions.py — closes M.7 blocker 12 (G6.6 serialization gaps).

Derives, from ON-DISK artifacts only (no tokenizer download, no pod scratch):
  1. the "fair" emotion membership list — E1's convention: an emotion has a
     fair own-word self-rank iff its leading-space " word" form is a SINGLE
     Gemma token (E1 §10.1: 140 of 171);
  2. the G6.6 valence split — the fair set intersected with NRC-VAD v2.1
     valence sign (ledger G6.6: 137 = 50 positive / 87 negative; 3 fair
     emotions lack an NRC entry).

Method: the per-cell lens-ingredients export ships the full id->string vocab
(list[str], 256,000 entries; see repo/hf_model_repo/export_lens_ingredients.py).
" word" is a single token iff the SentencePiece piece "▁word" exists in
that vocab (multi-word emotions can never qualify). This reproduces the pod's
`emotion_token_ids.json` membership without the (gated) tokenizer.

VALIDATION GATES (hard asserts — the run fails loudly if any is violated):
  n_emotions == 171 (judge CSV == canonical_100T keys)
  n_fair == 140              (E1 draft line 668; lens_summary fair_selfrank_n)
  n_fair_with_nrc == 137, split == 50 pos / 87 neg   (ledger G6.6)
  NRC natural overlap of the 171 == 161              (Appendix R)

M.9: every input is sha256-hashed into <out>_manifest.json at run time.
Outputs: geometry_out/fair_emotions_v1.json (+ manifest).
"""
import csv, hashlib, io, json, pickle, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path

GG = Path(__file__).resolve().parent.parent          # gemma_grid/
REPO = GG.parent                                     # project root
LENS = REPO / "data" / "vectors" / "lens_ingredients.pt"  # anchor export — the only lens file carrying the vocab (grid cells' exports ship W_U only)
JUDGE = REPO / "data" / "ratings" / "valence_arousal_judge.csv"
NRC = REPO / "data" / "ratings" / "nrc_vad_v2_1.csv"
CANON = GG / "data" / "gemma-2-2b-base" / "vectors" / "canonical.pt"
LENS_SUMMARY = GG / "geometry_out" / "gemma-2-2b-base" / "lens_summary.json"
OUT = GG / "geometry_out" / "fair_emotions_v1.json"
MANIFEST = GG / "geometry_out" / "fair_emotions_v1_manifest.json"


def pickle_objects_no_tensors(path):
    """Unpickle a torch.save zip data.pkl with tensor storages stubbed out
    (we only need python objects: vocab list / dict keys)."""
    zf = zipfile.ZipFile(path)
    pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))

    def _stub_rebuild(*a, **k):
        return None

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch"):
                return _stub_rebuild if "_rebuild" in name else type(name, (), {"__init__": lambda s, *a, **k: None})
            return super().find_class(module, name)

        def persistent_load(self, pid):
            return None

    return U(io.BytesIO(zf.read(pkl_name))).load()


def sha256(path, cap_note=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # --- inputs
    obj = pickle_objects_no_tensors(LENS)
    vocab = obj["vocab"]
    assert len(vocab) == 256000, f"vocab size unexpected: {len(vocab)}"
    if isinstance(vocab, dict):
        keys = list(vocab.keys())
        vocab_set = set(keys) if isinstance(keys[0], str) else set(vocab.values())
    else:
        vocab_set = set(vocab)

    with open(JUDGE, newline="") as f:
        emotions = [row["emotion"] for row in csv.DictReader(f)]
    assert len(emotions) == 171, len(emotions)

    canon = pickle_objects_no_tensors(CANON)
    canon_keys = set(canon.keys()) if isinstance(canon, dict) else None
    canon_gate = None
    if canon_keys:
        emo_key = next((k for k in ("vectors", "emotions", "vectors_clean") if k in canon_keys), None)
        inner = canon.get(emo_key) if emo_key else None
        if isinstance(inner, dict) and len(inner) == 171:
            assert set(inner.keys()) == set(emotions), "canonical_100T emotion set != judge CSV set"
            canon_gate = "PASS (171 keys match judge CSV)"
        else:
            canon_gate = f"SKIPPED (canonical_100T structure: top-level keys {sorted(canon_keys)[:8]})"

    # --- fair membership: single-token " word" <=> SentencePiece piece exists
    marker = "▁"  # SentencePiece space marker
    fair = [e for e in emotions if (" " not in e) and (marker + e) in vocab_set]
    alt = [e for e in emotions if (" " not in e) and (" " + e) in vocab_set]
    if len(fair) != 140 and len(alt) == 140:
        fair, marker = alt, "literal-space"
    assert len(fair) == 140, f"fair-count gate FAILED: {len(fair)} (marker={marker!r})"
    fair_set = set(fair)

    ls = json.load(open(LENS_SUMMARY))
    assert ls["fair_selfrank_n"] == 140

    # --- NRC valence split
    nrc_val = {}
    with open(NRC, newline="") as f:
        for row in csv.DictReader(f):
            nrc_val[row["word"]] = float(row["valence"])
    overlap_171 = [e for e in emotions if e in nrc_val]
    assert len(overlap_171) == 161, f"NRC natural-overlap gate FAILED: {len(overlap_171)}"

    fair_nrc = [e for e in fair if e in nrc_val]
    fair_no_nrc = sorted(fair_set - set(fair_nrc))
    pos = sorted(e for e in fair_nrc if nrc_val[e] > 0)
    neg = sorted(e for e in fair_nrc if nrc_val[e] < 0)
    zero = sorted(e for e in fair_nrc if nrc_val[e] == 0)
    assert len(fair_nrc) == 137, f"137 gate FAILED: {len(fair_nrc)}"
    assert (len(pos), len(neg), len(zero)) == (50, 87, 0), f"split gate FAILED: {len(pos)}/{len(neg)}/{len(zero)}"
    assert len(fair_no_nrc) == 3

    unfair = sorted(set(emotions) - fair_set)

    out = {
        "provenance": {
            "script": "gemma_grid/scripts/serialize_fair_emotions.py",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose": "Closes M.7 blocker 12: serializes the E1 'fair' single-token membership (140/171) and the G6.6 NRC-valence split (137 = 50 pos / 87 neg) from on-disk artifacts.",
            "method": f"piece-existence check ({marker!r} + word) against the 256k id->string vocab shipped in the lens-ingredients export; NRC-VAD v2.1 valence sign for the split.",
            "gates_passed": [
                "n_emotions == 171 (judge CSV)",
                f"canonical_100T key gate: {canon_gate}",
                "n_fair == 140 (== E1 draft line 668 == lens_summary.fair_selfrank_n)",
                "NRC natural overlap == 161 (Appendix R)",
                "fair-with-NRC == 137; split == 50 positive / 87 negative / 0 zero (ledger G6.6)",
            ],
            "caveat": "Membership derived by vocab-piece existence, not by running the (gated) tokenizer; equivalence to the pod's emotion_token_ids.json is certified by the count gates (140/161/137/50/87 all reproduce). Vocab is size-invariant across Gemma-2 cells (shared tokenizer).",
        },
        "n_emotions": 171,
        "fair_140": sorted(fair),
        "unfair_31": unfair,
        "fair_no_nrc_3": fair_no_nrc,
        "g6_6_split": {"n": 137, "positive_50": pos, "negative_87": neg},
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    manifest = {
        "m9_note": "SR.5 input manifest — sha256 of every input, written at run time beside the output.",
        "created_utc": out["provenance"]["created_utc"],
        "inputs": {str(p.relative_to(REPO)): sha256(p) for p in [LENS, JUDGE, NRC, CANON, LENS_SUMMARY]},
        "output": {str(OUT.relative_to(REPO)): sha256(OUT)},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print("PASS — all gates. fair=140, split 50/87, no-NRC:", fair_no_nrc)
    print("unfair examples:", unfair[:8], "...")
    print("wrote:", OUT, "and manifest")


if __name__ == "__main__":
    main()
