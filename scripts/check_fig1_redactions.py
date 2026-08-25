#!/usr/bin/env python3
"""Resolve every REDACT span in the marked review file to token index ranges.

v2 matcher: whitespace-insensitive. Each row's token window is reconstructed
from the per-token arrays, all whitespace is stripped from both the window text
and the span text, and the span is located in the resulting character stream.
Token boundaries are then recovered from a char->token map, so redactions
always cover whole tokens.

Also strips the review file's separator hashes and >>peak<< markers, and
ignores paragraph pilcrows.

Writes data/heldout/maxact/fig1_selection.json (the curation record).
Prints unresolved spans in full, plus the resolved token text for each span so
the reviewer can verify what will actually be blacked out.

Run from project root:
    python3 scripts/check_fig1_redactions.py "path/to/marked_review.txt" [--show]
"""
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

BASE = Path("data/heldout/maxact")
PANEL = ["afraid", "angry", "calm", "desperate", "guilty", "happy",
         "inspired", "loving", "nervous", "proud", "sad", "surprised"]
CTX = 30


def squash(s):
    """Strip peak markers, pilcrows, separator hashes, and ALL whitespace.
    Normalize curly quotes. Returns a dense character string for matching."""
    s = s.replace(">>", "").replace("<<", "")
    s = s.replace("\u00b6", "")
    s = re.sub(r"#{4,}", "", s)
    s = re.sub(r"-{4,}", "", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", "", s)


def parse_review(path):
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"-{20,}\n", text)
    rows = []
    for b in blocks:
        m = re.search(r"\[(\w+) rank (\d+)\]\s+value ([\d.]+)\s+source (\w+)\s+doc (\S+)", b)
        if not m:
            continue
        mk = re.search(r"MARK:\s*(.*)", b)
        rd = re.search(r"REDACT:(.*?)(?=\n\[|\Z)", b, re.S)
        rows.append({
            "emotion": m.group(1), "rank": int(m.group(2)),
            "value": float(m.group(3)), "source": m.group(4), "doc_id": m.group(5),
            "mark": (mk.group(1).strip().lower() if mk else ""),
            "redact_raw": (rd.group(1).strip() if rd else ""),
        })
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show" in sys.argv
    review = args[0] if args else str(BASE / "fig1_review.txt")
    rows = parse_review(review)
    print("parsed %d rows from %s\n" % (len(rows), review))

    r = json.loads((BASE / "pass1_results.json").read_text())
    emotions = list(r["by_emotion"].keys())
    pt_dir = BASE / "pertoken"
    tar = tarfile.open(BASE / "pertoken.tar") if (BASE / "pertoken.tar").exists() else None

    def load_npz(doc_id):
        p = pt_dir / (doc_id + ".npz")
        if p.exists():
            return np.load(p, allow_pickle=True)
        return np.load(tar.extractfile("pertoken/%s.npz" % doc_id), allow_pickle=True)

    unresolved, out = [], {}
    total_spans = total_ok = 0
    for emo in PANEL:
        e_idx = emotions.index(emo)
        wins = r["by_emotion"][emo]
        keep_rows = []
        for row in [x for x in rows if x["emotion"] == emo]:
            if row["mark"] not in ("keep", "redact"):
                continue
            w = wins[row["rank"] - 1]
            z = load_npz(w["doc_id"])
            toks = [str(t) for t in z["tokens"]]
            pos, seq = w["pos"], len(toks)
            lo, hi = max(1, pos - CTX), min(seq, pos + CTX + 1)
            win_toks = toks[lo:hi]

            # dense char stream + char->token index map
            dense, char2tok = "", []
            for k, t in enumerate(win_toks):
                st = squash(t)
                dense += st
                char2tok.extend([k] * len(st))

            spans_out = []
            if row["mark"] == "redact":
                pieces = [p for p in (squash(x) for x in row["redact_raw"].split("+")) if p]
                for piece, orig in zip(pieces, [x.strip() for x in row["redact_raw"].split("+") if squash(x)]):
                    total_spans += 1
                    i = dense.find(piece)
                    if i < 0 and len(piece) > 12:      # tolerate a clipped tail
                        i = dense.find(piece[:-4])
                        piece = piece[:-4] if i >= 0 else piece
                    if i < 0:
                        unresolved.append((emo, row["rank"], orig))
                        continue
                    total_ok += 1
                    j = i + len(piece) - 1
                    a, b = char2tok[i], char2tok[j]
                    spans_out.append([a, b])
                    if show:
                        covered = "".join(win_toks[a:b + 1])
                        print("  [%s r%02d] tokens %d-%d -> %r" % (emo, row["rank"], a, b, covered[:90]))
            keep_rows.append({
                "rank": row["rank"], "mark": row["mark"], "doc_id": row["doc_id"],
                "source": row["source"], "value": row["value"],
                "peak_index_in_row": pos - lo,
                "redact_token_spans": spans_out,
            })
        out[emo] = keep_rows
        nk = sum(1 for x in keep_rows if x["mark"] == "keep")
        docs = len({x["doc_id"] for x in keep_rows})
        print("%-11s usable %2d (keep %d, redact %d) | distinct docs %d"
              % (emo, len(keep_rows), nk, len(keep_rows) - nk, docs))

    print("\nspans: %d resolved of %d" % (total_ok, total_spans))
    if unresolved:
        print("\nUNRESOLVED (%d) -- full text, fix in the review file:" % len(unresolved))
        for emo, rank, piece in unresolved:
            print("  [%s rank %02d]\n      %s" % (emo, rank, piece))
    else:
        print("ALL REDACTION SPANS RESOLVED")

    path = BASE / "fig1_selection.json"
    path.write_text(json.dumps({"context_tokens": CTX, "panels": out}, indent=1))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
