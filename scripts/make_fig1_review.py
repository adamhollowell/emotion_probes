#!/usr/bin/env python3
"""Build the manual curation review file for the Figure 1 safe-subset attempt.

Writes data/heldout/maxact/fig1_review.txt: all 20 top windows for each of
the 12 panel emotions, in rank order, with the peak token marked, ready for
manual marking. NOTHING is filtered or hidden -- the reviewer sees every window
and makes every call.

Marking protocol (edit the MARK: line in place, save the file):
  MARK: keep      -> row is printable as-is
  MARK: drop      -> row is excluded entirely
  MARK: redact    -> row shown with spans blacked out; note the spans after it
  MARK: ?         -> undecided (default; unmarked rows count as undecided)

Run from project root: python3 scripts/make_fig1_review.py
"""
import json
import tarfile
import textwrap
from pathlib import Path

import numpy as np

BASE = Path("data/heldout/maxact")
PANEL = ["afraid", "angry", "calm", "desperate", "guilty", "happy",
         "inspired", "loving", "nervous", "proud", "sad", "surprised"]
CTX = 30
WRAP = 100


def main():
    r = json.loads((BASE / "pass1_results.json").read_text())
    emotions = list(r["by_emotion"].keys())

    pt_dir = BASE / "pertoken"
    tar = tarfile.open(BASE / "pertoken.tar") if (BASE / "pertoken.tar").exists() else None

    def load_npz(doc_id):
        p = pt_dir / (doc_id + ".npz")
        if p.exists():
            return np.load(p, allow_pickle=True)
        return np.load(tar.extractfile("pertoken/%s.npz" % doc_id), allow_pickle=True)

    out = []
    out.append("FIGURE 1 CURATION REVIEW -- 12 panel emotions, top-20 windows each")
    out.append("Source: pass1_results.json (v2 held-out sweep). Nothing filtered; all windows shown.")
    out.append("")
    out.append("Mark each row by editing its MARK: line to one of: keep / drop / redact / ?")
    out.append("For redact, add a REDACT: line naming the words or spans to black out.")
    out.append("Peak token is wrapped in >>double angle brackets<<.")
    out.append("Rows repeat when one document wins several ranks; note that when deciding.")
    out.append("=" * WRAP)

    counts = {}
    for emo in PANEL:
        e_idx = emotions.index(emo)
        out.append("")
        out.append("#" * WRAP)
        out.append("# EMOTION: %s   (20 windows, rank order)" % emo.upper())
        out.append("#" * WRAP)
        seen_docs = {}
        for w in r["by_emotion"][emo]:
            z = load_npz(w["doc_id"])
            toks = [str(t) for t in z["tokens"]]
            pos, seq = w["pos"], len(toks)
            lo, hi = max(1, pos - CTX), min(seq, pos + CTX + 1)
            marked = list(toks[lo:hi])
            marked[pos - lo] = ">>%s<<" % marked[pos - lo].strip()
            text = "".join(marked).replace("\n", " ¶ ")
            rank = r["by_emotion"][emo].index(w) + 1
            dup = seen_docs.get(w["doc_id"])
            seen_docs.setdefault(w["doc_id"], rank)
            out.append("")
            out.append("-" * WRAP)
            out.append("[%s rank %02d]  value %.1f  source %s  doc %s%s"
                       % (emo, rank, w["value"], str(z["source"]), w["doc_id"],
                          "   (SAME DOC as rank %d)" % dup if dup else ""))
            out.append("")
            for line in textwrap.wrap(text, WRAP - 4):
                out.append("    " + line)
            out.append("")
            out.append("MARK: ?")
            out.append("REDACT:")
        counts[emo] = len(r["by_emotion"][emo])

    path = BASE / "fig1_review.txt"
    path.write_text("\n".join(out), encoding="utf-8")
    print("wrote %s" % path)
    print("emotions: %d | rows: %d" % (len(PANEL), sum(counts.values())))
    print("\nunique documents per emotion (repeats mean one doc won several ranks):")
    for emo in PANEL:
        docs = {w["doc_id"] for w in r["by_emotion"][emo]}
        print("  %-11s %2d unique docs of 20 windows" % (emo, len(docs)))


if __name__ == "__main__":
    main()
