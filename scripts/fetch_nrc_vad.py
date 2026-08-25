#!/usr/bin/env python3
"""
fetch_nrc_vad.py — retrieve and canonicalize the NRC-VAD Lexicon v2.1.

The NRC-VAD lexicon is third-party data under a non-commercial license and is NOT
redistributed with this repository (its canonical local files are .gitignored). This
script fetches it directly from the official source under the user's acceptance of the
NRC terms, then canonicalizes it to the exact local layout the analysis code expects.

Companion provenance: data/ratings/nrc_vad_v2_1.PROVENANCE.txt
Consumer: scripts/analyze_sun_choi_engagement.py (#4 norm-source block)

License / terms (from the source):
    Free for non-commercial research and educational use WITH CITATION.
    Do NOT redistribute the raw data. Direct others to the official NRC-VAD homepage:
        https://saifmohammad.com/WebPages/nrc-vad.html
    Cite: Saif M. Mohammad. 2025. "NRC VAD Lexicon v2: Norms for Valence, Arousal, and
    Dominance for over 55k English Terms." arXiv:2503.23547.
Running this script fetches the data for your own use under those terms.

Usage:
    python scripts/fetch_nrc_vad.py            # writes data/ratings/nrc_vad_v2_1.csv
    python scripts/fetch_nrc_vad.py --out PATH

Stdlib only (urllib, zipfile, hashlib, csv). No third-party dependencies.
"""
import argparse
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile

SOURCE_ZIP = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
MEMBER = "Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"  # tab-separated: term,valence,arousal,dominance
OUT_DEFAULT = "data/ratings/nrc_vad_v2_1.csv"
EXPECTED_ROWS = 44728  # unigrams in v2.1 (sanity check, not a hard gate)


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch + canonicalize NRC-VAD v2.1.")
    ap.add_argument("--out", default=OUT_DEFAULT, help=f"output CSV path (default {OUT_DEFAULT})")
    ap.add_argument("--url", default=SOURCE_ZIP, help="source archive URL")
    args = ap.parse_args()

    print("NRC-VAD v2.1 is non-commercial licensed third-party data; see the header of this")
    print("script for terms. Fetching for your own use under those terms.\n")
    print(f"[get] {args.url}")
    try:
        with urllib.request.urlopen(args.url) as r:  # noqa: S310 (documented official source)
            raw = r.read()
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[err] download failed: {e}", file=sys.stderr)
        print("      If the URL has moved, get the archive from the official homepage:")
        print("      https://saifmohammad.com/WebPages/nrc-vad.html", file=sys.stderr)
        return 2
    print(f"[zip] {len(raw):,} bytes  sha256 {sha256_bytes(raw)}")

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member = MEMBER if MEMBER in z.namelist() else next(
            (n for n in z.namelist() if n.endswith("unigrams-NRC-VAD-Lexicon-v2.1.txt")), None)
        if member is None:
            print(f"[err] could not find the unigrams file in the archive; members:", file=sys.stderr)
            for n in z.namelist():
                print(f"        {n}", file=sys.stderr)
            return 3
        text = z.read(member).decode("utf-8")

    rows = []
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    header = next(reader)  # term,valence,arousal,dominance
    if [c.strip().lower() for c in header[:4]] != ["term", "valence", "arousal", "dominance"]:
        print(f"[warn] unexpected header {header!r}; proceeding on positional columns", file=sys.stderr)
    for row in reader:
        if len(row) < 4:
            continue
        word = row[0].strip().lower()
        rows.append((word, row[1].strip(), row[2].strip(), row[3].strip()))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word", "valence", "arousal", "dominance"])
        w.writerows(rows)

    with open(args.out, "rb") as f:
        out_sha = sha256_bytes(f.read())
    print(f"[out] {args.out}  {len(rows):,} rows  sha256 {out_sha}")
    if len(rows) != EXPECTED_ROWS:
        print(f"[note] row count {len(rows):,} != expected {EXPECTED_ROWS:,} "
              f"(source may have been revised; correlations are unaffected)")
    print("[ok ] canonicalized to word,valence,arousal,dominance (word lowercased).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
