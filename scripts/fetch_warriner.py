#!/usr/bin/env python3
"""
fetch_warriner.py — retrieve the Warriner et al. (2013) VAD norms into the local cache.

The Warriner norms are third-party data under a non-commercial license and are NOT
redistributed with this repository (the local cache is .gitignored). This script fetches
the published supplementary table directly from the Center for Reading Research (CRR,
Ghent University) archive under the user's acceptance of the source terms, and writes it
VERBATIM to the local cache path the analysis code expects. No transformation is applied
on disk; canonicalization to (word, valence, arousal) happens at read time in
scripts/analyze_sun_choi_engagement.py (#4 norm-source block).

Companion provenance: data/ratings/warriner_cache.PROVENANCE.txt

License / terms (from the source):
    Free for non-commercial research use WITH CITATION (recorded as CC BY-NC-ND; confirm
    the exact license at the source before relying on it). Do NOT redistribute the dataset.
    Direct others to the CRR archive:  http://crr.ugent.be/archives/1003
    Cite: Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). "Norms of valence,
    arousal, and dominance for 13,915 English lemmas." Behavior Research Methods,
    45(4), 1191-1207. doi:10.3758/s13428-012-0314-x
Running this script fetches the data for your own use under those terms.

Usage:
    python scripts/fetch_warriner.py            # writes data/ratings/warriner_cache.csv
    python scripts/fetch_warriner.py --out PATH
    python scripts/fetch_warriner.py --url URL  # if the primary mirror has moved

Stdlib only (urllib, hashlib, csv). No third-party dependencies.
"""
import argparse
import csv
import hashlib
import io
import sys
import urllib.request

# Primary published mirror of the supplementary table (13,915 lemmas + header).
# If this 404s, obtain the file from the CRR archive page and pass it with --url or place
# it directly at the --out path; the archive landing page is:
#   http://crr.ugent.be/archives/1003
PRIMARY_URL = "http://crr.ugent.be/papers/Ratings_Warriner_et_al.csv"
OUT_DEFAULT = "data/ratings/warriner_cache.csv"
EXPECTED_LEMMAS = 13915
REQUIRED_COLS = ("Word", "V.Mean.Sum", "A.Mean.Sum")  # the three the project reads


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Warriner et al. (2013) VAD norms.")
    ap.add_argument("--out", default=OUT_DEFAULT, help=f"output CSV path (default {OUT_DEFAULT})")
    ap.add_argument("--url", default=PRIMARY_URL, help="source CSV URL")
    args = ap.parse_args()

    print("The Warriner norms are non-commercial licensed third-party data; see the header")
    print("of this script for terms. Fetching for your own use under those terms.\n")
    print(f"[get] {args.url}")
    try:
        with urllib.request.urlopen(args.url) as r:  # noqa: S310 (documented official source)
            raw = r.read()
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[err] download failed: {e}", file=sys.stderr)
        print("      The primary mirror may have moved. Obtain the supplementary CSV from the")
        print("      CRR archive and re-run with --url, or drop it at the --out path directly:")
        print("      http://crr.ugent.be/archives/1003", file=sys.stderr)
        return 2

    # Write verbatim (no on-disk transformation), then validate the columns the project needs.
    with open(args.out, "wb") as f:
        f.write(raw)
    out_sha = sha256_bytes(raw)

    text = raw.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        print("[err] downloaded file is empty", file=sys.stderr)
        return 3
    n_rows = sum(1 for _ in reader)

    print(f"[out] {args.out}  {len(raw):,} bytes  {n_rows:,} data rows  sha256 {out_sha}")
    missing = [c for c in REQUIRED_COLS if c not in header]
    if missing:
        print(f"[err] downloaded file is missing required column(s) {missing}.", file=sys.stderr)
        print(f"      Header seen: {header[:8]}{' ...' if len(header) > 8 else ''}", file=sys.stderr)
        print("      This may not be the standard Warriner supplementary CSV — verify the source.",
              file=sys.stderr)
        return 4
    if n_rows != EXPECTED_LEMMAS:
        print(f"[note] row count {n_rows:,} != expected {EXPECTED_LEMMAS:,} "
              f"(verify the mirror; correlations are scale/row-count robust).")
    print("[ok ] verbatim cache written; required columns present "
          f"({', '.join(REQUIRED_COLS)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
