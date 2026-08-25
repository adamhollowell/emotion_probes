#!/usr/bin/env python3
"""
strip_pertoken_text.py — produce a text-stripped copy of pertoken.tar for release.

Redistribution policy (readable text removed so the artifact can ship):
the per-token projection archive ships WITHOUT readable document text. Each per-document
.npz keeps its numeric fields (token_ids, proj, proj_rand, norms, source label, doc_id,
layer, skip_first, rand_seed, n_rand) and DROPS the single readable-text array `tokens`
(the decoded token strings — the only object/unicode field in the archive). token_ids are
integer ids, kept per the ruling; decoding them back to text requires a separate tokenizer
step and is out of scope of this artifact.

Reviewer-verifiable: asserts the input sha256 before reading, prints input and output
sha256, and removes exactly the field(s) in STRIP_FIELDS. Run:

    python scripts/strip_pertoken_text.py [IN_TAR] [OUT_TAR]

Defaults: IN=data/heldout/maxact/pertoken.tar  OUT=<same dir>/pertoken_stripped.tar
"""
import sys, io, tarfile, hashlib
import numpy as np

IN_DEFAULT  = "data/heldout/maxact/pertoken.tar"
OUT_DEFAULT = "data/heldout/maxact/pertoken_stripped.tar"
SRC_SHA256_EXPECTED = "0f59a745232061e2cd5000fa87512ff203ecb02c8d810b94174438857ceda3d1"
STRIP_FIELDS = {"tokens"}   # the only readable-text (object/unicode) array in each .npz


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else IN_DEFAULT
    out = sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT

    got = sha256(src)
    assert got == SRC_SHA256_EXPECTED, (
        f"input hash mismatch: {got} != expected {SRC_SHA256_EXPECTED} "
        f"— refusing to strip an unexpected source")
    print(f"[in ] {src}  sha256 {got}")

    n_docs = n_stripped = 0
    with tarfile.open(src, "r") as tin, tarfile.open(out, "w") as tout:
        for m in tin.getmembers():
            if not m.name.endswith(".npz"):
                # directory entries and any non-.npz members: copy verbatim
                tout.addfile(m, tin.extractfile(m) if m.isfile() else None)
                continue
            z = np.load(io.BytesIO(tin.extractfile(m).read()), allow_pickle=True)
            n_stripped += len(set(z.files) & STRIP_FIELDS)
            keep = {k: z[k] for k in z.files if k not in STRIP_FIELDS}
            buf = io.BytesIO()
            np.savez_compressed(buf, **keep)
            payload = buf.getvalue()
            info = tarfile.TarInfo(name=m.name)
            info.size = len(payload)
            info.mtime = 0            # fixed for reproducibility
            info.mode = m.mode
            info.type = m.type
            tout.addfile(info, io.BytesIO(payload))
            n_docs += 1

    out_sha = sha256(out)
    print(f"[out] {out}  sha256 {out_sha}")
    print(f"[ok ] {n_docs} .npz rewritten; {n_stripped} '{','.join(sorted(STRIP_FIELDS))}' "
          f"arrays removed; kept: token_ids, proj, proj_rand, norms, source, doc_id, "
          f"layer, skip_first, rand_seed, n_rand")


if __name__ == "__main__":
    main()
