#!/usr/bin/env python3
"""
verify_pertoken_stripped.py — confirm the released pertoken_stripped.tar contains NO
readable document text.

This is the "released scan" referenced in the Reproducibility statement: an independent
check (separate from the stripping step) that opens the shipped archive and certifies that
the readable-text field was removed and that no per-token string/object array survives in
any record. It does NOT false-flag the intentionally-kept metadata scalars (`source`,
`doc_id`), which the release documents as retained.

Definition of a readable-text field (what must NOT remain):
  * the named `tokens` array (the decoded per-token strings that reconstruct the passage), and
  * any array of string/bytes/object dtype with ndim >= 1 (a per-token text vector).
Allowed: 0-d unicode scalars `source` (corpus label) and `doc_id` (identifier), plus all
numeric arrays (`token_ids`, `proj`, `proj_rand`, `norms`, `layer`, ...).

Exit 0 and print CLEAN if no readable text remains; exit 1 and list offenders otherwise.

Usage:
    python scripts/verify_pertoken_stripped.py [TAR]
Defaults to data/heldout/maxact/pertoken_stripped.tar. Stdlib + numpy only; no GPU, no torch.
"""
import io
import sys
import tarfile
import hashlib
import numpy as np

TAR_DEFAULT = "data/heldout/maxact/pertoken_stripped.tar"
FORBIDDEN_NAMED = {"tokens"}          # the readable-text field that must be gone
TEXT_KINDS = {"U", "S", "O"}          # unicode / bytes / python-object
ALLOWED_TEXT_SCALARS = {"source", "doc_id"}  # documented, retained; 0-d only


def is_readable_text(name, arr):
    """True iff `arr` is a readable-text field that should not be present."""
    if name in FORBIDDEN_NAMED:
        return True
    if arr.dtype.kind in TEXT_KINDS:
        # allow the two documented 0-d metadata scalars; flag any string ARRAY
        if name in ALLOWED_TEXT_SCALARS and arr.ndim == 0:
            return False
        return True
    return False


def main() -> int:
    tar_path = sys.argv[1] if len(sys.argv) > 1 else TAR_DEFAULT

    # Report the archive's own sha256 (the shipped file's hash).
    h = hashlib.sha256()
    with open(tar_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"[tar] {tar_path}  sha256 {h.hexdigest()}")

    n_docs = 0
    offenders = []          # (member, field, dtype, shape)
    fields_seen = set()
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".npz"):
                continue
            n_docs += 1
            z = np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=True)
            for name in z.files:
                arr = z[name]
                fields_seen.add(name)
                if is_readable_text(name, arr):
                    offenders.append((m.name, name, str(arr.dtype), arr.shape))

    print(f"[scan] {n_docs} per-document records scanned")
    print(f"[scan] fields present: {', '.join(sorted(fields_seen))}")
    if offenders:
        print(f"[FAIL] {len(offenders)} readable-text field(s) found:", file=sys.stderr)
        for member, name, dt, shp in offenders[:20]:
            print(f"         {member}: '{name}' dtype={dt} shape={shp}", file=sys.stderr)
        return 1
    print("[CLEAN] no readable-text field remains: 'tokens' absent in every record, and no "
          "per-token string/object array present (only documented scalar labels source/doc_id).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
