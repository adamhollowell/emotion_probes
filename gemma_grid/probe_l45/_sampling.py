"""Shared, dependency-light probe sampler (numpy + the 171 emotion list only —
no torch/transformer_lens), so both the extraction (GPU env) and the dry-run
(analysis env) draw IDENTICAL story positions."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from emotion_probes.emotions import EMOTIONS_171   # noqa: E402  (pure list, no heavy deps)

PROBE_SEED = 0
N_STORIES = 120


def probe_positions(avail: dict, seed: int = PROBE_SEED, n: int = N_STORIES) -> dict:
    """Deterministic seeded draw of n positions per emotion, one shared RNG
    consumed in EMOTIONS_171 order. Reproducible from (seed, per-emotion count)
    alone — identical whether called from the extractor or the dry-run."""
    rng = np.random.default_rng(seed)
    out = {}
    for e in EMOTIONS_171:
        if e not in avail:
            continue
        a = avail[e]; k = min(n, a)
        out[e] = sorted(int(x) for x in rng.choice(a, k, replace=False))
    return out
