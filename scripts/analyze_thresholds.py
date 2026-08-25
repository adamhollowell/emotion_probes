#!/usr/bin/env python3
"""Figure-1 thresholds: per-emotion and pooled percentiles from histograms.npz.
Run from project root: python3 scripts/analyze_thresholds.py"""
import json
import numpy as np

HIST = "data/heldout/maxact/histograms.npz"
RESULTS = "data/heldout/maxact/pass1_results.json"
PANEL_12 = ["happy", "sad", "angry", "afraid", "calm", "desperate",
            "nervous", "surprised", "proud", "guilty", "inspired", "loving"]


def quantile_from_counts(counts, edges, q):
    """Exact-within-bin-width quantile from histogram counts."""
    c = np.cumsum(counts)
    total = c[-1]
    idx = int(np.searchsorted(c, q * total))
    idx = min(idx, len(counts) - 1)
    return edges[idx + 1]  # upper edge of the bin containing the quantile


def main():
    z = np.load(HIST, allow_pickle=True)
    counts = z["counts"]              # (171, bins)
    lo, hi, bins = float(z["lo"]), float(z["hi"]), int(z["bins"])
    emotions = list(z["emotions"])
    edges = np.linspace(lo, hi, bins + 1)
    r = json.load(open(RESULTS))

    # ---- integrity: clipping check ----
    edge_mass = int(counts[:, 0].sum() + counts[:, -1].sum())
    total = int(counts.sum())
    print("tokens histogrammed: %d (x171 emotions = %d cells)" % (total // 171, total))
    print("edge-bin mass (clipping): %d  -> %s\n"
          % (edge_mass, "CLEAN (percentiles exact to bin width %.1f)" % (edges[1] - edges[0])
             if edge_mass == 0 else "WARNING: range clipped; investigate"))

    # ---- per-emotion thresholds ----
    p90 = np.array([quantile_from_counts(counts[i], edges, 0.90) for i in range(len(emotions))])
    p99 = np.array([quantile_from_counts(counts[i], edges, 0.99) for i in range(len(emotions))])

    print("per-emotion p90 across 171 emotions: min=%.0f  median=%.0f  max=%.0f"
          % (p90.min(), np.median(p90), p90.max()))
    print("per-emotion p99 across 171 emotions: min=%.0f  median=%.0f  max=%.0f\n"
          % (p99.min(), np.median(p99), p99.max()))

    # ---- pooled convention ----
    pooled = counts.sum(axis=0)
    pooled_p90 = quantile_from_counts(pooled, edges, 0.90)
    pooled_p99 = quantile_from_counts(pooled, edges, 0.99)
    print("POOLED convention: p90=%.0f  p99=%.0f\n" % (pooled_p90, pooled_p99))

    # ---- the 12 panel emotions: threshold vs top value ----
    print("%-11s %8s %8s %10s %12s" % ("emotion", "p90", "p99", "top-1", "top/p90"))
    for e in PANEL_12:
        i = emotions.index(e)
        top = r["by_emotion"][e][0]["value"]
        print("%-11s %8.0f %8.0f %10.1f %11.1fx" % (e, p90[i], p99[i], top, top / max(p90[i], 1e-9)))

    print("\nReading: per-emotion p90 is the paper's shading floor. top/p90 shows how")
    print("far peaks sit above the floor -- large ratios mean crisp, selective")
    print("highlighting in the Figure 1 render; ratios near 1 would mean mush.")


if __name__ == "__main__":
    main()
