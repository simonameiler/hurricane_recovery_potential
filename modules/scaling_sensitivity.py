"""
scaling_sensitivity.py
-----------------------
Exact damage-state transition matrix (wind-only -> multi-hazard-scaled),
used by scripts/compute_scaling_valueadd.py to build Figure S9
(notebooks/probabilistic_analysis.ipynb).

Because the scaling exponent alpha_{e,c} is shared by every housing unit in
a given (event, county) pair, the monotonic transform L -> 1-(1-L)^alpha
preserves rank, so the transition matrix between the wind-only (g=0) and
default-scaled (g=1) damage-state histograms is fully determined by the two
marginal histograms via an interval-overlap (order-statistics) argument --
no per-building data or approximation needed.
"""

from __future__ import annotations

import numpy as np

# Hazus damage-state raw-loss-ratio bin edges (DS0..DS4), matching
# impact_utils.assign_damage_state exactly.
DS_EDGES = np.array([0.0, 0.02, 0.05, 0.10, 0.50, 1.0])
N_DS = 5  # DS0..DS4


def transition_matrix_rows(raw_counts: np.ndarray, scaled_counts: np.ndarray) -> np.ndarray:
    """Exact (frequency-unweighted) transition-count tensor, one 5x5 per row.

    Because alpha_{e,c} is shared by every housing unit in a (event, county)
    row, the monotonic transform preserves rank: raw DS bin i's units and
    scaled DS bin j's units are each a contiguous block in the same
    loss-sorted order, so their overlap size is exactly

        T_ij = max(0, min(Cr_i, Cs_j) - max(Cr_{i-1}, Cs_{j-1}))

    where Cr/Cs are cumulative sums of the raw/scaled DS0..DS4 histograms
    (Cr_{-1} = Cs_{-1} = 0). No approximation and no per-building data
    needed -- only the two marginal histograms already cached in the
    per-event CSVs.

    Parameters
    ----------
    raw_counts, scaled_counts : array (M, 5) - DS0..DS4 housing-unit counts
        per (event, county) row. Row sums must match (scaling relabels
        damage state, it does not change the number of housing units).

    Returns
    -------
    array (M, 5, 5); T[:, i, j] = units moving from raw DS i to scaled DS j.
    """
    raw_counts = np.asarray(raw_counts, dtype=float)
    scaled_counts = np.asarray(scaled_counts, dtype=float)
    m = raw_counts.shape[0]

    zeros = np.zeros((m, 1))
    cr = np.concatenate([zeros, np.cumsum(raw_counts, axis=1)], axis=1)     # (M, 6)
    cs = np.concatenate([zeros, np.cumsum(scaled_counts, axis=1)], axis=1)  # (M, 6)

    t = np.zeros((m, N_DS, N_DS), dtype=float)
    for i in range(N_DS):
        for j in range(N_DS):
            lo = np.maximum(cr[:, i], cs[:, j])
            hi = np.minimum(cr[:, i + 1], cs[:, j + 1])
            t[:, i, j] = np.clip(hi - lo, 0.0, None)
    return t
