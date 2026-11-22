"""Temporal dependence of pain ratings and spectral features.

The autocorrelation of a z-scored series is estimated as the mean product
``z_i * z_j`` over all pairs of surveys whose separation falls in a lag bin.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# (label, min day difference, max day difference, min hours apart, max hours apart)
LAG_BINS = [
    ("same day, <2 h", 0, 0, 0, 2),
    ("same day, 2-6 h", 0, 0, 2, 6),
    ("same day, >6 h", 0, 0, 6, np.inf),
    ("1 day", 1, 1, 0, np.inf),
    ("2 days", 2, 2, 0, np.inf),
    ("3 days", 3, 3, 0, np.inf),
    ("4 days", 4, 4, 0, np.inf),
    ("5+ days", 5, np.inf, 0, np.inf),
]
LAG_LABELS = [lag[0] for lag in LAG_BINS]
ACROSS_DAY_BINS = [3, 4, 5, 6, 7]  # the 1, 2, 3, 4 and 5+ day bins
DECORRELATION_LABELS = [
    "within 1 day", "by 2 days", "by 3 days", "by 4 days", "by 5 days", "not within 5 days",
]


def zscore(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=0)) / a.std(axis=0)


def _pair_masks(days: np.ndarray, hours: np.ndarray):
    """Survey pairs (i < j) and, per lag bin, which pairs fall into it."""
    i, j = np.triu_indices(len(days), k=1)
    day_gap = np.abs(days[i] - days[j])
    hour_gap = np.abs(hours[i] - hours[j])
    masks = [
        (day_gap >= d0) & (day_gap <= d1) & (hour_gap >= h0) & (hour_gap < h1)
        for _, d0, d1, h0, h1 in LAG_BINS
    ]
    return i, j, masks


def lagged_autocorrelation(
    Z: np.ndarray, days: np.ndarray, hours: np.ndarray, min_pairs: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Lag-binned autocorrelation of each column of the z-scored matrix ``Z``.

    Returns the number of survey pairs per bin and an array of shape
    (n_bins, n_columns); bins with fewer than ``min_pairs`` pairs are NaN.
    """
    i, j, masks = _pair_masks(days, hours)
    n_pairs = np.array([mask.sum() for mask in masks])
    acf = np.full((len(masks), Z.shape[1]), np.nan)
    for b, mask in enumerate(masks):
        if n_pairs[b] >= min_pairs:
            acf[b] = (Z[i[mask]] * Z[j[mask]]).mean(axis=0)
    return n_pairs, acf


def bootstrap_autocorrelation_ci(
    Z: np.ndarray,
    days: np.ndarray,
    hours: np.ndarray,
    n_boot: int,
    rng: np.random.RandomState,
    min_pairs: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """95% bootstrap interval of the lag-binned autocorrelation.

    Surveys are resampled with replacement and re-standardized; pairs formed
    by two copies of the same survey are excluded. Returns (low, high), each
    of shape (n_bins, n_columns).
    """
    n = len(days)
    i, j, masks = _pair_masks(days, hours)
    adjacency = []
    for mask in masks:
        A = np.zeros((n, n))
        A[i[mask], j[mask]] = 1
        adjacency.append(A + A.T)

    boot = np.full((n_boot, len(masks), Z.shape[1]), np.nan)
    for k in range(n_boot):
        sample = rng.randint(0, n, n)
        Zb = zscore(Z[sample])
        distinct = sample[:, None] != sample[None, :]
        for b, A in enumerate(adjacency):
            Ab = A[np.ix_(sample, sample)] * distinct
            n_pairs = Ab.sum() / 2
            if n_pairs >= min_pairs:
                boot[k, b] = (Zb * (Ab @ Zb)).sum(axis=0) / (2 * n_pairs)
    return np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


def decorrelation_lag(ci_low: np.ndarray) -> int:
    """First across-day lag at which the 95% interval includes zero.

    ``ci_low`` holds the lower bounds for all lag bins of one series. Returns
    an index into ``DECORRELATION_LABELS`` (0 = decorrelated within one day).
    """
    for k, b in enumerate(ACROSS_DAY_BINS):
        if not ci_low[b] > 0:
            return k
    return len(ACROSS_DAY_BINS)


def between_day_variance_share(Z: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Share of each column's variance explained by calendar day (eta squared)."""
    grand = Z.mean(axis=0)
    total = ((Z - grand) ** 2).sum(axis=0)
    between = sum(
        (days == d).sum() * (Z[days == d].mean(axis=0) - grand) ** 2 for d in np.unique(days)
    )
    return between / total
