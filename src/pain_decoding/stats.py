"""Statistical helpers shared by all analyses."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from statsmodels.stats.multitest import multipletests


def permutation_pvalue(observed: float, null: np.ndarray) -> float:
    """One-sided permutation p-value, (b + 1) / (n + 1).

    ``b`` is the number of null values greater than or equal to the observed
    statistic and ``n`` the number of permutations.
    """
    null = np.asarray(null)
    return (np.sum(null >= observed) + 1) / (len(null) + 1)


def fdr_bh(pvalues: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg correction. Returns (reject, adjusted p-values)."""
    reject, adjusted, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    return reject, adjusted


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int, rng: np.random.RandomState, level: float = 95.0
) -> Tuple[float, float]:
    """Percentile-bootstrap confidence interval of the mean.

    ``rng`` is advanced by the call. Where one generator is shared across
    participants, the intervals depend on the order of the calls; the scripts
    fix that order.
    """
    values = np.asarray(values)
    means = rng.choice(values, (n_boot, len(values)), replace=True).mean(axis=1)
    tail = (100.0 - level) / 2.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)


def permutation_seeds(seed: int, n_permutations: int) -> np.ndarray:
    """One independent seed per permutation, derived from the master seed."""
    return np.random.RandomState(seed).randint(0, 2**31 - 1, size=n_permutations)


def parallel_map(
    func: Callable,
    items: Iterable,
    n_workers: int,
    initializer: Callable = None,
    initargs: tuple = (),
) -> List:
    """Ordered map over worker processes (serial when ``n_workers`` is 1)."""
    items = list(items)
    if n_workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [func(item) for item in items]
    chunksize = max(1, len(items) // (n_workers * 8))
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=initializer, initargs=initargs
    ) as pool:
        return list(pool.map(func, items, chunksize=chunksize))
