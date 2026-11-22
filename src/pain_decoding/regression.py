"""Regression on continuous pain ratings.

Same region-level features and Leave-2-Days-Out scheme as the classifier,
with a random forest regressor and all surveys (no binarization).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from .multivariate import Fold, build_folds
from .stats import parallel_map, permutation_seeds


def regression_folds(
    X: np.ndarray,
    vas: np.ndarray,
    days: np.ndarray,
    groups: Sequence[str],
    min_test: int = 5,
    min_train: int = 10,
    seed: int = 0,
) -> List[Fold]:
    """Folds in which a within-fold rank correlation can be estimated."""
    def keep(train, test):
        return len(test) >= min_test and len(train) >= min_train and np.ptp(vas[test]) > 0

    return build_folds(X, X, days, groups, keep, seed=seed)


def _predict(fold: Fold, y: np.ndarray, n_trees: int, seed: int, n_jobs: int) -> np.ndarray:
    # max_features="sqrt" matches the per-split feature sampling of the
    # classifier (scikit-learn's default for classification forests).
    forest = RandomForestRegressor(
        n_estimators=n_trees, max_features="sqrt", random_state=seed, n_jobs=n_jobs
    )
    return forest.fit(fold.X_train, y[fold.train]).predict(fold.X_test)


def fold_correlations(
    folds: List[Fold], y: np.ndarray, n_trees: int, seed: int, n_jobs: int = -1
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Spearman correlation between predicted and reported ratings in each fold.

    A constant prediction is scored as zero correlation. Returns the per-fold
    correlations and the per-fold predictions.
    """
    correlations, predictions = [], []
    for fold in folds:
        predicted = _predict(fold, y, n_trees, seed, n_jobs)
        if np.ptp(predicted) == 0:
            correlations.append(0.0)
        else:
            correlations.append(spearmanr(predicted, y[fold.test]).correlation)
        predictions.append(predicted)
    return np.array(correlations), predictions


def pooled_r2(folds: List[Fold], predictions: List[np.ndarray], y: np.ndarray) -> float:
    """Coefficient of determination of the per-survey predictions.

    Each survey's prediction is averaged over the folds in which it was held
    out and compared with the participant's mean rating; negative values mean
    the predictions are less accurate than that constant.
    """
    total = np.zeros(len(y))
    count = np.zeros(len(y))
    for fold, predicted in zip(folds, predictions):
        total[fold.test] += predicted
        count[fold.test] += 1
    held_out = count > 0
    predicted = total[held_out] / count[held_out]
    observed = y[held_out]
    return float(
        1 - np.sum((observed - predicted) ** 2) / np.sum((observed - observed.mean()) ** 2)
    )


_worker = {}


def _init_worker(folds: List[Fold], n_trees: int) -> None:
    _worker["folds"] = folds
    _worker["n_trees"] = n_trees


def _null_correlation(task: Tuple[int, np.ndarray]) -> float:
    seed, y_permuted = task
    correlations, _ = fold_correlations(
        _worker["folds"], y_permuted, _worker["n_trees"], seed, n_jobs=1
    )
    return float(correlations.mean())


def permutation_null(
    folds: List[Fold],
    y: np.ndarray,
    n_permutations: int,
    n_trees: int,
    seed: int,
    n_workers: int,
) -> np.ndarray:
    """Null distribution of the mean within-fold correlation under shuffled ratings."""
    seeds = permutation_seeds(seed, n_permutations)
    tasks = [(int(s), np.random.RandomState(s).permutation(y)) for s in seeds]
    null = np.array(
        parallel_map(_null_correlation, tasks, n_workers, _init_worker, (folds, n_trees))
    )
    return null[~np.isnan(null)]
