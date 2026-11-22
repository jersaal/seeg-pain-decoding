"""Multivariate decoding of high versus low pain.

Within each Leave-2-Days-Out fold, the contacts of an anatomical region are
reduced to one feature per frequency band (first principal component, fitted
on the training surveys), and a random forest is trained on the resulting
region-level features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from .cv import has_both_classes, leave_two_days_out
from .stats import parallel_map, permutation_seeds


# ---------------------------------------------------------------------------
# Region-level features
# ---------------------------------------------------------------------------

def region_features(
    X_train: np.ndarray, X_test: np.ndarray, groups: Sequence[str], seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Reduce contact-level power to one feature per region and band.

    Columns sharing a ``groups`` label (same anatomical region and frequency
    band) are z-scored with the training mean and standard deviation and
    projected on their first principal component, fitted on the training
    surveys. A group with a single contact contributes that contact directly.
    The stacked region-level features are then z-scored with training
    statistics.

    Returns the training and test feature matrices and the group name of
    each column (sorted alphabetically).
    """
    groups = np.asarray(groups)
    names = sorted(set(groups))
    train_columns, test_columns = [], []
    for name in names:
        columns = np.flatnonzero(groups == name)
        if len(columns) == 1:
            train_columns.append(X_train[:, columns[0]])
            test_columns.append(X_test[:, columns[0]])
            continue
        scaler = StandardScaler().fit(X_train[:, columns])
        pca = PCA(n_components=1, random_state=seed)
        train_columns.append(pca.fit_transform(scaler.transform(X_train[:, columns])).ravel())
        test_columns.append(pca.transform(scaler.transform(X_test[:, columns])).ravel())

    Z_train = np.column_stack(train_columns)
    Z_test = np.column_stack(test_columns)
    scaler = StandardScaler().fit(Z_train)
    return scaler.transform(Z_train), scaler.transform(Z_test), names


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    """One cross-validation fold with its region-level features."""

    train: np.ndarray
    test: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray


def build_folds(
    X_fit: np.ndarray,
    X_eval: np.ndarray,
    days: np.ndarray,
    groups: Sequence[str],
    keep: Callable[[np.ndarray, np.ndarray], bool],
    buffer_days: int = 0,
    seed: int = 0,
) -> List[Fold]:
    """Leave-2-Days-Out folds with region-level features.

    ``X_fit`` supplies the training surveys and ``X_eval`` the held-out
    surveys; they are the same matrix except in the cross-context control
    (train during the survey, test before it). The feature reduction does not
    use the labels, so the folds can be reused across label permutations.

    ``keep(train, test)`` decides whether a split is used.
    """
    folds = []
    for train, test in leave_two_days_out(days, buffer_days):
        if not keep(train, test):
            continue
        Z_train, Z_test, _ = region_features(X_fit[train], X_eval[test], groups, seed)
        folds.append(Fold(train, test, Z_train, Z_test))
    return folds


def classification_folds(
    X_fit: np.ndarray,
    X_eval: np.ndarray,
    y: np.ndarray,
    days: np.ndarray,
    groups: Sequence[str],
    buffer_days: int = 0,
    min_train: int = 10,
    seed: int = 0,
) -> List[Fold]:
    """Folds whose training set has both classes and at least ``min_train`` surveys.

    Whether the *test* set has both classes is checked at evaluation time,
    because it changes when the labels are permuted.
    """
    def keep(train, test):
        return len(train) >= min_train and has_both_classes(y[train])

    return build_folds(X_fit, X_eval, days, groups, keep, buffer_days, seed)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass
class DecodingResult:
    """Cross-validated classifier output for one participant.

    Only folds with both classes in the held-out days are evaluated;
    all per-fold lists are aligned with ``fold_auc``.
    """

    fold_auc: np.ndarray
    fold_test: List[np.ndarray]
    fold_score: List[np.ndarray]
    fold_n_train: np.ndarray
    importance: Optional[np.ndarray] = None  # (n_folds, n_features)

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_auc))


def _fit(fold: Fold, y: np.ndarray, n_trees: int, seed: int, n_jobs: int):
    forest = RandomForestClassifier(
        n_estimators=n_trees, class_weight="balanced", random_state=seed, n_jobs=n_jobs
    )
    return forest.fit(fold.X_train, y[fold.train])


def _evaluable(fold: Fold, y: np.ndarray) -> bool:
    return has_both_classes(y[fold.test]) and has_both_classes(y[fold.train])


def decode(
    folds: List[Fold],
    y: np.ndarray,
    n_trees: int,
    seed: int,
    n_jobs: int = -1,
    n_importance_repeats: int = 0,
) -> DecodingResult:
    """Train and test the random forest in every evaluable fold.

    With ``n_importance_repeats > 0``, permutation importance is computed on
    the held-out surveys of each fold: the decrease in AUC when one
    region-level feature is shuffled, averaged over that many shuffles.
    """
    aucs, tests, scores, n_train, importances = [], [], [], [], []
    for fold in folds:
        if not _evaluable(fold, y):
            continue
        forest = _fit(fold, y, n_trees, seed, n_jobs)
        score = forest.predict_proba(fold.X_test)[:, 1]
        aucs.append(roc_auc_score(y[fold.test], score))
        tests.append(fold.test)
        scores.append(score)
        n_train.append(len(fold.train))
        if n_importance_repeats > 0:
            # Parallelize over features only; nested forest threads are slower.
            forest.set_params(n_jobs=1)
            result = permutation_importance(
                forest,
                fold.X_test,
                y[fold.test],
                n_repeats=n_importance_repeats,
                scoring="roc_auc",
                random_state=seed,
                n_jobs=n_jobs,
            )
            importances.append(result.importances_mean)
    return DecodingResult(
        fold_auc=np.array(aucs),
        fold_test=tests,
        fold_score=scores,
        fold_n_train=np.array(n_train),
        importance=np.vstack(importances) if importances else None,
    )


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

_worker = {}


def _init_worker(folds: List[Fold], n_trees: int) -> None:
    _worker["folds"] = folds
    _worker["n_trees"] = n_trees


def _null_auc(task: Tuple[int, np.ndarray]) -> float:
    seed, y_permuted = task
    aucs = []
    for fold in _worker["folds"]:
        if not _evaluable(fold, y_permuted):
            continue
        forest = _fit(fold, y_permuted, _worker["n_trees"], seed, n_jobs=1)
        aucs.append(roc_auc_score(y_permuted[fold.test], forest.predict_proba(fold.X_test)[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan


def permutation_null(
    folds: List[Fold],
    y: np.ndarray,
    n_permutations: int,
    n_trees: int,
    seed: int,
    n_workers: int,
) -> np.ndarray:
    """Null distribution of the mean fold AUC under shuffled labels.

    For each permutation the labels are shuffled across surveys and the
    classifier is retrained and tested in every fold. Scaling and PCA do not
    depend on the labels, so the precomputed fold features are exactly those
    a full re-run of the pipeline would produce.
    """
    seeds = permutation_seeds(seed, n_permutations)
    tasks = [(int(s), np.random.RandomState(s).permutation(y)) for s in seeds]
    null = np.array(parallel_map(_null_auc, tasks, n_workers, _init_worker, (folds, n_trees)))
    return null[~np.isnan(null)]


# ---------------------------------------------------------------------------
# Per-survey predictions and operating point
# ---------------------------------------------------------------------------

def pooled_predictions(result: DecodingResult, y: np.ndarray) -> pd.DataFrame:
    """One prediction per survey, pooled over the folds in which it was held out.

    Predicted probabilities are not comparable across folds, so they are
    rank-transformed within each fold and divided by the fold size before
    averaging over folds.

    Returns a table indexed by survey with ``label``, ``score`` (mean raw
    probability) and ``ranked_score``.
    """
    rows = pd.DataFrame(
        {
            "survey": np.concatenate(result.fold_test),
            "score": np.concatenate(result.fold_score),
            "ranked_score": np.concatenate(
                [rankdata(s, "average") / len(s) for s in result.fold_score]
            ),
        }
    )
    pooled = rows.groupby("survey")[["score", "ranked_score"]].mean().sort_index()
    pooled.insert(0, "label", y[pooled.index.to_numpy()])
    return pooled


def youden_operating_point(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Sensitivity and specificity at the threshold maximizing Youden's J.

    The threshold is chosen on the same pooled predictions it is applied to,
    so the resulting metrics are optimistic and descriptive only.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold = thresholds[np.argmax(tpr - fpr)]
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    fp = int(np.sum(predicted & ~positive))
    return {
        "threshold": float(threshold),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


# ---------------------------------------------------------------------------
# Feature redundancy
# ---------------------------------------------------------------------------

def feature_redundancy(
    X: np.ndarray, groups: Sequence[str], variance_explained: float = 0.95, seed: int = 0
) -> dict:
    """Correlation structure of the region-level features.

    The feature reduction is applied to all labeled surveys at once. Returns
    the absolute pairwise Pearson correlations between region-level features
    and the effective dimensionality: the number of principal components
    needed to explain ``variance_explained`` of their total variance.
    """
    Z, _, names = region_features(X, X, groups, seed)
    correlation = np.corrcoef(Z.T)
    abs_r = np.abs(correlation[np.triu_indices(len(names), k=1)])
    cumulative = np.cumsum(PCA(random_state=seed).fit(Z).explained_variance_ratio_)
    return {
        "n_features": len(names),
        "abs_r": abs_r,
        "median_abs_r": float(np.median(abs_r)),
        "effective_dim": int(np.searchsorted(cumulative, variance_explained) + 1),
    }
