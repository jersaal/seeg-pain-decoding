"""Univariate decoding: one logistic regression per contact and frequency band."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .cv import has_both_classes, leave_two_days_out
from .stats import fdr_bh, parallel_map

LOW_BANDS = ("delta", "theta", "alpha")
HIGH_BANDS = ("beta", "low_gamma", "high_gamma")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

@dataclass
class StandardizedSplit:
    """A Leave-2-Days-Out split with every feature z-scored on the training surveys."""

    train: np.ndarray
    test: np.ndarray
    Z_train: np.ndarray
    Z_test: np.ndarray


def standardized_splits(X: np.ndarray, days: np.ndarray) -> List[StandardizedSplit]:
    """All Leave-2-Days-Out splits of ``X`` with training-set z-scoring.

    The scaling does not depend on the labels, so the splits are computed
    once and reused for every label permutation.
    """
    splits = []
    for train, test in leave_two_days_out(days):
        Z_train = np.empty((len(train), X.shape[1]))
        Z_test = np.empty((len(test), X.shape[1]))
        # One scaler per column: fitting all columns at once differs in the
        # last floating-point bit, which can flip ties with the permutation null.
        for j in range(X.shape[1]):
            scaler = StandardScaler().fit(X[train, j : j + 1])
            Z_train[:, j] = scaler.transform(X[train, j : j + 1]).ravel()
            Z_test[:, j] = scaler.transform(X[test, j : j + 1]).ravel()
        splits.append(StandardizedSplit(train, test, Z_train, Z_test))
    return splits


def fold_aucs(splits: List[StandardizedSplit], y: np.ndarray, seed: int) -> np.ndarray:
    """AUC of every single-feature classifier in every evaluable fold.

    A fold is evaluated if both classes are present in its held-out days.
    Each classifier is an L2-regularized logistic regression (C = 1).

    Returns an array of shape (n_features, n_folds).
    """
    rows = []
    for split in splits:
        y_train, y_test = y[split.train], y[split.test]
        if not has_both_classes(y_test):
            continue
        row = np.empty(split.Z_train.shape[1])
        for j in range(len(row)):
            model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed)
            model.fit(split.Z_train[:, [j]], y_train)
            row[j] = roc_auc_score(y_test, model.predict_proba(split.Z_test[:, [j]])[:, 1])
        rows.append(row)
    # Contiguous per feature, so each feature's mean over folds is summed in
    # the same order as a 1-D array (bit-reproducible means).
    return np.ascontiguousarray(np.array(rows).T)


def summarize_models(
    aucs: np.ndarray, X: np.ndarray, y: np.ndarray, features: pd.DataFrame
) -> pd.DataFrame:
    """One row per contact-band model, sorted by mean AUC.

    ``power_difference`` is the mean power in high-pain minus low-pain
    surveys; its sign gives the direction of the pain-related modulation.
    """
    difference = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    table = features[["feature", "contact", "band", "region"]].copy()
    table["auc"] = aucs.mean(axis=1)
    table["auc_sd"] = aucs.std(axis=1)
    table["n_folds"] = aucs.shape[1]
    table["power_difference"] = difference
    table["direction"] = np.where(difference > 0, "+", "-")
    return table.sort_values("auc", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Permutation null
# ---------------------------------------------------------------------------

_worker = {}


def _init_worker(X: np.ndarray, y: np.ndarray, days: np.ndarray, seed: int) -> None:
    _worker["splits"] = standardized_splits(X, days)
    _worker["y"] = y
    _worker["seed"] = seed


def _null_row(permutation: int) -> np.ndarray:
    seed = _worker["seed"]
    y_permuted = np.random.RandomState(seed + permutation).permutation(_worker["y"])
    aucs = fold_aucs(_worker["splits"], y_permuted, seed)
    if aucs.size == 0:
        return np.full(_worker["splits"][0].Z_train.shape[1], 0.5)
    return aucs.mean(axis=1)


def permutation_null(
    X: np.ndarray, y: np.ndarray, days: np.ndarray, n_permutations: int, seed: int, n_workers: int
) -> np.ndarray:
    """Null mean AUC of every feature under shuffled labels.

    Each permutation shuffles the labels across surveys and re-runs the full
    cross-validated pipeline for every contact-band feature.

    Returns an array of shape (n_permutations, n_features).
    """
    rows = parallel_map(
        _null_row, range(n_permutations), n_workers, _init_worker, (X, y, days, seed)
    )
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def feature_significance(models: pd.DataFrame, null: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Permutation p-value of every model under two correction schemes.

    ``p_fdr_contact`` corrects across the frequency bands of each contact
    (primary analysis: does this contact carry pain-related information in
    any band?). ``p_fdr_participant`` corrects across all contact-band tests
    of the participant (sensitivity analysis).

    ``null`` has one column per feature and one row per permutation.
    """
    table = models.copy()
    null_values = null[table["feature"]].to_numpy()
    observed = table["auc"].to_numpy()
    table["p"] = (np.sum(null_values >= observed[None, :], axis=0) + 1) / (len(null) + 1)
    table["null_95th_percentile"] = np.percentile(null_values, 100 * (1 - alpha), axis=0)

    table["p_fdr_contact"] = np.nan
    table["significant_contact_fdr"] = False
    for _, rows in table.groupby("contact").groups.items():
        reject, adjusted = fdr_bh(table.loc[rows, "p"], alpha)
        table.loc[rows, "p_fdr_contact"] = adjusted
        table.loc[rows, "significant_contact_fdr"] = reject
    reject, adjusted = fdr_bh(table["p"], alpha)
    table["p_fdr_participant"] = adjusted
    table["significant_participant_fdr"] = reject
    return table


def contact_significance(features: pd.DataFrame) -> pd.DataFrame:
    """Per contact: is at least one band significant, under each scheme?"""
    grouped = features.groupby(["participant", "contact", "region"], sort=False)
    return grouped.agg(
        significant_contact_fdr=("significant_contact_fdr", "any"),
        significant_participant_fdr=("significant_participant_fdr", "any"),
        n_significant_bands=("significant_contact_fdr", "sum"),
        best_auc=("auc", "max"),
    ).reset_index()


def band_tests(models: pd.DataFrame, bands: Sequence[str]) -> dict:
    """Do contact-level AUCs differ between frequency bands?

    Friedman test across the bands over contacts, followed by a one-sided
    Wilcoxon signed-rank test of each contact's mean AUC in the upper three
    bands against its mean AUC in the lower three.
    """
    wide = models.pivot(index="contact", columns="band", values="auc")[list(bands)].dropna()
    chi2, p_friedman = friedmanchisquare(*[wide[band] for band in bands])
    difference = wide[list(HIGH_BANDS)].mean(axis=1) - wide[list(LOW_BANDS)].mean(axis=1)
    _, p_wilcoxon = wilcoxon(difference, alternative="greater")
    return {
        "n_contacts": len(wide),
        "friedman_chi2": chi2,
        "friedman_p": p_friedman,
        "high_minus_low_median": difference.median(),
        "wilcoxon_p": p_wilcoxon,
    }


def region_consistency(
    contacts: pd.DataFrame, column: str, min_participants: int
) -> pd.DataFrame:
    """Regions with a significant contact in every participant that covers them.

    Only regions sampled in at least ``min_participants`` participants are
    considered. ``column`` selects the correction scheme.
    """
    per_participant = contacts.groupby(["region", "participant"])[column].any().reset_index()
    table = per_participant.groupby("region")[column].agg(
        n_participants_covered="size", n_participants_significant="sum"
    )
    table = table[table["n_participants_covered"] >= min_participants].reset_index()
    table["all_covered_significant"] = (
        table["n_participants_significant"] == table["n_participants_covered"]
    )
    return table
