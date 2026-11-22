import numpy as np
import pandas as pd

from pain_decoding import stats, univariate


def test_permutation_pvalue_counts_ties_and_adds_one():
    null = np.array([0.4, 0.5, 0.6, 0.7])
    assert stats.permutation_pvalue(0.6, null) == (2 + 1) / (4 + 1)
    assert stats.permutation_pvalue(0.9, null) == 1 / 5


def test_fdr_bh_matches_hand_computation():
    reject, adjusted = stats.fdr_bh([0.01, 0.02, 0.03, 0.5])
    np.testing.assert_allclose(adjusted, [0.04, 0.04, 0.04, 0.5])
    assert reject.tolist() == [True, True, True, False]


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    values = np.random.RandomState(0).rand(40)
    low, high = stats.bootstrap_mean_ci(values, 2000, np.random.RandomState(1))
    assert low < values.mean() < high
    assert (low, high) == stats.bootstrap_mean_ci(values, 2000, np.random.RandomState(1))


def test_parallel_map_matches_serial():
    assert stats.parallel_map(abs, [-1, 2, -3], n_workers=2) == [1, 2, 3]


def test_univariate_separates_informative_from_noise_features(toy):
    X, y, days = toy["X"][:, [0, 23]], toy["y"], toy["days"]  # region A and region D
    aucs = univariate.fold_aucs(univariate.standardized_splits(X, days), y, seed=0)
    assert aucs.shape[0] == 2
    assert aucs[0].mean() > 0.8 and abs(aucs[1].mean() - 0.5) < 0.2

    features = pd.DataFrame(
        {"feature": ["a_beta", "d_gamma"], "contact": ["a", "d"],
         "band": ["beta", "gamma"], "region": ["A", "D"]}
    )
    models = univariate.summarize_models(aucs, X, y, features)
    assert models["feature"].tolist() == ["a_beta", "d_gamma"]
    assert models.loc[0, "direction"] == "+"

    null = univariate.permutation_null(X, y, days, 30, seed=0, n_workers=1)
    assert null.shape == (30, 2)
    table = univariate.feature_significance(
        models, pd.DataFrame(null, columns=features["feature"]), alpha=0.05
    )
    assert table.loc[0, "p"] == 1 / 31
    assert table.loc[0, "significant_contact_fdr"] and not table.loc[1, "significant_contact_fdr"]


def test_region_consistency_requires_every_covered_participant():
    contacts = pd.DataFrame(
        {
            "participant": ["P1", "P1", "P2", "P3", "P1", "P2", "P3"],
            "region": ["X", "X", "X", "X", "Y", "Y", "Y"],
            "sig": [True, False, True, True, True, False, True],
        }
    )
    table = univariate.region_consistency(contacts, "sig", min_participants=3)
    table = table.set_index("region")
    assert table.loc["X", "all_covered_significant"]
    assert not table.loc["Y", "all_covered_significant"]
