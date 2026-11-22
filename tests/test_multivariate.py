import numpy as np

from pain_decoding import multivariate as mv
from pain_decoding import regression


def test_region_features_use_training_statistics_only(toy):
    train, test = np.arange(60), np.arange(60, 80)
    Z_train, Z_test, names = mv.region_features(toy["X"][train], toy["X"][test], toy["groups"])
    assert Z_train.shape == (60, 8) and Z_test.shape == (20, 8)
    assert names == sorted(set(toy["groups"]))
    np.testing.assert_allclose(Z_train.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z_train.std(axis=0), 1, atol=1e-12)
    # Changing the test surveys must not change the training features.
    Z_train_2, _, _ = mv.region_features(toy["X"][train], 10 * toy["X"][test], toy["groups"])
    np.testing.assert_array_equal(Z_train, Z_train_2)


def test_single_contact_region_is_passed_through():
    rng = np.random.RandomState(1)
    X = rng.randn(30, 3)
    groups = np.array(["A_beta", "A_beta", "B_beta"])
    Z_train, _, names = mv.region_features(X[:20], X[20:], groups)
    column = names.index("B_beta")
    expected = (X[:20, 2] - X[:20, 2].mean()) / X[:20, 2].std()
    np.testing.assert_allclose(Z_train[:, column], expected)


def test_decoder_finds_planted_signal_and_null_is_at_chance(toy):
    folds = mv.classification_folds(
        toy["X"], toy["X"], toy["y"], toy["days"], toy["groups"], seed=0
    )
    assert len(folds) == 28  # 8 choose 2
    result = mv.decode(folds, toy["y"], n_trees=25, seed=0, n_jobs=1, n_importance_repeats=2)
    assert result.mean_auc > 0.9
    assert result.importance.shape == (len(result.fold_auc), 8)

    null = mv.permutation_null(folds, toy["y"], 20, n_trees=25, seed=0, n_workers=1)
    assert abs(null.mean() - 0.5) < 0.1 and null.max() < result.mean_auc
    again = mv.permutation_null(folds, toy["y"], 20, n_trees=25, seed=0, n_workers=1)
    np.testing.assert_array_equal(null, again)


def test_buffered_folds_have_smaller_training_sets(toy):
    plain = mv.classification_folds(toy["X"], toy["X"], toy["y"], toy["days"], toy["groups"])
    buffered = mv.classification_folds(
        toy["X"], toy["X"], toy["y"], toy["days"], toy["groups"], buffer_days=1
    )
    assert max(len(f.train) for f in buffered) < max(len(f.train) for f in plain)


def test_pooled_predictions_and_operating_point(toy):
    folds = mv.classification_folds(toy["X"], toy["X"], toy["y"], toy["days"], toy["groups"])
    result = mv.decode(folds, toy["y"], n_trees=25, seed=0, n_jobs=1)
    pooled = mv.pooled_predictions(result, toy["y"])
    assert len(pooled) == len(toy["y"])  # every survey is held out at least once
    assert pooled["ranked_score"].between(0, 1).all()
    point = mv.youden_operating_point(
        pooled["label"].to_numpy(), pooled["ranked_score"].to_numpy()
    )
    assert point["TP"] + point["FN"] == int(toy["y"].sum())
    assert point["sensitivity"] > 0.8 and point["specificity"] > 0.8


def test_feature_redundancy_counts_dimensions():
    rng = np.random.RandomState(0)
    base = rng.randn(200, 2)
    X = np.column_stack([base[:, 0], base[:, 0], base[:, 1], base[:, 1]]) + 0.01 * rng.randn(200, 4)
    stats = mv.feature_redundancy(X, np.array(["A_b", "B_b", "C_b", "D_b"]))
    assert stats["n_features"] == 4 and stats["effective_dim"] == 2
    assert len(stats["abs_r"]) == 6


def test_regression_recovers_graded_signal(toy):
    rng = np.random.RandomState(2)
    vas = 50 + 20 * toy["y"] + 5 * rng.randn(len(toy["y"]))
    folds = regression.regression_folds(toy["X"], vas, toy["days"], toy["groups"])
    correlations, predictions = regression.fold_correlations(folds, vas, 25, seed=0, n_jobs=1)
    assert correlations.mean() > 0.5
    assert regression.pooled_r2(folds, predictions, vas) > 0.3
    null = regression.permutation_null(folds, vas, 10, 25, seed=0, n_workers=1)
    assert null.max() < correlations.mean()
