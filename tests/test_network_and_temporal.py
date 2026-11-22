import numpy as np
import pandas as pd
import pytest

from pain_decoding import network, temporal


def toy_atlas():
    return network.Atlas(
        n_parcels=2,
        centroids=np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]),
        networks=np.array(["Salience", "Default Mode"]),
        volumes_mm3=np.array([1000.0, 3000.0]),
    )


def test_assignment_uses_nearest_centroid_within_threshold():
    xyz = np.array([[3.0, 0, 0], [36.0, 0, 0], [20.0, 0, 0]])
    assigned = network.assign_networks(xyz, toy_atlas(), max_distance_mm=7)
    assert assigned.tolist() == ["Salience", "Default Mode", network.UNASSIGNED]


def test_ranks_are_scaled_within_participant():
    models = pd.DataFrame(
        {"participant": ["P1"] * 4 + ["P2"] * 2, "auc": [0.1, 0.9, 0.5, 0.7, 0.6, 0.4]}
    )
    ranked = network.rank_within_participant(models)
    assert ranked.tolist() == [0.25, 1.0, 0.5, 0.75, 1.0, 0.5]


@pytest.mark.parametrize("make_rng", [np.random.RandomState, np.random.default_rng])
def test_enrichment_detects_planted_network(make_rng):
    rng = np.random.RandomState(0)
    rows = []
    for participant in ("P1", "P2", "P3"):
        for contact in range(30):
            net = "Salience" if contact < 10 else "Default Mode"
            for band in ("beta", "gamma"):
                boost = 0.3 if (net, band) == ("Salience", "gamma") else 0.0
                rows.append((participant, net, band, rng.rand() + boost))
    models = pd.DataFrame(rows, columns=["participant", "network", "band", "auc"])
    models["ranked_auc"] = network.rank_within_participant(models)

    result = network.enrichment_test(models, ("beta", "gamma"), 2000, make_rng(0))
    assert len(result) == 4
    hit = result[result["significant"]]
    assert hit[["network", "band"]].values.tolist() == [["Salience", "gamma"]]

    rest = network.network_versus_rest(models, ("beta", "gamma")).set_index(["network", "band"])
    assert rest.loc[("Salience", "gamma"), "effect"] > 0.7


@pytest.mark.parametrize("make_rng", [np.random.RandomState, np.random.default_rng])
def test_permutation_keeps_values_within_participants(make_rng):
    values = np.arange(10.0)
    participant = np.array(["a"] * 4 + ["b"] * 6)
    permuted = network._permute_within_participants(values, participant, 50, make_rng(0))
    assert permuted.shape == (50, 10)
    assert np.all(np.sort(permuted[:, :4], axis=1) == values[:4])
    assert np.all(np.sort(permuted[:, 4:], axis=1) == values[4:])
    assert not np.all(permuted == values)


def test_jitter_moves_every_contact_by_the_given_distance():
    xyz = np.random.RandomState(0).randn(100, 3) * 30
    moved = network.jitter_coordinates(xyz, 3.0, np.random.default_rng(0))
    np.testing.assert_allclose(np.linalg.norm(moved - xyz, axis=1), 3.0)


def test_atlas_parcel_size():
    atlas = toy_atlas()
    assert atlas.median_volume_cm3 == 2.0
    assert atlas.equivalent_radius_mm == pytest.approx((3 * 2000 / (4 * np.pi)) ** (1 / 3))


def test_autocorrelation_of_a_day_level_signal():
    rng = np.random.RandomState(0)
    days = np.repeat(np.arange(12), 8)
    hours = days * 24.0 + np.tile(np.arange(8), 12)
    slow = np.repeat(rng.randn(12), 8)                    # constant within a day
    Z = temporal.zscore(np.column_stack([slow + 0.1 * rng.randn(96), rng.randn(96)]))
    n_pairs, acf = temporal.lagged_autocorrelation(Z, days, hours)
    assert n_pairs[0] == 12 * 1 * 7 and acf.shape == (len(temporal.LAG_BINS), 2)
    assert acf[0, 0] > 0.8          # same day, slow signal
    assert abs(acf[0, 1]) < 0.3     # same day, white noise
    assert abs(acf[3, 0]) < 0.5     # next day

    low, high = temporal.bootstrap_autocorrelation_ci(Z, days, hours, 200, np.random.RandomState(1))
    assert low[0, 0] > 0 and low[0, 1] < 0 < high[0, 1]
    assert temporal.between_day_variance_share(Z, days)[0] > 0.9


def test_decorrelation_lag_is_first_across_day_bin_including_zero():
    ci_low = np.array([0.5, 0.4, 0.3, 0.2, 0.1, -0.1, 0.2, 0.2])
    assert temporal.DECORRELATION_LABELS[temporal.decorrelation_lag(ci_low)] == "by 3 days"
    assert temporal.decorrelation_lag(np.full(8, 0.1)) == 5
    assert temporal.decorrelation_lag(np.array([1, 1, 1, np.nan, 1, 1, 1, 1.0])) == 0
