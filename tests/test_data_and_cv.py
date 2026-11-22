import numpy as np

from pain_decoding.cv import leave_two_days_out
from pain_decoding.data import binarize_pain


def test_binarize_excludes_middle_third():
    vas = np.arange(1.0, 101.0)
    keep, y, low, high = binarize_pain(vas, 0.33, 0.66)
    assert (low, high) == (34.0, 67.0)
    assert np.all(vas[keep][y == 0] <= low) and np.all(vas[keep][y == 1] >= high)
    assert len(keep) == 34 + 34
    assert not np.any((vas[keep] > low) & (vas[keep] < high))


def test_binarize_ties_go_to_the_class_of_the_threshold():
    vas = np.array([10, 10, 10, 50, 50, 90, 90, 90, 90], dtype=float)
    keep, y, low, high = binarize_pain(vas)
    assert low == 10 and high == 90
    assert y.tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_leave_two_days_out_never_shares_a_day():
    days = np.repeat([1, 2, 3, 5, 6], 4)
    splits = leave_two_days_out(days)
    assert len(splits) == 10  # 5 choose 2
    for train, test in splits:
        assert len(np.unique(days[test])) == 2
        assert not set(days[train]) & set(days[test])
        assert len(train) + len(test) == len(days)


def test_buffer_removes_adjacent_training_days():
    days = np.repeat([1, 2, 3, 5, 6, 9, 10], 4)
    plain = leave_two_days_out(days)
    buffered = leave_two_days_out(days, buffer_days=1)
    assert len(buffered) == len(plain)
    for (train, test), (plain_train, _) in zip(buffered, plain):
        gap = np.abs(days[train][:, None] - np.unique(days[test])[None, :]).min()
        assert gap > 1
        assert set(train) <= set(plain_train)
    # Holding out days 2 and 5 removes days 1, 3 and 6 from training.
    train, test = next(s for s in buffered if set(days[s[1]]) == {2, 5})
    assert set(days[train]) == {9, 10}
