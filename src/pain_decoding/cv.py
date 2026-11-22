"""Leave-2-Days-Out cross-validation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import LeavePGroupsOut

Split = Tuple[np.ndarray, np.ndarray]


def leave_two_days_out(days: np.ndarray, buffer_days: int = 0) -> List[Split]:
    """All (train, test) index splits that hold out two recording days.

    Every pair of days is held out once, so training and test surveys never
    come from the same day. With ``buffer_days > 0``, training surveys
    recorded within that many calendar days of a held-out day are removed as
    well, so that at least ``buffer_days`` full days separate training from
    test surveys.

    Splits are returned for every pair of days; callers decide which ones are
    usable (for example, both classes present in the test set).
    """
    days = np.asarray(days)
    placeholder = np.zeros(len(days))
    splits = []
    for train, test in LeavePGroupsOut(n_groups=2).split(placeholder, groups=days):
        if buffer_days > 0:
            held_out = np.unique(days[test])
            distance = np.abs(days[train][:, None] - held_out[None, :]).min(axis=1)
            train = train[distance > buffer_days]
        splits.append((train, test))
    return splits


def has_both_classes(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2
