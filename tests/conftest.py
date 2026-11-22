import numpy as np
import pytest


@pytest.fixture
def toy():
    """Synthetic participant: 8 days x 10 surveys, 4 regions x 3 contacts x 2 bands.

    Region A carries the label; the other regions are noise.
    """
    rng = np.random.RandomState(0)
    n_days, per_day = 8, 10
    days = np.repeat(np.arange(100, 100 + n_days), per_day)
    y = rng.permutation(np.tile([0, 1], n_days * per_day // 2))
    groups, columns = [], []
    for region in "ABCD":
        for band in ("beta", "gamma"):
            shared = rng.randn(len(y))
            for _ in range(3):
                signal = 2.0 * y if region == "A" else 0.0
                columns.append(signal + shared + 0.5 * rng.randn(len(y)))
                groups.append(f"{region}_{band}")
    return {"X": np.column_stack(columns), "y": y, "days": days, "groups": np.array(groups)}
