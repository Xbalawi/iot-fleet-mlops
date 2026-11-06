import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.model.features import (
    LONG_WINDOW,
    MODEL_FEATURES,
    SHORT_WINDOW,
    add_features,
    last_row_features,
)

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def offline_last_row(values, short_window, long_window):
    """What the training code says the features of the last reading are."""
    frame = pd.DataFrame(
        {
            "sensor_id": "s",
            "type": "temperature",
            "value": values,
            "timestamp": pd.to_datetime(
                [T0 + timedelta(seconds=i) for i in range(len(values))], utc=True
            ),
            "run_id": 0,
        }
    )
    row = add_features(frame, short_window, long_window).iloc[-1]
    return row[MODEL_FEATURES].to_numpy(dtype=float)


def check(values, short_window=SHORT_WINDOW, long_window=LONG_WINDOW):
    online = last_row_features(values[:-1], values[-1], short_window, long_window)
    offline = offline_last_row(values, short_window, long_window)
    np.testing.assert_allclose(online, offline, rtol=1e-9, atol=1e-9)  # NaN equals NaN


@pytest.mark.parametrize("seed", range(10))
def test_matches_add_features_on_random_data(seed):
    rng = np.random.default_rng(seed)
    check(24 + rng.normal(0, 1 + seed % 3, size=51 + seed * 3))


def test_matches_with_other_window_sizes():
    rng = np.random.default_rng(5)
    check(24 + rng.normal(0, 1, size=12), short_window=3, long_window=7)


@pytest.mark.parametrize("n_past", [0, 1, 5, 9, 10, 11, 49, 50, 80])
def test_short_history_gives_nan_exactly_where_add_features_does(n_past):
    rng = np.random.default_rng(n_past)
    check(24 + rng.normal(0, 1, size=n_past + 1))


def test_flat_windows_have_no_z_score():
    # the whole history is flat: nothing to measure the new value against
    check(np.array([5.0] * 60 + [6.0]))

    # only the last 10 readings are flat: z_short is NaN, the long-window features are not
    rng = np.random.default_rng(1)
    values = np.concatenate([24 + rng.normal(0, 1, 45), np.full(10, 24.0), [30.0]])
    check(values)
    online = last_row_features(values[:-1], values[-1])
    assert math.isnan(online[2]) and not math.isnan(online[3])