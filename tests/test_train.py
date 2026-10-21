from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.model.features import add_features, drop_warm_up
from src.model.train import evaluate, predict, split_by_run, train_models

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def make_readings(spike_at=None):
    """Two runs (1000 s apart) of calm temperature and random occupancy, 1 reading per second."""
    rng = np.random.default_rng(0)
    rows = []
    for start in (0, 1000):
        for i in range(300):
            when = T0 + timedelta(seconds=start + i)
            value = 24 + rng.normal(0, 1)
            if start == 1000 and i == spike_at:
                value = 60.0
            rows.append(("t1", "temperature", value, when))
            rows.append(("o1", "occupancy", float(rng.integers(0, 2)), when))
    df = pd.DataFrame(rows, columns=["sensor_id", "type", "value", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def featured(spike_at=None):
    return drop_warm_up(add_features(make_readings(spike_at)))


def test_split_by_run_trains_on_the_first_run_only():
    train, test = split_by_run(featured())
    assert set(train["run_id"]) == {0}
    assert set(test["run_id"]) == {1}


def test_split_needs_at_least_two_runs():
    one_run = featured()
    one_run = one_run[one_run["run_id"] == 0]
    with pytest.raises(ValueError):
        split_by_run(one_run)


def test_only_temperature_and_vibration_get_a_model():
    train, _ = split_by_run(featured())
    models = train_models(train)
    assert set(models) == {"temperature"}  # no vibration data here, and never occupancy


def test_a_big_spike_in_the_test_run_is_flagged():
    train, test = split_by_run(featured(spike_at=250))
    scored = predict(train_models(train), test)
    assert scored.loc[scored["value"] == 60.0, "is_flagged"].tolist() == [True]


def test_occupancy_is_never_flagged():
    train, test = split_by_run(featured(spike_at=250))
    scored = predict(train_models(train), test)
    assert not scored.loc[scored["type"] == "occupancy", "is_flagged"].any()


def test_same_seed_gives_same_scores():
    train, test = split_by_run(featured())
    first = predict(train_models(train, seed=1), test)["score"]
    second = predict(train_models(train, seed=1), test)["score"]
    pd.testing.assert_series_equal(first, second)


def test_evaluate_matches_a_hand_calculation():
    rows = [
        # type,          fault_type, is_fault, is_flagged
        ("temperature", "spike", True, True),
        ("temperature", "spike", True, False),
        ("temperature", "drift", True, False),
        ("temperature", "drift", True, False),
        ("temperature", "normal", False, True),
        ("temperature", "normal", False, False),
        ("temperature", "normal", False, False),
        ("temperature", "normal", False, False),
        ("occupancy", "normal", False, True),  # must be ignored
    ]
    scored = pd.DataFrame(rows, columns=["type", "fault_type", "is_fault", "is_flagged"])

    table, summary = evaluate(scored)

    table = table.set_index("fault_type")
    assert table.loc["spike", "rows"] == 2
    assert table.loc["spike", "caught"] == 1
    assert table.loc["spike", "recall"] == pytest.approx(0.5)
    assert table.loc["drift", "recall"] == pytest.approx(0.0)
    assert summary["precision"] == pytest.approx(0.5)  # 1 of the 2 flagged rows is faulty
    assert summary["false_alarm_rate"] == pytest.approx(0.25)  # 1 of 4 normal rows flagged