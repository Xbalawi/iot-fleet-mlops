from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.model.features import FEATURE_COLUMNS, add_features, assign_runs, drop_warm_up

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
VALUES = [1, 3, 2, 4, 3, 5]  # small enough to check by hand with windows 2 and 4


def frame(rows):
    """rows: (sensor_id, seconds, value) -> a readings DataFrame."""
    return pd.DataFrame(
        {
            "sensor_id": [r[0] for r in rows],
            "type": "temperature",
            "value": [float(r[2]) for r in rows],
            "timestamp": pd.to_datetime(
                [T0 + timedelta(seconds=r[1]) for r in rows], utc=True
            ),
        }
    )


def test_assign_runs_splits_on_a_long_pause():
    rows = [("A", 0, 1), ("B", 0, 1), ("A", 1, 1), ("B", 1, 1),
            ("A", 100, 1), ("B", 100, 1)]
    out = assign_runs(frame(rows))
    assert list(out["run_id"]) == [0, 0, 0, 0, 1, 1]


def test_a_pause_of_exactly_the_gap_stays_in_the_same_run():
    out = assign_runs(frame([("A", 0, 1), ("A", 30, 1), ("A", 61, 1)]))
    assert list(out["run_id"]) == [0, 0, 1]


def test_features_match_a_hand_calculation():
    rows = [("A", i, v) for i, v in enumerate(VALUES)]
    out = add_features(frame(rows), short_window=2, long_window=4)

    assert out["z_short"].isna().tolist() == [True, True, False, False, False, False]
    assert out["z_long"].isna().tolist() == [True, True, True, True, False, False]
    assert out.loc[4, "z_short"] == pytest.approx(0.0)
    assert out.loc[4, "z_long"] == pytest.approx(0.3873, abs=1e-3)
    assert out.loc[5, "z_short"] == pytest.approx(2.1213, abs=1e-3)
    assert out.loc[5, "z_long"] == pytest.approx(2.4495, abs=1e-3)
    assert out.loc[5, "mean_shift"] == pytest.approx(0.6124, abs=1e-3)
    assert out.loc[5, "diff1"] == 2.0


def test_a_spike_does_not_hide_inside_its_own_baseline():
    rows = [("A", i, v) for i, v in enumerate([1, 3] * 5 + [100])]
    out = add_features(frame(rows), short_window=2, long_window=4)
    assert out.loc[10, "z_short"] > 50


def test_features_are_computed_per_sensor_and_do_not_depend_on_scale():
    rows = []
    for i, v in enumerate(VALUES):
        rows.append(("A", i, v))
        rows.append(("B", i, v * 1000))
    out = add_features(frame(rows), short_window=2, long_window=4)

    a = out[out["sensor_id"] == "A"].reset_index(drop=True)
    b = out[out["sensor_id"] == "B"].reset_index(drop=True)
    assert a.loc[5, "z_long"] == pytest.approx(2.4495, abs=1e-3)
    assert b.loc[5, "z_long"] == pytest.approx(a.loc[5, "z_long"])


def test_features_are_computed_per_run():
    rows = [("A", i, v) for i, v in enumerate(VALUES)]
    rows += [("A", 100 + i, v) for i, v in enumerate(VALUES)]
    out = add_features(frame(rows), short_window=2, long_window=4)

    assert list(out["run_id"]) == [0] * 6 + [1] * 6
    assert out["z_long"].isna().sum() == 8
    assert out.loc[6:9, "z_long"].isna().all()
    assert out.loc[10, "z_long"] == pytest.approx(out.loc[4, "z_long"])


def test_gap_ratio_jumps_after_a_pause():
    rows = [("A", s, 1) for s in (0, 1, 2, 3, 13, 14)]
    out = add_features(frame(rows))
    assert out["gap_ratio"].tolist() == pytest.approx(
        [float("nan"), 1, 1, 1, 10, 1], nan_ok=True
    )


def test_drop_warm_up_keeps_only_complete_rows():
    rows = [("A", i, v) for i, v in enumerate(VALUES)]
    out = add_features(frame(rows), short_window=2, long_window=4)

    kept = drop_warm_up(out)

    assert len(kept) == 2
    assert list(kept.index) == [0, 1]
    assert not kept[FEATURE_COLUMNS].isna().any().any()


def test_labels_stay_attached_to_their_readings_even_if_the_input_is_shuffled():
    df = frame([("A", i, v) for i, v in enumerate([1, 3, 1, 3, 1, 3, 100])])
    df["fault_type"] = np.where(df["value"] == 100, "spike", "normal")

    out = add_features(df.iloc[::-1], short_window=2, long_window=4)

    assert list(out[out["value"] == 100]["fault_type"]) == ["spike"]
    assert out["timestamp"].is_monotonic_increasing


def test_the_input_dataframe_is_not_modified():
    df = frame([("A", i, v) for i, v in enumerate([1, 3, 2])])
    before = list(df.columns)
    add_features(df)
    assert list(df.columns) == before