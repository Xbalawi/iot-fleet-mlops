from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.model.drift import grade_windows, scan_windows

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def readings(sensor_id, values, run_id=0, sensor_type="temperature", fault_type=None, start=0):
    df = pd.DataFrame(
        {
            "sensor_id": sensor_id,
            "type": sensor_type,
            "value": values,
            "timestamp": pd.to_datetime(
                [T0 + timedelta(seconds=start + i) for i in range(len(values))], utc=True
            ),
            "run_id": run_id,
        }
    )
    if fault_type is not None:
        df["fault_type"] = fault_type
    return df


def reference():
    return readings("t1", np.random.default_rng(0).normal(24, 1, 500))


def test_no_alarm_when_the_data_has_not_changed():
    current = readings("t1", np.random.default_rng(1).normal(24, 1, 300))
    windows = scan_windows(reference(), current)
    assert len(windows) == 3
    assert not windows["drifted"].any()


def test_a_shifted_window_raises_an_alarm():
    rng = np.random.default_rng(2)
    current = readings("t1", np.concatenate([rng.normal(24, 1, 100), rng.normal(26, 1, 100)]))
    windows = scan_windows(reference(), current)
    assert windows["drifted"].tolist() == [False, True]


def test_windows_never_cross_a_run():
    rng = np.random.default_rng(3)
    current = pd.concat(
        [
            readings("t1", rng.normal(24, 1, 150), run_id=0),
            readings("t1", rng.normal(24, 1, 150), run_id=1, start=1000),
        ]
    )
    windows = scan_windows(reference(), current)
    assert windows["run_id"].tolist() == [0, 1]  # 150 readings = one full window + 50 dropped, per run


def test_sensor_types_without_a_model_are_skipped():
    rng = np.random.default_rng(4)
    current = pd.concat(
        [
            readings("t1", rng.normal(24, 1, 100)),
            readings("o1", rng.integers(0, 2, 100).astype(float), sensor_type="occupancy"),
        ]
    )
    windows = scan_windows(reference(), current)
    assert windows["sensor_id"].tolist() == ["t1"]


def test_a_sensor_without_reference_data_is_skipped():
    current = readings("other", np.random.default_rng(5).normal(24, 1, 100))
    windows = scan_windows(reference(), current)
    assert windows.empty
    assert "p_value" in windows.columns


def test_drift_share_is_the_share_of_readings_labelled_drift():
    values = np.random.default_rng(6).normal(24, 1, 100)
    current = readings("t1", values, fault_type=["drift"] * 60 + ["normal"] * 40)
    windows = scan_windows(reference(), current)
    assert windows["drift_share"].iloc[0] == pytest.approx(0.6)


def test_grade_windows_matches_a_hand_calculation():
    windows = pd.DataFrame(
        {
            "drift_share": [1.0, 0.6, 0.3, 0.0, 0.0],
            "drifted": [True, False, True, False, True],
        }
    )
    result = grade_windows(windows)
    # drift windows: the first two (share >= 0.5), one caught. The 0.3 window is mixed: left out.
    assert result["drift_windows"] == 2
    assert result["caught"] == 1
    assert result["recall"] == pytest.approx(0.5)
    # clean windows: the last two (share == 0), one false alarm.
    assert result["clean_windows"] == 2
    assert result["false_alarms"] == 1
    assert result["false_alarm_rate"] == pytest.approx(0.5)