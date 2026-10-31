from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.model.pipeline import run_retraining, split_recent_validation

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
BAD = {"recall_spike": 0.9, "false_alarm_rate": 0.05}
GOOD = {"recall_spike": 0.9, "false_alarm_rate": 0.01}


def frame(values, sensor_id="t1", sensor_type="temperature", start=0, run_id=1):
    return pd.DataFrame(
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


def reference():
    return frame(np.random.default_rng(0).normal(24, 1, 500), run_id=0)


def calm_recent():
    return frame(np.random.default_rng(1).normal(24, 1, 300), start=1000)  # same as the reference


def drifted_recent():
    return frame(np.random.default_rng(2).normal(27, 1, 300), start=1000)  # 3 standard deviations up


def trained_models(df):
    return {"temperature": "NEW", "vibration": "NEW"}


def judge(models, validation):
    return GOOD if models["temperature"] == "NEW" else BAD


def worse_if_new(models, validation):
    return BAD if models["temperature"] == "NEW" else GOOD


CURRENT = {"temperature": "OLD", "vibration": "OLD"}


def test_split_cuts_in_time_and_keeps_the_newest_for_validation():
    recent, validation = split_recent_validation(frame(np.arange(10.0)), recent_fraction=0.6)
    assert len(recent) == 6
    assert len(validation) == 4
    assert recent["timestamp"].max() < validation["timestamp"].min()


def test_no_sustained_drift_means_no_retraining():
    calls = []

    def train_fn(df):
        calls.append(df)
        return trained_models(df)

    result = run_retraining(
        calm_recent(), calm_recent(), reference(), CURRENT, train_fn=train_fn, measure_fn=judge
    )

    assert result["retrained"] is False
    assert result["decision"] is None
    assert calls == []


def test_a_better_candidate_is_promoted_and_only_the_triggered_type_changes():
    result = run_retraining(
        drifted_recent(), drifted_recent(), reference(), CURRENT, train_fn=trained_models, measure_fn=judge
    )

    assert result["triggered"] == ["t1"]
    assert result["decision"]["promote"] is True
    assert result["candidate_models"] == {"temperature": "NEW", "vibration": "OLD"}


def test_a_worse_candidate_is_rejected():
    result = run_retraining(
        drifted_recent(), drifted_recent(), reference(), CURRENT, train_fn=trained_models, measure_fn=worse_if_new
    )

    assert result["retrained"] is True
    assert result["decision"]["promote"] is False


def test_force_retrains_even_without_drift():
    result = run_retraining(
        calm_recent(), calm_recent(), reference(), CURRENT, force=True, train_fn=trained_models, measure_fn=judge
    )

    assert result["retrained"] is True
    assert result["triggered"] == ["t1"]