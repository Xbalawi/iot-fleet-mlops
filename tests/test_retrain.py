import pandas as pd

from src.model.retrain import promotion_decision, sensors_to_retrain

CURRENT = {"recall_spike": 0.88, "false_alarm_rate": 0.020}


def windows(sensor_id, drifted):
    return pd.DataFrame({"sensor_id": sensor_id, "start": range(len(drifted)), "drifted": drifted})


def test_sustained_drift_triggers_a_retrain():
    assert sensors_to_retrain(windows("t1", [False, False, True, False, True])) == ["t1"]


def test_a_single_drifted_window_is_not_enough():
    assert sensors_to_retrain(windows("t1", [False, False, True, False, False])) == []


def test_only_the_most_recent_windows_count():
    old_drift = [True, True, False, False, False, False, False]
    assert sensors_to_retrain(windows("t1", old_drift), last=5) == []


def test_sensors_are_judged_separately():
    both = pd.concat([windows("t1", [True, True]), windows("t2", [False, False])])
    assert sensors_to_retrain(both) == ["t1"]


def test_no_windows_means_no_retrain():
    empty = pd.DataFrame(columns=["sensor_id", "start", "drifted"])
    assert sensors_to_retrain(empty) == []


def test_a_clearly_better_candidate_is_promoted():
    candidate = {"recall_spike": 0.88, "false_alarm_rate": 0.010}
    result = promotion_decision(CURRENT, candidate)
    assert result["promote"] is True
    assert "false alarm rate fell" in result["reasons"][0]


def test_a_candidate_that_loses_recall_is_rejected():
    candidate = {"recall_spike": 0.80, "false_alarm_rate": 0.010}
    result = promotion_decision(CURRENT, candidate)
    assert result["promote"] is False
    assert "recall fell" in result["reasons"][0]


def test_a_candidate_with_more_false_alarms_is_rejected():
    candidate = {"recall_spike": 0.95, "false_alarm_rate": 0.030}
    result = promotion_decision(CURRENT, candidate)
    assert result["promote"] is False
    assert "false alarm rate rose" in result["reasons"][0]


def test_an_equal_candidate_is_not_worth_the_change():
    result = promotion_decision(CURRENT, dict(CURRENT))
    assert result["promote"] is False
    assert "no meaningful improvement" in result["reasons"][0]


def test_an_undefined_metric_blocks_the_promotion():
    candidate = {"recall_spike": float("nan"), "false_alarm_rate": 0.010}
    result = promotion_decision(CURRENT, candidate)
    assert result["promote"] is False
    assert "undefined" in result["reasons"][0]



def test_a_one_spike_gain_is_treated_as_noise():
    current = {"recall_spike": 0.80, "false_alarm_rate": 0.020, "validation_spikes": 20}
    candidate = {"recall_spike": 0.85, "false_alarm_rate": 0.020, "validation_spikes": 20}

    result = promotion_decision(current, candidate)

    assert result["promote"] is False
    assert "noise" in result["reasons"][1]


def test_a_gain_worth_several_spikes_is_promoted():
    current = {"recall_spike": 0.80, "false_alarm_rate": 0.020, "validation_spikes": 20}
    candidate = {"recall_spike": 0.95, "false_alarm_rate": 0.020, "validation_spikes": 20}

    result = promotion_decision(current, candidate)

    assert result["promote"] is True
    assert "spike recall rose" in result["reasons"][0]