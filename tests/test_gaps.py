import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.model.features import add_features
from src.model.gaps import evaluate_dropouts, flag_gaps

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def at(seconds):
    return T0 + timedelta(seconds=seconds)


def readings_with_a_pause():
    """Sensor A: one reading per second at 0-9 s, silent 10-19 s, then 20-29 s."""
    seconds = list(range(10)) + list(range(20, 30))
    return pd.DataFrame(
        {
            "sensor_id": "A",
            "type": "temperature",
            "value": 1.0,
            "timestamp": pd.to_datetime([at(s) for s in seconds], utc=True),
        }
    )


def events(*rows):
    df = pd.DataFrame(rows, columns=["sensor_id", "fault_type", "start", "end"])
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    return df


def test_flag_gaps_flags_only_the_reading_after_the_pause():
    df = add_features(readings_with_a_pause())
    flagged = df.loc[flag_gaps(df), "timestamp"].tolist()
    assert flagged == [pd.Timestamp(at(20))]


def test_a_labelled_dropout_is_caught():
    df = add_features(readings_with_a_pause())
    labels = events(("A", "dropout", at(10), at(19)))

    result = evaluate_dropouts(df, labels)

    assert result == {"events": 1, "caught": 1, "recall": 1.0, "false_alarms": 0}


def test_a_gap_without_a_label_is_a_false_alarm():
    df = add_features(readings_with_a_pause())
    labels = events()

    result = evaluate_dropouts(df, labels)

    assert result["events"] == 0
    assert result["caught"] == 0
    assert result["false_alarms"] == 1
    assert math.isnan(result["recall"])


def test_a_dropout_at_the_end_of_the_data_cannot_be_graded():
    df = add_features(readings_with_a_pause())
    labels = events(("A", "dropout", at(30), at(39)))  # no reading comes after it

    result = evaluate_dropouts(df, labels)

    assert result["events"] == 0