from datetime import datetime, timedelta, timezone

import pandas as pd

from src.faults.faults import FaultEvent
from src.ingestion.store import ParquetStore
from src.model.data import label_readings, load_labels, load_readings

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def at(seconds):
    return T0 + timedelta(seconds=seconds)


def readings(rows):
    """rows: (sensor_id, seconds) pairs -> a readings DataFrame."""
    return pd.DataFrame(
        {
            "sensor_id": [sensor for sensor, _ in rows],
            "type": "temperature",
            "value": 20.0,
            "timestamp": pd.to_datetime([at(sec) for _, sec in rows], utc=True),
        }
    )


def events(rows):
    """rows: (sensor_id, fault_type, start_s, end_s) -> a labels DataFrame."""
    return pd.DataFrame(
        {
            "sensor_id": [r[0] for r in rows],
            "fault_type": [r[1] for r in rows],
            "start": pd.to_datetime([at(r[2]) for r in rows], utc=True),
            "end": pd.to_datetime([at(r[3]) for r in rows], utc=True),
        }
    )


def test_spike_marks_only_the_matching_reading():
    out = label_readings(readings([("A", 0), ("A", 1), ("A", 2)]),
                         events([("A", "spike", 1, 1)]))
    assert list(out["fault_type"]) == ["normal", "spike", "normal"]


def test_drift_window_includes_both_ends():
    out = label_readings(readings([("A", s) for s in range(6)]),
                         events([("A", "drift", 1, 3)]))
    assert list(out["fault_type"]) == [
        "normal", "drift", "drift", "drift", "normal", "normal"
    ]


def test_an_event_only_affects_its_own_sensor():
    out = label_readings(readings([("A", 1), ("B", 1)]),
                         events([("A", "spike", 1, 1)]))
    assert list(out["fault_type"]) == ["spike", "normal"]


def test_dropout_events_do_not_mark_any_reading():
    out = label_readings(readings([("A", s) for s in range(4)]),
                         events([("A", "dropout", 1, 2)]))
    assert set(out["fault_type"]) == {"normal"}


def test_no_labels_means_everything_is_normal():
    out = label_readings(readings([("A", 0), ("A", 1)]), events([]))
    assert set(out["fault_type"]) == {"normal"}
    assert not out["is_fault"].any()


def test_is_fault_matches_fault_type():
    out = label_readings(readings([("A", 0), ("A", 1)]),
                         events([("A", "spike", 1, 1)]))
    assert list(out["is_fault"]) == [False, True]


def test_input_readings_are_not_modified():
    original = readings([("A", 0), ("A", 1)])
    label_readings(original, events([("A", "spike", 1, 1)]))
    assert "fault_type" not in original.columns


def test_load_labels_parses_jsonl(tmp_path):
    path = tmp_path / "faults.jsonl"
    path.write_text(
        FaultEvent("temp-01", "spike", at(1), at(1)).to_json_line() + "\n"
        + FaultEvent("vib-01", "dropout", at(2), at(5)).to_json_line() + "\n",
        encoding="utf-8",
    )

    labels = load_labels(path)

    assert list(labels.columns) == ["sensor_id", "fault_type", "start", "end"]
    assert list(labels["fault_type"]) == ["spike", "dropout"]
    assert str(labels["start"].dt.tz) == "UTC"
    assert labels.loc[1, "end"] == pd.Timestamp(at(5))


def test_load_labels_on_an_empty_file(tmp_path):
    path = tmp_path / "faults.jsonl"
    path.write_text("", encoding="utf-8")

    labels = load_labels(path)

    assert len(labels) == 0
    assert list(labels.columns) == ["sensor_id", "fault_type", "start", "end"]


def test_load_readings_reads_all_day_folders_sorted_without_the_date_column(tmp_path):
    store = ParquetStore(root=tmp_path, max_rows=100, max_age_s=999)
    store.add({"sensor_id": "B", "type": "t", "value": 1.0, "timestamp": at(5)})
    store.add({"sensor_id": "A", "type": "t", "value": 2.0, "timestamp": at(86400 + 1)})
    store.add({"sensor_id": "A", "type": "t", "value": 3.0, "timestamp": at(1)})
    store.flush()

    df = load_readings(tmp_path)

    assert "date" not in df.columns
    assert list(df["sensor_id"]) == ["A", "A", "B"]
    assert list(df["value"]) == [3.0, 2.0, 1.0]


def test_a_spike_survives_the_parquet_and_json_round_trip(tmp_path):
    exact = T0 + timedelta(microseconds=769256)  # microseconds, like real ticks
    store = ParquetStore(root=tmp_path / "raw", max_rows=100, max_age_s=999)
    store.add({"sensor_id": "temp-01", "type": "temperature", "value": 45.0, "timestamp": exact})
    store.add({"sensor_id": "temp-01", "type": "temperature", "value": 21.0,
               "timestamp": exact + timedelta(seconds=1)})
    store.flush()
    labels_path = tmp_path / "faults.jsonl"
    labels_path.write_text(
        FaultEvent("temp-01", "spike", exact, exact).to_json_line() + "\n", encoding="utf-8"
    )

    out = label_readings(load_readings(tmp_path / "raw"), load_labels(labels_path))

    assert list(out["fault_type"]) == ["spike", "normal"]