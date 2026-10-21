"""Load the stored readings and the fault labels, and mark which readings are faulty."""

import json
from pathlib import Path

import pandas as pd

LABEL_COLUMNS = ["sensor_id", "fault_type", "start", "end"]
READING_FAULTS = ("spike", "drift")


def load_readings(root="data/raw"):
    """Read every Parquet file under root into one DataFrame, sorted by sensor and time."""
    df = pd.read_parquet(root)
    df = df.drop(columns=["date"], errors="ignore")  # added by the day folders
    return df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)


def load_labels(path="data/labels/faults.jsonl"):
    """Read the ground-truth fault events (one JSON object per line)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    labels = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    labels["start"] = pd.to_datetime(labels["start"], utc=True)
    labels["end"] = pd.to_datetime(labels["end"], utc=True)
    return labels


def label_readings(readings, labels):
    """Return a copy of readings with two new columns.

    fault_type: 'normal', 'spike' or 'drift'
    is_fault:   True when fault_type is not 'normal'

    A reading is faulty when an event of the same sensor has a [start, end]
    window that contains the reading's timestamp (both ends included).
    Dropouts are skipped on purpose: a dropout is a missing reading, so there
    is no row to mark.
    """
    out = readings.copy()
    out["fault_type"] = "normal"
    for event in labels[labels["fault_type"].isin(READING_FAULTS)].itertuples():
        inside = (
            (out["sensor_id"] == event.sensor_id)
            & (out["timestamp"] >= event.start)
            & (out["timestamp"] <= event.end)
        )
        out.loc[inside, "fault_type"] = event.fault_type
    out["is_fault"] = out["fault_type"] != "normal"
    return out