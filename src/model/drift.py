"""Detect data drift: does a window of recent readings still look like the reference data?"""

import pandas as pd
from scipy.stats import ks_2samp

from src.model.data import label_readings, load_labels, load_readings
from src.model.features import assign_runs
from src.model.train import MODELED_TYPES

WINDOW_SIZE = 100  # readings per window
ALPHA = 0.001  # strict on purpose: many windows are tested

COLUMNS = ["run_id", "sensor_id", "start", "end", "ks", "p_value", "drifted", "drift_share"]


def scan_windows(reference, current, window=WINDOW_SIZE, alpha=ALPHA):
    """Compare each window of `current` with the reference values of the same sensor.

    reference: DataFrame with sensor_id and value (the known-good readings).
    current: DataFrame with sensor_id, type, value, timestamp and run_id.
    Windows are consecutive, never overlap and never cross a run. A last window
    that is not full is dropped. drift_share (share of readings labelled drift)
    is only filled when current has a fault_type column.
    """
    rows = []
    typed = current[current["type"].isin(MODELED_TYPES)]
    for (run_id, sensor_id), group in typed.groupby(["run_id", "sensor_id"]):
        reference_values = reference.loc[reference["sensor_id"] == sensor_id, "value"]
        if reference_values.empty:
            continue
        group = group.sort_values("timestamp")
        for first in range(0, len(group) - window + 1, window):
            chunk = group.iloc[first : first + window]
            test = ks_2samp(reference_values, chunk["value"])
            share = (chunk["fault_type"] == "drift").mean() if "fault_type" in chunk else float("nan")
            rows.append(
                {
                    "run_id": run_id,
                    "sensor_id": sensor_id,
                    "start": chunk["timestamp"].iloc[0],
                    "end": chunk["timestamp"].iloc[-1],
                    "ks": test.statistic,
                    "p_value": test.pvalue,
                    "drifted": bool(test.pvalue < alpha),
                    "drift_share": share,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def grade_windows(windows, min_share=0.5):
    """Grade the drift alarms against the labels.

    Drift windows: at least min_share of the readings are labelled drift.
    Clean windows: no reading labelled drift. Mixed windows are left out.
    """
    drift = windows[windows["drift_share"] >= min_share]
    clean = windows[windows["drift_share"] == 0]
    caught = int(drift["drifted"].sum())
    alarms = int(clean["drifted"].sum())
    return {
        "drift_windows": len(drift),
        "caught": caught,
        "recall": caught / len(drift) if len(drift) else float("nan"),
        "clean_windows": len(clean),
        "false_alarms": alarms,
        "false_alarm_rate": alarms / len(clean) if len(clean) else float("nan"),
    }


def main():
    labels = load_labels()
    readings = assign_runs(label_readings(load_readings(), labels))
    first_run = readings["run_id"].min()
    reference = readings[(readings["run_id"] == first_run) & (~readings["is_fault"])]
    current = readings[readings["run_id"] != first_run]

    windows = scan_windows(reference, current)
    result = grade_windows(windows)

    print(f"reference readings: {len(reference)}   windows scanned: {len(windows)}")
    print(f"drift windows caught: {result['caught']}/{result['drift_windows']}   recall: {result['recall']:.3f}")
    print(
        f"false alarms on clean windows: {result['false_alarms']}/{result['clean_windows']}"
        f"   rate: {result['false_alarm_rate']:.3f}"
    )


if __name__ == "__main__":
    main()