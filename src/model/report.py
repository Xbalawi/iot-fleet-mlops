"""Save an Evidently drift report for the window that drifted the most."""

from pathlib import Path

import pandas as pd

from src.model.data import label_readings, load_labels, load_readings
from src.model.drift import scan_windows
from src.model.features import assign_runs


def window_readings(current, window):
    """The readings of the window described by one row of scan_windows."""
    inside = (
        (current["run_id"] == window.run_id)
        & (current["sensor_id"] == window.sensor_id)
        & (current["timestamp"] >= window.start)
        & (current["timestamp"] <= window.end)
    )
    return current[inside]


def save_drift_report(reference_values, current_values, path):
    """Write an HTML report comparing two series of readings. Returns the path."""
    # imported here so the rest of this module works without Evidently installed
    from evidently import Dataset, Report
    from evidently.presets import DataDriftPreset

    reference = pd.DataFrame({"value": pd.Series(reference_values).to_numpy()})
    current = pd.DataFrame({"value": pd.Series(current_values).to_numpy()})

    report = Report([DataDriftPreset()])
    result = report.run(Dataset.from_pandas(current), Dataset.from_pandas(reference))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(path))
    return path


def main():
    labels = load_labels()
    readings = assign_runs(label_readings(load_readings(), labels))
    first_run = readings["run_id"].min()
    reference = readings[(readings["run_id"] == first_run) & (~readings["is_fault"])]
    current = readings[readings["run_id"] != first_run]

    windows = scan_windows(reference, current)
    worst = windows.sort_values("ks", ascending=False).iloc[0]
    window = window_readings(current, worst)
    reference_values = reference.loc[reference["sensor_id"] == worst.sensor_id, "value"]

    path = save_drift_report(reference_values, window["value"], f"reports/drift_{worst.sensor_id}.html")
    print(f"sensor {worst.sensor_id}, window {worst.start} to {worst.end}")
    print(f"KS statistic {worst.ks:.3f}, p-value {worst.p_value:.2e}, drift share {worst.drift_share:.2f}")
    print(f"report saved to {path}")


if __name__ == "__main__":
    main()