"""Run the project's drift monitor on SKAB's real sustained faults.

  python -m scripts.eval_skab_drift

Uses scan_windows and grade_windows from src.model.drift unchanged: KS test of
each 100-second window against a reference, alpha 0.001.

global reference    the anomaly-free recording
per-file reference  the first normal seconds of the file itself
Windows start where each file's calibration window ends, so both references
are graded on exactly the same windows (and the same files as eval_skab_variants).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.eval_skab import COLUMNS, VOTES, read
from scripts.eval_skab_variants import CAL_SECONDS, MIN_CAL_END
from src.model.drift import WINDOW_SIZE, grade_windows, scan_windows


def reference_frame(frame, last_row=None):
    """The known-good readings in the shape scan_windows expects: sensor_id and value."""
    part = frame if last_row is None else frame.iloc[:last_row]
    return pd.concat(
        [pd.DataFrame({"sensor_id": c, "value": part[c].astype(float).to_numpy()}) for c in COLUMNS],
        ignore_index=True,
    )


def current_frame(frame, file_id, first_row):
    """The readings to scan, in long format. `type` only has to pass scan_windows' sensor-type filter."""
    part = frame.iloc[first_row:]
    fault = np.where(part["anomaly"].astype(bool).to_numpy(), "drift", "normal")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "run_id": file_id,
                    "sensor_id": c,
                    "type": "temperature",
                    "value": part[c].astype(float).to_numpy(),
                    "timestamp": part["datetime"].to_numpy(),
                    "fault_type": fault,
                }
            )
            for c in COLUMNS
        ],
        ignore_index=True,
    )


def alerts_of(windows):
    """One row per window start: how many columns drifted, and the window's fault share."""
    return (
        windows.groupby("start")
        .agg(columns_drifted=("drifted", "sum"), share=("drift_share", "first"), end=("end", "first"))
        .reset_index()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/skab")
    args = parser.parse_args(argv)
    root = Path(args.data)

    global_reference = reference_frame(read(root / "anomaly-free" / "anomaly-free.csv"))

    per_column = {"global": [], "per-file": []}
    per_file = {"global": [], "per-file": []}  # (alerts table, first anomaly time)
    paths = sorted(p for folder in ("valve1", "valve2", "other") for p in (root / folder).glob("*.csv"))
    used = 0
    for file_id, path in enumerate(paths):
        frame = read(path)
        first_anomaly = int(frame["anomaly"].astype(bool).to_numpy().argmax())
        cal_end = min(CAL_SECONDS, first_anomaly)
        if cal_end < MIN_CAL_END:
            continue
        used += 1
        current = current_frame(frame, file_id, cal_end)
        anomaly_start = frame["datetime"].iloc[first_anomaly]
        references = {"global": global_reference, "per-file": reference_frame(frame, last_row=cal_end)}
        for name, reference in references.items():
            windows = scan_windows(reference, current)
            per_column[name].append(windows)
            per_file[name].append((alerts_of(windows), anomaly_start))

    print(f"scored {used} files, windows of {WINDOW_SIZE} s starting after each file's calibration window\n")

    print("Per sensor column (window level, pooled over files)")
    rows = []
    for name, parts in per_column.items():
        pooled = pd.concat(parts, ignore_index=True)
        for column in COLUMNS:
            result = grade_windows(pooled[pooled["sensor_id"] == column])
            rows.append(
                {
                    "reference": name,
                    "column": column,
                    "fault_windows": result["drift_windows"],
                    "recall": result["recall"],
                    "clean_windows": result["clean_windows"],
                    "false_alarm": result["false_alarm_rate"],
                }
            )
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nAlert = at least k columns drifted in the same window")
    rows = []
    for name, parts in per_file.items():
        pooled = pd.concat([alerts for alerts, _ in parts], ignore_index=True)
        fault = pooled[pooled["share"] >= 0.5]
        clean = pooled[pooled["share"] == 0]
        for k in VOTES:
            detected, delays, false_windows = 0, [], []
            for alerts, anomaly_start in parts:
                fires = alerts["columns_drifted"] >= k
                inside = alerts[fires & (alerts["share"] > 0)]
                if len(inside):
                    detected += 1
                    delays.append((inside["end"].iloc[0] - anomaly_start).total_seconds())
                false_windows.append(int((fires & (alerts["share"] == 0)).sum()))
            rows.append(
                {
                    "reference": name,
                    "alert when": f">= {k} of {len(COLUMNS)}",
                    "fault_recall": (fault["columns_drifted"] >= k).mean(),
                    "false_alarm": (clean["columns_drifted"] >= k).mean(),
                    "files": f"{detected}/{used}",
                    "delay_s": float(np.median(delays)) if delays else float("nan"),
                    "false_windows/file": round(float(np.mean(false_windows)), 2),
                }
            )
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()