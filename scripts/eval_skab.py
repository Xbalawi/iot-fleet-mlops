"""Grade the project's Isolation Forest on SKAB, a real public benchmark.

  python -m scripts.fetch_skab      # once, downloads the data
  python -m scripts.eval_skab

Training: anomaly-free.csv only (no labelled row is used for training).
Testing:  each of the 34 experiment files, as an independent run, after the
          same 50-reading warm-up the service has.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.model.features import LONG_WINDOW, add_features
from src.model.train import CONTAMINATION, MODEL_FEATURES, N_ESTIMATORS, SEED

COLUMNS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]
VOTES = (1, 2, 3)  # alert when at least this many columns flag the same second


def read(path):
    return pd.read_csv(path, sep=";", parse_dates=["datetime"])


def features_of(frame):
    """Long format (one row per column and second) with the project's rolling features. One run per call."""
    parts = [
        pd.DataFrame(
            {
                "sensor_id": column,
                "timestamp": frame["datetime"],
                "value": frame[column].astype(float),
                "row": np.arange(len(frame)),
            }
        )
        for column in COLUMNS
    ]
    long = pd.concat(parts, ignore_index=True)
    long["run_id"] = 0
    return add_features(long)


def train(path):
    """One Isolation Forest per column, fitted on the anomaly-free recording."""
    features = features_of(read(path))
    models = {}
    for column in COLUMNS:
        part = features[features["sensor_id"] == column].dropna(subset=MODEL_FEATURES)
        models[column] = IsolationForest(
            n_estimators=N_ESTIMATORS, contamination=CONTAMINATION, random_state=SEED
        ).fit(part[MODEL_FEATURES])
        print(f"trained {column}: {len(part)} rows")
    return models


def flags_of(models, frame):
    """A boolean table: one row per second, one column per sensor column. Warm-up rows are never flagged."""
    features = features_of(frame)
    table = pd.DataFrame(False, index=range(len(frame)), columns=COLUMNS)
    for column, model in models.items():
        part = features[features["sensor_id"] == column].dropna(subset=MODEL_FEATURES)
        part = part[part["row"] >= LONG_WINDOW]
        table.loc[part["row"].to_numpy(), column] = model.predict(part[MODEL_FEATURES]) == -1
    table.iloc[:LONG_WINDOW] = False
    return table


def metrics(flagged, truth):
    """Row-level precision, recall and false alarm rate."""
    caught = int((flagged & truth).sum())
    return {
        "flagged": int(flagged.sum()),
        "precision": caught / flagged.sum() if flagged.sum() else float("nan"),
        "recall": caught / truth.sum() if truth.sum() else float("nan"),
        "false_alarm_rate": float((flagged & ~truth).sum() / (~truth).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/skab")
    args = parser.parse_args(argv)
    root = Path(args.data)

    models = train(root / "anomaly-free" / "anomaly-free.csv")

    files = sorted(p for folder in ("valve1", "valve2", "other") for p in (root / folder).glob("*.csv"))
    flags, truths, spans = [], [], []
    for path in files:
        frame = read(path)
        table = flags_of(models, frame)
        truth = frame["anomaly"].astype(bool).iloc[LONG_WINDOW:].reset_index(drop=True)
        flags.append(table.iloc[LONG_WINDOW:].reset_index(drop=True))
        truths.append(truth)
        spans.append(path.relative_to(root).as_posix())
    print(f"\nscored {len(files)} files, warm-up of {LONG_WINDOW} s skipped in each\n")

    all_flags = pd.concat(flags, ignore_index=True)
    all_truth = pd.concat(truths, ignore_index=True)

    rows = []
    for column in COLUMNS:
        rows.append({"rule": f"only {column}", **metrics(all_flags[column], all_truth)})
    for k in VOTES:
        rows.append({"rule": f"at least {k} of {len(COLUMNS)} columns", **metrics(all_flags.sum(axis=1) >= k, all_truth)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nPer file (an alert = the rule fires on that second)")
    summary = []
    for k in VOTES:
        detected, delays, false_alarm_seconds = 0, [], []
        for table, truth in zip(flags, truths):
            alert = table.sum(axis=1) >= k
            hits = alert & truth
            if hits.any():
                detected += 1
                delays.append(int(hits.to_numpy().argmax() - truth.to_numpy().argmax()))
            false_alarm_seconds.append(int((alert & ~truth).sum()))
        summary.append(
            {
                "rule": f"at least {k} of {len(COLUMNS)}",
                "files_detected": f"{detected}/{len(files)}",
                "median_delay_s": float(np.median(delays)) if delays else float("nan"),
                "false_alarm_s_per_file": round(float(np.mean(false_alarm_seconds)), 1),
            }
        )
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()