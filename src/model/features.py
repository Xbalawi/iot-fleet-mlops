"""Turn labelled readings into rolling-window features, one row per reading."""

import numpy as np
import pandas as pd

RUN_GAP = pd.Timedelta("30s")
SHORT_WINDOW = 10
LONG_WINDOW = 50

FEATURE_COLUMNS = ["value", "diff1", "z_short", "z_long", "mean_shift", "gap_ratio"]


def assign_runs(readings, gap=RUN_GAP):
    """Add a run_id column. A new run starts after a pause longer than `gap`.

    The pause is measured on the whole timeline, all sensors together.
    Returns a copy sorted by time.
    """
    out = readings.sort_values("timestamp").reset_index(drop=True)
    out["run_id"] = (out["timestamp"].diff() > gap).cumsum()
    return out


def add_features(readings, short_window=SHORT_WINDOW, long_window=LONG_WINDOW):
    """Return a copy of readings with rolling-window features added.

    Features are computed per run and per sensor. Every window looks only at
    the readings BEFORE the current one, so a spike cannot pull its own
    baseline up and hide. The first readings of each run and sensor do not
    have enough history yet; their features are NaN (the warm-up).
    """
    df = assign_runs(readings)
    df = df.sort_values(["run_id", "sensor_id", "timestamp"]).reset_index(drop=True)

    keys = [df["run_id"], df["sensor_id"]]
    value = df["value"]
    past = value.groupby(keys).shift(1)  # the previous reading of the same sensor
    past_by_sensor = past.groupby(keys)

    mean_short = past_by_sensor.transform(lambda s: s.rolling(short_window).mean())
    std_short = past_by_sensor.transform(lambda s: s.rolling(short_window).std())
    mean_long = past_by_sensor.transform(lambda s: s.rolling(long_window).mean())
    std_long = past_by_sensor.transform(lambda s: s.rolling(long_window).std())
    std_short = std_short.replace(0, np.nan)  # a flat window has no noise to measure
    std_long = std_long.replace(0, np.nan)

    df["diff1"] = value - past
    df["z_short"] = (value - mean_short) / std_short
    df["z_long"] = (value - mean_long) / std_long
    df["mean_shift"] = (mean_short - mean_long) / std_long

    previous_time = df.groupby(["run_id", "sensor_id"])["timestamp"].shift(1)
    gap_s = (df["timestamp"] - previous_time).dt.total_seconds()
    typical_gap = gap_s.groupby(keys).transform("median")
    df["gap_ratio"] = gap_s / typical_gap
    return df


def drop_warm_up(df):
    """Keep only the rows whose features are all available."""
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)