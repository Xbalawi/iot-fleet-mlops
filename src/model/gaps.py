"""Detect dropouts from the pause they leave in a sensor's readings."""

import pandas as pd

from src.model.features import RUN_GAP

GAP_THRESHOLD = 3.0  # flag a reading that came more than 3x later than usual


def flag_gaps(df, threshold=GAP_THRESHOLD):
    """True for readings that arrived much later than the sensor's typical gap.

    A NaN gap_ratio (first reading of a run) compares as False, so it is never flagged.
    """
    return df["gap_ratio"] > threshold


def evaluate_dropouts(df, labels, threshold=GAP_THRESHOLD, max_wait=RUN_GAP):
    """Grade the gap rule against the labelled dropouts.

    df must be the output of add_features (warm-up rows kept: a dropout can
    happen early in a run). Returns a dict with:
    - events: dropouts that can be graded (a reading follows within max_wait)
    - caught: how many of them the rule flagged
    - recall: caught / events
    - false_alarms: flagged readings that match no labelled dropout
    """
    dropouts = labels[labels["fault_type"] == "dropout"]
    events = caught = 0
    for event in dropouts.itertuples():
        later = df[(df["sensor_id"] == event.sensor_id) & (df["timestamp"] > event.end)]
        if later.empty:
            continue  # the run ended during the dropout: nothing to grade
        following = later.loc[later["timestamp"].idxmin()]
        if following["timestamp"] - event.end > max_wait or pd.isna(following["gap_ratio"]):
            continue
        events += 1
        caught += int(following["gap_ratio"] > threshold)

    false_alarms = int(flag_gaps(df, threshold).sum()) - caught
    recall = caught / events if events else float("nan")
    return {"events": events, "caught": caught, "recall": recall, "false_alarms": false_alarms}