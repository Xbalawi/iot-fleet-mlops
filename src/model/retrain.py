"""When to retrain, and when a retrained model may replace the current one."""

import math

LAST_WINDOWS = 5  # how many recent windows per sensor we look at
MIN_DRIFTED = 2  # drifted windows among those that make the drift "sustained"

MAX_RECALL_DROP = 0.02  # a candidate may lose at most this much spike recall
MAX_FALSE_ALARM_RISE = 0.005  # ... and raise the false alarm rate by at most this much
MIN_RECALL_GAIN = 0.02  # to be promoted it must gain this much recall ...
MIN_ALARM_GAIN = 0.002  # ... or cut the false alarm rate by this much
MIN_SPIKES_GAINED = 2  # a recall gain must be worth at least this many extra spikes caught

METRICS = ("recall_spike", "false_alarm_rate")


def sensors_to_retrain(windows, last=LAST_WINDOWS, min_drifted=MIN_DRIFTED):
    """Sensors whose most recent windows show sustained drift.

    windows: the output of scan_windows (needs sensor_id, start, drifted).
    One odd window is not enough: at least `min_drifted` of the last `last` windows must drift.
    """
    triggered = []
    for sensor_id, group in windows.groupby("sensor_id"):
        recent = group.sort_values("start").tail(last)
        if recent["drifted"].sum() >= min_drifted:
            triggered.append(sensor_id)
    return sorted(triggered)


def promotion_decision(
    current,
    candidate,
    max_recall_drop=MAX_RECALL_DROP,
    max_false_alarm_rise=MAX_FALSE_ALARM_RISE,
    min_recall_gain=MIN_RECALL_GAIN,
    min_alarm_gain=MIN_ALARM_GAIN,
    min_spikes_gained=MIN_SPIKES_GAINED,
):
    """Decide whether the candidate model may replace the current one.

    current, candidate: {"recall_spike": float, "false_alarm_rate": float} and, optionally,
    "validation_spikes" (how many spikes the validation data holds), all measured on the
    same labelled validation data. Returns {"promote": bool, "reasons": [str]}.
    """
    if any(math.isnan(model[name]) for model in (current, candidate) for name in METRICS):
        return {"promote": False, "reasons": ["a metric is undefined, so the models cannot be compared"]}

    recall_change = candidate["recall_spike"] - current["recall_spike"]
    alarm_change = candidate["false_alarm_rate"] - current["false_alarm_rate"]

    problems = []
    if recall_change < -max_recall_drop:
        problems.append(f"spike recall fell by {-recall_change:.3f}")
    if alarm_change > max_false_alarm_rise:
        problems.append(f"false alarm rate rose by {alarm_change:.3f}")
    if problems:
        return {"promote": False, "reasons": problems}

    # With few spikes in the validation data, one spike moves the recall a lot.
    # Require the gain to be worth several spikes, not just a bigger number.
    needed_recall_gain = min_recall_gain
    spikes = candidate.get("validation_spikes")
    if spikes:
        needed_recall_gain = max(min_recall_gain, min_spikes_gained / spikes)

    gains = []
    if recall_change >= needed_recall_gain:
        gains.append(f"spike recall rose by {recall_change:.3f}")
    if -alarm_change >= min_alarm_gain:
        gains.append(f"false alarm rate fell by {-alarm_change:.3f}")
    if gains:
        return {"promote": True, "reasons": gains}

    reasons = ["no meaningful improvement over the current model"]
    if recall_change > 0:
        reasons.append(
            f"spike recall rose by {recall_change:.3f}, but a gain below {needed_recall_gain:.3f} "
            "is within the noise"
        )
    return {"promote": False, "reasons": reasons}