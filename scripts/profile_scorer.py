"""Time Scorer.score and the parts it is made of.

Run from the repo root:  python -m scripts.profile_scorer
"""

import statistics
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.model.features import MODEL_FEATURES, add_features
from src.model.scorer import Scorer

WARM_UP = 50
MEASURED = 300
T0 = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)


def make_readings(n):
    rng = np.random.default_rng(0)
    return [
        {
            "sensor_id": "temp-01",
            "type": "temperature",
            "value": float(24 + rng.normal(0, 1)),
            "timestamp": T0 + timedelta(seconds=i),
        }
        for i in range(n)
    ]


def median_ms(fn, repeat=30):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def old_features(window):
    """What the scorer used to do for every reading: a DataFrame plus add_features."""
    frame = pd.DataFrame(window)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["run_id"] = 0
    return add_features(frame).iloc[[-1]][MODEL_FEATURES]


def main():
    readings = make_readings(WARM_UP + MEASURED)
    scorer = Scorer.from_dir("model_store")
    for reading in readings[:WARM_UP]:
        scorer.score(reading)

    times, statuses = [], {}
    for reading in readings[WARM_UP:]:
        start = time.perf_counter()
        answer = scorer.score(reading)
        times.append((time.perf_counter() - start) * 1000)
        statuses[answer["status"]] = statuses.get(answer["status"], 0) + 1

    print(f"scorer.score over {MEASURED} readings: {statuses}")
    print(
        f"  median {statistics.median(times):.1f} ms"
        f"   95th percentile {np.percentile(times, 95):.1f} ms"
    )

    window = readings[: WARM_UP + 1]
    features = old_features(window)
    model = scorer.models["temperature"]
    print("the parts, timed separately:")
    print(f"  DataFrame + add_features (old way)  {median_ms(lambda: old_features(window)):6.1f} ms")
    print(f"  model.predict                        {median_ms(lambda: model.predict(features)):6.1f} ms")
    print(f"  model.score_samples                  {median_ms(lambda: model.score_samples(features)):6.1f} ms")


if __name__ == "__main__":
    main()