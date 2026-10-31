from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.model.report import save_drift_report, window_readings

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def test_window_readings_returns_exactly_the_window():
    current = pd.DataFrame(
        {
            "run_id": 0,
            "sensor_id": ["a"] * 6 + ["b"] * 6,
            "value": np.arange(12.0),
            "timestamp": pd.to_datetime([T0 + timedelta(seconds=i % 6) for i in range(12)], utc=True),
        }
    )
    window = pd.Series(
        {
            "run_id": 0,
            "sensor_id": "a",
            "start": pd.Timestamp(T0 + timedelta(seconds=2)),
            "end": pd.Timestamp(T0 + timedelta(seconds=4)),
        }
    )

    result = window_readings(current, window)

    assert result["value"].tolist() == [2.0, 3.0, 4.0]


def test_the_drift_report_is_written_as_html(tmp_path):
    rng = np.random.default_rng(0)
    path = save_drift_report(rng.normal(24, 1, 300), rng.normal(27, 1, 100), tmp_path / "out" / "drift.html")

    assert path.exists()
    assert path.stat().st_size > 1000
    assert "<html" in path.read_text(encoding="utf-8").lower()