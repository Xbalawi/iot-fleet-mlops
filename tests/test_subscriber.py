import json

import pandas as pd

from src.ingestion.store import ParquetStore
from src.ingestion.subscriber import handle_message


def msg(value, sensor="temp-01", kind="temperature"):
    return json.dumps(
        {
            "sensor_id": sensor,
            "type": kind,
            "value": value,
            "timestamp": "2026-10-13T13:46:14.769256+00:00",
        }
    ).encode()


def test_good_and_bad_messages_are_counted_and_stored(tmp_path):
    store = ParquetStore(root=tmp_path, max_rows=100, max_age_s=999)
    stats = {"accepted": 0, "rejected": 0}

    handle_message(store, msg(21.5), stats)
    handle_message(store, msg(1, sensor="occ-01", kind="occupancy"), stats)
    handle_message(store, b"garbage", stats)  # must not raise
    handle_message(store, msg("hot"), stats)  # must not raise

    assert stats == {"accepted": 2, "rejected": 2}
    store.flush()
    df = pd.concat(pd.read_parquet(f) for f in tmp_path.rglob("*.parquet"))
    assert sorted(df["sensor_id"]) == ["occ-01", "temp-01"]
