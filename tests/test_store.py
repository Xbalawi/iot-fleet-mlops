from datetime import datetime, timezone

import pandas as pd

from src.ingestion.store import ParquetStore


def reading(value=20.0, ts="2026-10-13T13:46:14+00:00", sensor="temp-01", kind="temperature"):
    return {
        "sensor_id": sensor,
        "type": kind,
        "value": value,
        "timestamp": datetime.fromisoformat(ts),
    }


def read_all(root):
    files = sorted(root.rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), files


def test_no_flush_below_thresholds(tmp_path):
    store = ParquetStore(root=tmp_path, max_rows=5, max_age_s=999)
    assert store.add(reading()) == []
    assert len(store) == 1
    assert list(tmp_path.rglob("*.parquet")) == []


def test_flush_when_buffer_is_full(tmp_path):
    store = ParquetStore(root=tmp_path, max_rows=3, max_age_s=999)
    store.add(reading(1.0))
    store.add(reading(2.0))
    written = store.add(reading(3.0))
    assert len(written) == 1
    assert len(store) == 0
    df, _ = read_all(tmp_path)
    assert list(df["value"]) == [1.0, 2.0, 3.0]


def test_flush_when_buffer_is_old_enough(tmp_path):
    now = [0.0]
    store = ParquetStore(root=tmp_path, max_rows=100, max_age_s=30, clock=lambda: now[0])
    assert store.add(reading(1.0)) == []
    now[0] = 31.0
    written = store.add(reading(2.0))
    assert len(written) == 1
    df, _ = read_all(tmp_path)
    assert len(df) == 2


def test_partitioned_by_reading_date_across_midnight(tmp_path):
    store = ParquetStore(root=tmp_path, max_rows=100, max_age_s=999)
    store.add(reading(1.0, ts="2026-10-13T23:59:58+00:00"))
    store.add(reading(2.0, ts="2026-10-14T00:00:01+00:00"))
    written = store.flush()
    assert len(written) == 2
    assert (tmp_path / "date=2026-10-13").is_dir()
    assert (tmp_path / "date=2026-10-14").is_dir()


def test_schema_roundtrip(tmp_path):
    store = ParquetStore(root=tmp_path, max_rows=100, max_age_s=999)
    store.add(reading(19.48))
    store.flush()
    df, _ = read_all(tmp_path)
    assert set(df.columns) == {"sensor_id", "type", "value", "timestamp", "ingested_at"}
    assert df["value"].dtype == "float64"
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df.loc[0, "sensor_id"] == "temp-01"
    assert df.loc[0, "timestamp"] == pd.Timestamp("2026-10-13T13:46:14", tz="UTC")


def test_flush_on_empty_buffer_writes_nothing(tmp_path):
    store = ParquetStore(root=tmp_path)
    assert store.flush() == []
    assert list(tmp_path.rglob("*.parquet")) == []


def test_two_flushes_create_two_files(tmp_path):
    store = ParquetStore(root=tmp_path, max_rows=100, max_age_s=999)
    store.add(reading(1.0))
    store.flush()
    store.add(reading(2.0))
    store.flush()
    df, files = read_all(tmp_path)
    assert len(files) == 2
    assert sorted(df["value"]) == [1.0, 2.0]
