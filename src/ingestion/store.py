"""Buffered, time-partitioned Parquet storage for sensor readings.

Layout on disk (one folder per UTC day of the *reading's* timestamp)::

    data/raw/date=2026-10-13/readings-20261013T134700123456Z.parquet

Readings are buffered in memory and written in small batches. This keeps the
subscriber simple; many small files are fine for now and can be compacted
later if needed.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class ParquetStore:
    def __init__(self, root="data/raw", max_rows=50, max_age_s=30.0, clock=time.monotonic):
        self.root = Path(root)
        self.max_rows = max_rows
        self.max_age_s = max_age_s
        self._clock = clock
        self._rows = []
        self._last_flush = clock()

    def __len__(self):
        return len(self._rows)

    def add(self, reading):
        """Buffer one reading; flush if the buffer is full or old enough.

        Returns the list of files written (empty when nothing was flushed).
        """
        row = dict(reading)
        row["ingested_at"] = datetime.now(timezone.utc)
        self._rows.append(row)
        if self._should_flush():
            return self.flush()
        return []

    def _should_flush(self):
        if len(self._rows) >= self.max_rows:
            return True
        return bool(self._rows) and (self._clock() - self._last_flush) >= self.max_age_s

    def flush(self):
        """Write everything buffered to Parquet. Returns the paths written."""
        if not self._rows:
            self._last_flush = self._clock()
            return []

        df = pd.DataFrame(self._rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
        day = df["timestamp"].dt.strftime("%Y-%m-%d")

        written = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for date, part in df.groupby(day):
            folder = self.root / f"date={date}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"readings-{stamp}.parquet"
            part.to_parquet(path, index=False)
            written.append(path)

        self._rows = []
        self._last_flush = self._clock()
        return written
