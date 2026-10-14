"""Validation and parsing of raw MQTT payloads from iot-fleet-monitor v1.

Expected payload (published on topic ``sensors/readings``)::

    {"sensor_id": "temp-01", "type": "temperature",
     "value": 19.48, "timestamp": "2026-10-13T13:46:14.769256+00:00"}
"""

import json
from datetime import datetime, timezone

REQUIRED_FIELDS = ("sensor_id", "type", "value", "timestamp")


class InvalidReading(ValueError):
    """Raised when a payload cannot be turned into a clean reading."""


def parse_reading(payload):
    """Turn a raw MQTT payload (bytes or str) into a clean reading dict.

    The returned ``timestamp`` is a timezone-aware UTC datetime and ``value``
    is always a float (occupancy 0/1 becomes 0.0/1.0).

    Raises InvalidReading with a human-readable reason on any problem.
    """
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidReading(f"payload is not valid UTF-8: {exc}") from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InvalidReading(f"malformed JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise InvalidReading(f"expected a JSON object, got {type(data).__name__}")

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise InvalidReading(f"missing fields: {missing}")

    value = data["value"]
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidReading(f"'value' is not numeric: {value!r}")

    for name in ("sensor_id", "type"):
        if not isinstance(data[name], str) or not data[name]:
            raise InvalidReading(f"'{name}' must be a non-empty string")

    return {
        "sensor_id": data["sensor_id"],
        "type": data["type"],
        "value": float(value),
        "timestamp": _parse_timestamp(data["timestamp"]),
    }


def _parse_timestamp(raw):
    if not isinstance(raw, str):
        raise InvalidReading(f"'timestamp' must be a string, got {raw!r}")
    try:
        # "Z" suffix is not accepted by fromisoformat before Python 3.11.
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidReading(f"invalid timestamp {raw!r}") from exc
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)  # assume UTC when none is given
    return ts.astimezone(timezone.utc)
