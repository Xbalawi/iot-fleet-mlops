import json
from datetime import timezone

import pytest

from src.ingestion.parser import InvalidReading, parse_reading

GOOD = {
    "sensor_id": "temp-01",
    "type": "temperature",
    "value": 19.48,
    "timestamp": "2026-10-13T13:46:14.769256+00:00",
}


def payload(**overrides):
    data = {**GOOD, **overrides}
    return json.dumps(data).encode()


def test_valid_reading_is_parsed():
    r = parse_reading(payload())
    assert r["sensor_id"] == "temp-01"
    assert r["type"] == "temperature"
    assert r["value"] == 19.48
    assert r["timestamp"].tzinfo is not None
    assert r["timestamp"].utcoffset().total_seconds() == 0


def test_integer_value_becomes_float():
    r = parse_reading(payload(sensor_id="occ-01", type="occupancy", value=1))
    assert r["value"] == 1.0
    assert isinstance(r["value"], float)


def test_str_payload_is_accepted():
    assert parse_reading(json.dumps(GOOD))["sensor_id"] == "temp-01"


def test_z_suffix_timestamp():
    r = parse_reading(payload(timestamp="2026-10-13T13:46:14Z"))
    assert r["timestamp"].tzinfo == timezone.utc


def test_naive_timestamp_is_assumed_utc():
    r = parse_reading(payload(timestamp="2026-10-13T13:46:14"))
    assert r["timestamp"].tzinfo == timezone.utc
    assert r["timestamp"].hour == 13


def test_other_offset_is_converted_to_utc():
    r = parse_reading(payload(timestamp="2026-10-13T15:00:00+02:00"))
    assert r["timestamp"].hour == 13


@pytest.mark.parametrize("field", ["sensor_id", "type", "value", "timestamp"])
def test_missing_field_is_rejected(field):
    data = dict(GOOD)
    del data[field]
    with pytest.raises(InvalidReading, match="missing"):
        parse_reading(json.dumps(data))


@pytest.mark.parametrize("bad", ["abc", None, True, [1], {"x": 1}])
def test_non_numeric_value_is_rejected(bad):
    with pytest.raises(InvalidReading, match="not numeric"):
        parse_reading(payload(value=bad))


def test_malformed_json_is_rejected():
    with pytest.raises(InvalidReading, match="malformed JSON"):
        parse_reading(b"{not json")


def test_non_object_json_is_rejected():
    with pytest.raises(InvalidReading, match="JSON object"):
        parse_reading(b"[1, 2, 3]")


def test_invalid_utf8_is_rejected():
    with pytest.raises(InvalidReading, match="UTF-8"):
        parse_reading(b"\xff\xfe\x00")


def test_bad_timestamp_is_rejected():
    with pytest.raises(InvalidReading, match="timestamp"):
        parse_reading(payload(timestamp="yesterday-ish"))


def test_empty_sensor_id_is_rejected():
    with pytest.raises(InvalidReading, match="sensor_id"):
        parse_reading(payload(sensor_id=""))
