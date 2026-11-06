import json
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

from src.alerts.bridge import handle_message, make_score_fn

NOW = datetime(2026, 10, 17, 12, 0, 5, tzinfo=timezone.utc)
PAYLOAD = json.dumps(
    {"sensor_id": "temp-01", "type": "temperature", "value": 45.0, "timestamp": "2026-10-17T12:00:00+00:00"}
).encode()
ANOMALY = {"status": "ok", "is_anomaly": True, "score": 0.7}


def run(answer, payload=PAYLOAD):
    """Handle one message with a stubbed scoring service. Returns (published, stats, calls)."""
    published, calls, stats = [], [], Counter()

    def score_fn(reading):
        calls.append(reading)
        if isinstance(answer, Exception):
            raise answer
        return answer

    handle_message(payload, score_fn, published.append, stats, clock=lambda: NOW)
    return published, stats, calls


def test_an_anomaly_publishes_one_alert():
    published, stats, _ = run(ANOMALY)

    assert len(published) == 1
    assert json.loads(published[0]) == {
        "kind": "anomaly",
        "sensor_id": "temp-01",
        "type": "temperature",
        "value": 45.0,
        "timestamp": "2026-10-17T12:00:00+00:00",
        "score": 0.7,
        "detected_at": "2026-10-17T12:00:05+00:00",
    }
    assert stats["alerts"] == 1
    assert stats["ok"] == 1


def test_a_normal_reading_publishes_nothing():
    published, stats, _ = run({"status": "ok", "is_anomaly": False, "score": 0.1})
    assert published == []
    assert stats["ok"] == 1
    assert stats["alerts"] == 0


def test_a_warming_up_sensor_publishes_nothing():
    published, stats, _ = run({"status": "warming_up", "is_anomaly": None, "score": None})
    assert published == []
    assert stats["warming_up"] == 1


def test_a_sensor_without_a_model_publishes_nothing():
    published, stats, _ = run({"status": "not_modeled", "is_anomaly": None, "score": None})
    assert published == []
    assert stats["not_modeled"] == 1


def test_an_invalid_message_never_reaches_the_service():
    published, stats, calls = run(ANOMALY, payload=b"not json")
    assert calls == []
    assert published == []
    assert stats["invalid"] == 1


def test_a_service_outage_is_counted_and_does_not_crash():
    published, stats, _ = run(urllib.error.URLError("service down"))
    assert published == []
    assert stats["errors"] == 1


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return self._body


def test_the_score_function_posts_the_reading_as_json(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return FakeResponse(json.dumps(ANOMALY).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    reading = {
        "sensor_id": "temp-01",
        "type": "temperature",
        "value": 45.0,
        "timestamp": datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc),
    }

    answer = make_score_fn("http://service/predict")(reading)

    assert seen["url"] == "http://service/predict"
    assert seen["body"] == {
        "sensor_id": "temp-01",
        "type": "temperature",
        "value": 45.0,
        "timestamp": "2026-10-17T12:00:00+00:00",
    }
    assert answer == ANOMALY