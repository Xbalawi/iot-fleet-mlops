import http.client
import json
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

from src.alerts.bridge import handle_message, health_url_for, make_score_fn, wait_for_service

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


class FakeClock:
    """Sleeping moves time forward; nothing really waits."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def flaky(*failures, then=None):
    """A fetch that raises the given errors one by one, then answers `then`."""
    pending = list(failures)
    calls = []

    def fetch(url):
        calls.append(url)
        if pending:
            raise pending.pop(0)
        return then

    fetch.calls = calls
    return fetch


def wait(fetch, timeout=10):
    clock = FakeClock()
    ready = wait_for_service("http://svc/health", timeout, fetch=fetch, sleep=clock.sleep, clock=clock)
    return ready, clock


def test_a_service_that_is_ready_is_not_waited_for():
    ready, clock = wait(flaky(then={"status": "ok"}))
    assert ready is True
    assert clock.now == 0


def test_the_bridge_waits_through_refused_connections():
    fetch = flaky(urllib.error.URLError("refused"), urllib.error.URLError("refused"), then={"status": "ok"})
    ready, clock = wait(fetch)
    assert ready is True
    assert len(fetch.calls) == 3
    assert clock.now == 2


def test_a_connection_that_is_closed_without_an_answer_is_not_ready_yet():
    # what Docker does while the app inside is still starting
    fetch = flaky(http.client.RemoteDisconnected("closed"), then={"status": "ok"})
    ready, _ = wait(fetch)
    assert ready is True


def test_the_bridge_gives_up_after_the_timeout():
    ready, clock = wait(flaky(*[urllib.error.URLError("refused")] * 100), timeout=5)
    assert ready is False
    assert clock.now == 5


def test_an_answer_that_is_not_ok_counts_as_not_ready():
    for answer in ({"status": "starting"}, [], None):
        ready, _ = wait(flaky(then=answer), timeout=2)
        assert ready is False


def test_the_health_url_is_on_the_same_host_as_the_scoring_url():
    assert health_url_for("http://127.0.0.1:8765/predict") == "http://127.0.0.1:8765/health"
    assert health_url_for("http://svc:8000/predict") == "http://svc:8000/health"


def test_the_default_check_reads_the_health_endpoint(monkeypatch):
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        return FakeResponse(b'{"status": "ok"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert wait_for_service("http://svc/health", 1) is True
    assert seen["url"] == "http://svc/health"