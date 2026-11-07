"""Bridge: readings from MQTT -> scoring service -> anomaly alerts back to MQTT.

Run from the repo root:

    python -m src.alerts.bridge

Configuration (environment variables):

    BROKER_HOST       default "localhost"
    BROKER_PORT       default 1883
    TOPIC             readings to score, default "sensors/readings"
    ALERT_TOPIC       where alerts go, default "alerts/anomaly"
    SERVICE_URL       default "http://127.0.0.1:8765/predict"
    HEALTH_URL        default: /health on the same host as SERVICE_URL
    WAIT_FOR_SERVICE  seconds to wait for the service at startup, default 60 (0 = don't wait)
"""

import http.client
import json
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urljoin

from src.ingestion.parser import InvalidReading, parse_reading
from src.ingestion.subscriber import make_client

log = logging.getLogger("alert_bridge")

ALERT_TOPIC = "alerts/anomaly"


def _utc_now():
    return datetime.now(timezone.utc)


def build_alert(reading, answer, now):
    """The JSON-ready alert for one anomalous reading."""
    return {
        "kind": "anomaly",
        "sensor_id": reading["sensor_id"],
        "type": reading["type"],
        "value": reading["value"],
        "timestamp": reading["timestamp"].isoformat(),
        "score": answer["score"],
        "detected_at": now.isoformat(),
    }


def make_score_fn(url, timeout=5):
    """A function that sends one parsed reading to the scoring service."""

    def score(reading):
        body = json.dumps(
            {
                "sensor_id": reading["sensor_id"],
                "type": reading["type"],
                "value": reading["value"],
                "timestamp": reading["timestamp"].isoformat(),
            }
        ).encode()
        request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    return score


def health_url_for(service_url):
    """/health on the same host as the scoring URL."""
    return urljoin(service_url, "/health")


def _fetch_health(url, timeout=2):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def wait_for_service(health_url, timeout, fetch=None, sleep=time.sleep, clock=time.monotonic, interval=1.0):
    """Poll the service's /health until it says ok. True if it did, False after `timeout` seconds.

    While the service starts, Docker can refuse the connection or accept it and
    hang up; all of that just means "not ready yet". fetch, sleep and clock can
    be replaced, which is how the tests run this without waiting.
    """
    fetch = fetch or _fetch_health
    deadline = clock() + timeout
    while True:
        try:
            answer = fetch(health_url)
            if isinstance(answer, dict) and answer.get("status") == "ok":
                return True
        except (OSError, http.client.HTTPException, ValueError):
            pass  # refused, hung up, timed out, an HTTP error or not JSON: not ready yet
        if clock() >= deadline:
            return False
        sleep(interval)


def handle_message(payload, score_fn, publish_fn, stats, clock=_utc_now):
    """Score one MQTT payload and publish an alert if it is an anomaly. Never raises on bad data."""
    try:
        reading = parse_reading(payload)
    except InvalidReading as exc:
        stats["invalid"] += 1
        log.warning("rejected message: %s", exc)
        return

    try:
        answer = score_fn(reading)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        stats["errors"] += 1
        log.warning("scoring failed: %s", exc)
        return

    stats[answer["status"]] += 1
    if answer["status"] == "ok" and answer["is_anomaly"]:
        alert = build_alert(reading, answer, clock())
        publish_fn(json.dumps(alert))
        stats["alerts"] += 1
        log.info("ALERT %s value=%s score=%.3f", reading["sensor_id"], reading["value"], answer["score"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    host = os.environ.get("BROKER_HOST", "localhost")
    port = int(os.environ.get("BROKER_PORT", "1883"))
    topic = os.environ.get("TOPIC", "sensors/readings")
    alert_topic = os.environ.get("ALERT_TOPIC", ALERT_TOPIC)
    url = os.environ.get("SERVICE_URL", "http://127.0.0.1:8765/predict")
    wait = float(os.environ.get("WAIT_FOR_SERVICE", "60"))
    health_url = os.environ.get("HEALTH_URL") or health_url_for(url)

    # Wait BEFORE subscribing: readings are only worth receiving once they can be scored.
    if wait > 0:
        log.info("waiting up to %.0f s for the service at %s", wait, health_url)
        try:
            ready = wait_for_service(health_url, wait)
        except KeyboardInterrupt:
            ready = False
        if not ready:
            log.error("the service is not ready, not starting")
            return 1
        log.info("the service is ready")

    stats = Counter()
    client = make_client()
    score_fn = make_score_fn(url)

    def publish(payload):
        client.publish(alert_topic, payload, qos=1)

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            log.info("connected to %s:%s, listening on '%s', alerts go to '%s'", host, port, topic, alert_topic)
            client.subscribe(topic)
        else:
            log.error("connection refused: %s", reason_code)

    def on_message(client, userdata, msg):
        handle_message(msg.payload, score_fn, publish, stats)

    def shutdown(*_):
        client.disconnect()

    client.on_connect = on_connect
    client.on_message = on_message
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    client.connect_async(host, port, keepalive=60)
    try:
        client.loop_forever(retry_first_connection=True)
    finally:
        log.info("stopped. %s", dict(stats))


if __name__ == "__main__":
    sys.exit(main())

# Old version:

""" Bridge: readings from MQTT -> scoring service -> anomaly alerts back to MQTT.

Run from the repo root:

    python -m src.alerts.bridge

Configuration (environment variables):

    BROKER_HOST    default "localhost"
    BROKER_PORT    default 1883
    TOPIC          readings to score, default "sensors/readings"
    ALERT_TOPIC    where alerts go, default "alerts/anomaly"
    SERVICE_URL    default "http://127.0.0.1:8765/predict"


import json
import logging
import os
import signal
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

from src.ingestion.parser import InvalidReading, parse_reading
from src.ingestion.subscriber import make_client

log = logging.getLogger("alert_bridge")

ALERT_TOPIC = "alerts/anomaly"


def _utc_now():
    return datetime.now(timezone.utc)


def build_alert(reading, answer, now):
    The JSON-ready alert for one anomalous reading.
    return {
        "kind": "anomaly",
        "sensor_id": reading["sensor_id"],
        "type": reading["type"],
        "value": reading["value"],
        "timestamp": reading["timestamp"].isoformat(),
        "score": answer["score"],
        "detected_at": now.isoformat(),
    }


def make_score_fn(url, timeout=5):
    A function that sends one parsed reading to the scoring service.

    def score(reading):
        body = json.dumps(
            {
                "sensor_id": reading["sensor_id"],
                "type": reading["type"],
                "value": reading["value"],
                "timestamp": reading["timestamp"].isoformat(),
            }
        ).encode()
        request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    return score


def handle_message(payload, score_fn, publish_fn, stats, clock=_utc_now):
    Score one MQTT payload and publish an alert if it is an anomaly. 
    Never raises on bad data.
    try:
        reading = parse_reading(payload)
    except InvalidReading as exc:
        stats["invalid"] += 1
        log.warning("rejected message: %s", exc)
        return

    try:
        answer = score_fn(reading)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        stats["errors"] += 1
        log.warning("scoring failed: %s", exc)
        return

    stats[answer["status"]] += 1
    if answer["status"] == "ok" and answer["is_anomaly"]:
        alert = build_alert(reading, answer, clock())
        publish_fn(json.dumps(alert))
        stats["alerts"] += 1
        log.info("ALERT %s value=%s score=%.3f", reading["sensor_id"], reading["value"], answer["score"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    host = os.environ.get("BROKER_HOST", "localhost")
    port = int(os.environ.get("BROKER_PORT", "1883"))
    topic = os.environ.get("TOPIC", "sensors/readings")
    alert_topic = os.environ.get("ALERT_TOPIC", ALERT_TOPIC)
    url = os.environ.get("SERVICE_URL", "http://127.0.0.1:8765/predict")

    stats = Counter()
    client = make_client()
    score_fn = make_score_fn(url)

    def publish(payload):
        client.publish(alert_topic, payload, qos=1)

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            log.info("connected to %s:%s, listening on '%s', alerts go to '%s'", host, port, topic, alert_topic)
            client.subscribe(topic)
        else:
            log.error("connection refused: %s", reason_code)

    def on_message(client, userdata, msg):
        handle_message(msg.payload, score_fn, publish, stats)

    def shutdown(*_):
        client.disconnect()

    client.on_connect = on_connect
    client.on_message = on_message
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    client.connect_async(host, port, keepalive=60)
    try:
        client.loop_forever(retry_first_connection=True)
    finally:
        log.info("stopped. %s", dict(stats))


if __name__ == "__main__":
    sys.exit(main()) """