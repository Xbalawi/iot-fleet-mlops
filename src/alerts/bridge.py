"""Bridge: readings from MQTT -> scoring service -> anomaly alerts back to MQTT.

Run from the repo root:

    python -m src.alerts.bridge

Configuration (environment variables):

    BROKER_HOST    default "localhost"
    BROKER_PORT    default 1883
    TOPIC          readings to score, default "sensors/readings"
    ALERT_TOPIC    where alerts go, default "alerts/anomaly"
    SERVICE_URL    default "http://127.0.0.1:8765/predict"
"""

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