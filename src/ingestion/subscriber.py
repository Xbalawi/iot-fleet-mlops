"""MQTT subscriber: listens to the v1 fleet and stores readings as Parquet.

Run from the repo root:

    python -m src.ingestion.subscriber

Configuration (environment variables, same BROKER_HOST name as v1):

    BROKER_HOST    default "localhost"
    BROKER_PORT    default 1883
    TOPIC          default "sensors/readings"
    DATA_DIR       default "data/raw"
    FLUSH_ROWS     default 50   (write after this many buffered readings)
    FLUSH_SECONDS  default 30   (or after this many seconds)
"""

import logging
import os
import signal
import sys

import paho.mqtt.client as mqtt

from src.ingestion.parser import InvalidReading, parse_reading
from src.ingestion.store import ParquetStore

log = logging.getLogger("ingestion")


def handle_message(store, payload, stats):
    """Parse one payload and hand it to the store. Never raises on bad data."""
    try:
        reading = parse_reading(payload)
    except InvalidReading as exc:
        stats["rejected"] += 1
        log.warning("rejected message: %s", exc)
        return
    stats["accepted"] += 1
    for path in store.add(reading):
        log.info("wrote %s", path)


def make_client():
    # Same paho v1/v2 compatibility trick as the v1 project.
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    host = os.environ.get("BROKER_HOST", "localhost")
    port = int(os.environ.get("BROKER_PORT", "1883"))
    topic = os.environ.get("TOPIC", "sensors/readings")

    store = ParquetStore(
        root=os.environ.get("DATA_DIR", "data/raw"),
        max_rows=int(os.environ.get("FLUSH_ROWS", "50")),
        max_age_s=float(os.environ.get("FLUSH_SECONDS", "30")),
    )
    stats = {"accepted": 0, "rejected": 0}
    client = make_client()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            log.info("connected to %s:%s, subscribing to '%s'", host, port, topic)
            # Subscribing here (not once at startup) re-subscribes after reconnects.
            client.subscribe(topic)
        else:
            log.error("connection refused: %s", reason_code)

    def on_message(client, userdata, msg):
        handle_message(store, msg.payload, stats)

    def shutdown(*_):
        # Stops loop_forever() below; the finally block does the final flush.
        client.disconnect()

    client.on_connect = on_connect
    client.on_message = on_message
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("connecting to %s:%s ...", host, port)
    # connect_async + retry_first_connection=True keeps retrying until the broker
    # is up, which is exactly where the v1 simulator gave up and exited.
    client.connect_async(host, port, keepalive=60)

    try:
        client.loop_forever(retry_first_connection=True)
    finally:
        for path in store.flush():
            log.info("final flush wrote %s", path)
        log.info("stopped. accepted=%d rejected=%d", stats["accepted"], stats["rejected"])


if __name__ == "__main__":
    sys.exit(main())
