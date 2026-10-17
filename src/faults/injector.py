"""Publish simulated sensor readings with separate ground-truth fault labels."""

import argparse
import json
import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

from src.faults.faults import SensorSim

log = logging.getLogger("fault_simulator")
TOPIC = "sensors/readings"
CONNECT_TIMEOUT_S = 10
SENSORS = (
    ("temp-01", "temperature"),
    ("vib-01", "vibration"),
    ("occ-01", "occupancy"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, required=True,
                        help="simulation duration in seconds")
    parser.add_argument("--interval", type=float, default=3.5,
                        help="seconds between simulation ticks")
    parser.add_argument("--labels", default="data/labels/faults.jsonl",
                        help="path for ground-truth fault events")
    args = parser.parse_args(argv)
    if args.duration < 0:
        parser.error("--duration must be non-negative")
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    return args


def make_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()


def main(argv=None, clock=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if clock is None:
        clock = lambda: datetime.now(timezone.utc)
    host = os.environ.get("BROKER_HOST", "localhost")
    port = int(os.environ.get("BROKER_PORT", "1883"))
    rng = random.Random(args.seed)
    sensors = [SensorSim(sensor_id, sensor_type, rng)
               for sensor_id, sensor_type in SENSORS]
    label_path = Path(args.labels)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    client = make_client()

    connected = threading.Event()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            connected.set()

    client.on_connect = on_connect

    log.info("connecting asynchronously to %s:%s", host, port)
    client.connect_async(host, port, keepalive=60)
    client.loop_start()
    if not connected.wait(timeout=CONNECT_TIMEOUT_S):
        client.loop_stop()
        log.error("no broker connection at %s:%s after %ss", host, port, CONNECT_TIMEOUT_S)
        return 1

    try:
        with label_path.open("a", encoding="utf-8") as labels:
            try:
                deadline = time.monotonic() + args.duration
                while time.monotonic() < deadline:
                    now = clock()
                    for sim in sensors:
                        value, event = sim.step(now)
                        if value is not None:
                            payload = {
                                "sensor_id": sim.sensor_id,
                                "type": sim.sensor_type,
                                "value": value,
                                "timestamp": now.isoformat(),
                            }
                            client.publish(TOPIC, json.dumps(payload))
                        if event is not None:
                            labels.write(event.to_json_line() + "\n")
                    time.sleep(args.interval)
            finally:
                now = clock()
                for sim in sensors:
                    event = sim.close(now)
                    if event is not None:
                        labels.write(event.to_json_line() + "\n")
    finally:
        client.loop_stop()
        client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())