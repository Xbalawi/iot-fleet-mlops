"""Replay recorded readings through the running service and grade its answers."""

import argparse
import http.client
import json
import sys
import urllib.parse

import pandas as pd

from src.model.data import label_readings, load_labels, load_readings
from src.model.features import assign_runs
from src.model.train import evaluate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/predict")
    return parser.parse_args(argv)


class ServiceError(Exception):
    def __init__(self, status, text):
        super().__init__(f"{status}: {text}")
        self.status = status
        self.text = text


class Client:
    """One connection reused for every reading (a new connection per reading exhausts Windows ports)."""

    def __init__(self, url):
        parts = urllib.parse.urlsplit(url)
        self.path = parts.path or "/"
        self.connection = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=30)

    def post(self, row):
        body = json.dumps(
            {
                "sensor_id": row.sensor_id,
                "type": row.type,
                "value": float(row.value),
                "timestamp": row.timestamp.isoformat(),
            }
        )
        self.connection.request("POST", self.path, body, {"Content-Type": "application/json"})
        response = self.connection.getresponse()
        text = response.read().decode()  # always read the full body, or the connection can't be reused
        if response.status != 200:
            raise ServiceError(response.status, text)
        return json.loads(text)

    def close(self):
        self.connection.close()


def main(argv=None):
    args = parse_args(argv)
    labels = load_labels()
    readings = assign_runs(label_readings(load_readings(), labels))
    test_run = readings[readings["run_id"] == readings["run_id"].max()]  # the run the models never saw
    print(f"replaying {len(test_run)} readings to {args.url}")

    client = Client(args.url)
    answers = []
    try:
        for n, row in enumerate(test_run.itertuples(), start=1):
            answers.append(client.post(row))
            if n % 500 == 0:
                print(f"sent {n}/{len(test_run)}")
    except ServiceError as error:
        print(f"the service answered {error.status}: {error.text}")
        print("It remembers the readings it has already seen. Restart the service, then try again.")
        sys.exit(1)
    finally:
        client.close()

    answers = pd.DataFrame(answers, index=test_run.index)
    scored = test_run.copy()
    scored["status"] = answers["status"]
    scored["is_flagged"] = answers["is_anomaly"].eq(True)

    print(scored["status"].value_counts().to_string())
    table, summary = evaluate(scored[scored["status"] == "ok"])
    print(table.round(3).to_string(index=False))
    print(f"precision: {summary['precision']:.3f}   false alarm rate: {summary['false_alarm_rate']:.3f}")


if __name__ == "__main__":
    main()