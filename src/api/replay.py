"""Replay recorded readings through the running service and grade its answers."""

import argparse
import json
import sys
import urllib.error
import urllib.request

import pandas as pd

from src.model.data import label_readings, load_labels, load_readings
from src.model.features import assign_runs
from src.model.train import evaluate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/predict")
    return parser.parse_args(argv)


def post_reading(url, row):
    """Send one reading to the service and return its answer as a dict."""
    body = json.dumps(
        {
            "sensor_id": row.sensor_id,
            "type": row.type,
            "value": float(row.value),
            "timestamp": row.timestamp.isoformat(),
        }
    ).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def main(argv=None):
    args = parse_args(argv)
    labels = load_labels()
    readings = assign_runs(label_readings(load_readings(), labels))
    test_run = readings[readings["run_id"] == readings["run_id"].max()]  # the run the models never saw
    print(f"replaying {len(test_run)} readings to {args.url}")

    answers = []
    try:
        for n, row in enumerate(test_run.itertuples(), start=1):
            answers.append(post_reading(args.url, row))
            if n % 500 == 0:
                print(f"sent {n}/{len(test_run)}")
    except urllib.error.HTTPError as error:
        print(f"the service answered {error.code}: {error.read().decode()}")
        print("It remembers the readings it has already seen. Restart the service, then try again.")
        sys.exit(1)

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