# iot-fleet-mlops

Closed-loop MLOps for IoT fleets: drift detection, automatic retraining, and
shadow-mode rollout, built on top of [iot-fleet-monitor](https://github.com/Xbalawi/iot-fleet-monitor) v1.0.

**Status:** Week 1, Days 1-2 done (ingestion + fault injection). Next: features and a baseline anomaly model.

## Built on iot-fleet-monitor v1.0

The v1 project is included as a submodule, pinned at `v1.0`, and is never modified.
This repo only talks to it through MQTT.

```bash
git clone --recurse-submodules <url>
```

or, after a normal clone:

```bash
git submodule update --init
```

## Architecture

```
 iot-fleet-monitor (v1.0)             iot-fleet-mlops (this repo)
 ┌─────────────────────┐   MQTT    ┌─────────────────────────────────┐
 │ Mosquitto broker    │──────────▶│ subscriber ─▶ Parquet (by day)  │
 │ Flask + dashboard   │           │                                 │
 └─────────────────────┘           │ fault injector ─▶ MQTT          │
            ▲                      │        └─▶ labels (faults.jsonl)│
            └── sensors/readings ──┴─────────────────────────────────┘
```

v1's simulator is replaced by the fault injector during experiments. It publishes the
same messages, with faults injected on purpose.

## Results (baseline, test run)

Isolation Forest trained on the first run, scored on the second run, per sensor type.
Dropouts use a gap rule (a reading arriving more than 3x later than usual), because a missing row cannot be scored.

| Fault | Detector | Result |
|---|---|---|
| Spike | Isolation Forest | 43 / 49 caught (recall 0.88) |
| Drift | Isolation Forest | 7 / 357 caught (recall 0.02, chance level) |
| Dropout | Gap rule | 16 / 16 caught, 0 false alarms (both runs, no training needed) |

False alarm rate on normal readings: 0.3%.

## Experiment tracking

Every training run is recorded in MLflow (parameters, metrics and the trained models):

    python -m src.model.train
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

Then open http://localhost:5001.

**Takeaway:** a point-anomaly model catches sudden faults but is blind to slow drift.
Drift is the case for monitoring the data distribution over time (Evidently, Week 2).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                      # 41 tests
```

1. Start v1's broker and backend, and stop its simulator so there is only one publisher:

```bash
   cd vendor/iot-fleet-monitor
   docker compose up -d
   docker compose stop simulator
```

2. Terminal A, the subscriber (stop it with Ctrl+C, which flushes what is buffered):

```bash
   python -m src.ingestion.subscriber
```

3. Terminal B, the fault injector (5 minutes, 5 ticks per second):

```bash
   python -m src.faults.injector --seed 42 --duration 300 --interval 0.2
```

| Setting | Where | Default |
|---|---|---|
| `BROKER_HOST`, `BROKER_PORT` | env, both programs | `localhost`, `1883` |
| `TOPIC`, `DATA_DIR`, `FLUSH_ROWS`, `FLUSH_SECONDS` | env, subscriber | `sensors/readings`, `data/raw`, `50`, `30` |
| `--seed`, `--duration`, `--interval`, `--labels` | injector arguments | `42`, required, `3.5`, `data/labels/faults.jsonl` |

## Data

Topic `sensors/readings`, one JSON message per reading:

```json
{"sensor_id": "temp-01", "type": "temperature", "value": 19.48,
 "timestamp": "2026-10-13T13:46:14.769256+00:00"}
```

- **Readings:** `data/raw/date=YYYY-MM-DD/readings-<timestamp>.parquet`, partitioned by the UTC day of the
  reading and written in batches. Read them all with `pd.read_parquet("data/raw")`.
- **Labels (ground truth):** `data/labels/faults.jsonl`, one fault per line with `sensor_id`, `fault_type`,
  `start` and `end`. Labels are never sent over MQTT, so a model cannot see them.
- Readings and labels are paired. To start a clean experiment, delete `data/raw` and `data/labels` together.

## Fault injection

| Fault | Applies to | Behaviour |
|---|---|---|
| Spike | temperature, vibration | One reading outside the normal range. Temperature: 31-39 or 11-17 (inside v1's static limits, so only a model can catch them) or 41-50 (v1 catches these). Vibration: 6-10. |
| Drift | temperature | 150 readings with an offset growing by 0.05 per reading (up to about +7.5 °C) on top of the normal noise. Stays inside v1's limits. |
| Dropout | all sensors | No readings for 10 ticks. |

Normal ranges are copied from v1: temperature 18-30, vibration 0-5, occupancy 0 or 1.
Each tick, a healthy sensor starts a spike with probability 0.02, and a drift or dropout with
probability 0.002 each. All randomness comes from `random.Random(seed)`.

Example run (300 s, `--interval 0.2`): 61 faults (50 spikes, 8 dropouts, 3 drifts) and 4,387 readings,
which equals 3 sensors x 1,489 ticks minus 8 dropouts x 10 readings.

## Design decisions

- **Labels stay out of the stream.** The injector writes them to a separate file, so a model cannot cheat.
- **Seeded randomness.** The same seed gives the same fault sequence.
- **Time compression.** `--interval` makes training data in minutes instead of hours.
- **No silent data loss.** The injector waits for the broker connection and exits with code 1 if there is none.
  The subscriber rejects and counts malformed messages instead of crashing.
- **Logic is separate from I/O.** The parser, the store and the fault state machine are plain functions and
  classes, so the 41 tests run in seconds without a broker.

## Roadmap

- [x] Repo, v1 submodule, MQTT subscriber, Parquet storage
- [x] Fault injector with ground-truth labels
- [x] Rolling-window features and Isolation Forest baseline
- [x] MLflow tracking, FastAPI serving in Docker, CI with GitHub Actions
- [ ] Drift monitoring and drift-triggered retraining with a promotion rule
- [ ] Shadow-mode rollout with automatic rollback, validation on a real dataset, public deployment