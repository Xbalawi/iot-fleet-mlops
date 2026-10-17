import json
import random
from datetime import datetime, timedelta, timezone

from src.faults.faults import FaultEvent, NORMAL_RANGES, SensorSim
from src.ingestion.parser import parse_reading


class SequenceRng:
    def __init__(self, rolls):
        self.rolls = iter(rolls)
        self.random_calls = 0

    def random(self):
        self.random_calls += 1
        return next(self.rolls)

    def uniform(self, low, high):
        return low

    def choice(self, values):
        return values[0]


def test_fault_event_serializes_as_iso8601_json_line():
    start = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
    event = FaultEvent("temp-01", "spike", start, start)

    line = event.to_json_line()

    assert json.loads(line) == {
        "sensor_id": "temp-01",
        "fault_type": "spike",
        "start": start.isoformat(),
        "end": start.isoformat(),
    }
    assert "\n" not in line


def test_spike_is_immediate_and_uses_one_fault_roll():
    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    rng = SequenceRng([0.5])
    simulator = SensorSim("vib-01", "vibration", rng,
                          p_spike=1.0, p_drift=0.0, p_dropout=0.0)

    value, event = simulator.step(now)

    assert value == 10.0
    assert value > NORMAL_RANGES["vibration"][1]
    assert event == FaultEvent("vib-01", "spike", now, now)
    assert rng.random_calls == 1


def test_dropout_suppresses_exactly_ten_ticks_and_emits_on_completion():
    start = datetime(2026, 10, 16, tzinfo=timezone.utc)
    rng = SequenceRng([0.0])
    simulator = SensorSim("vib-01", "vibration", rng,
                          p_spike=0.0, p_drift=0.0, p_dropout=1.0)

    results = [simulator.step(start + timedelta(seconds=tick)) for tick in range(10)]
    readings = [value for value, _ in results]
    assert len(readings) == 10
    assert readings == [None] * 10
    assert all(event is None for _, event in results[:-1])

    end = start + timedelta(seconds=9)
    value, event = results[-1]
    assert value is None
    assert event == FaultEvent("vib-01", "dropout", start, end)


def test_drift_returns_150_readings_and_emits_when_complete():
    start = datetime(2026, 10, 16, tzinfo=timezone.utc)
    rng = SequenceRng([0.5])
    simulator = SensorSim("temp-01", "temperature", rng,
                          p_spike=0.0, p_drift=1.0, p_dropout=0.0)

    first_value, first_event = simulator.step(start)
    assert first_value == 18.0
    assert first_event is None

    previous_value = first_value
    for tick in range(1, 149):
        value, event = simulator.step(start + timedelta(seconds=tick))
        assert value == 18.0 + 0.05 * tick
        assert value > previous_value
        assert event is None
        previous_value = value

    end = start + timedelta(seconds=149)
    last_value, event = simulator.step(end)

    assert last_value == 18.0 + 0.05 * 149
    assert event == FaultEvent("temp-01", "drift", start, end)


def test_same_seed_produces_identical_readings_and_events():
    start = datetime(2026, 10, 16, tzinfo=timezone.utc)

    def run():
        rng = random.Random(17)
        simulator = SensorSim("temp-01", "temperature", rng)
        return [
            simulator.step(start + timedelta(seconds=tick))
            for tick in range(300)
        ]

    assert run() == run()


def test_fault_state_machine_does_not_overlap_faults():
    start = datetime(2026, 10, 16, tzinfo=timezone.utc)
    rng = SequenceRng([0.0, 0.2, 0.8])
    simulator = SensorSim(
        "temp-01",
        "temperature",
        rng,
        p_spike=0.5,
        p_drift=0.5,
        p_dropout=0.1,
    )

    events = []
    for tick in range(161):
        _, event = simulator.step(start + timedelta(seconds=tick))
        if event is not None:
            events.append(event)

    assert [event.fault_type for event in events] == ["dropout", "drift", "spike"]
    assert events[0].end < events[1].start
    assert events[1].end < events[2].start


def test_close_emits_event_for_active_fault():
    start = datetime(2026, 10, 16, tzinfo=timezone.utc)
    end = start + timedelta(seconds=3)
    simulator = SensorSim(
        "vib-01",
        "vibration",
        SequenceRng([0.0]),
        p_spike=0.0,
        p_drift=0.0,
        p_dropout=1.0,
    )

    simulator.step(start)

    assert simulator.close(end) == FaultEvent("vib-01", "dropout", start, end)
    assert simulator.close(end) is None


def test_injector_publishes_readings_and_keeps_labels_separate(tmp_path, monkeypatch):
    from src.faults import injector

    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    event = FaultEvent("temp-01", "spike", now, now)
    published = []
    clients = []

    class FakeClient:
        def connect_async(self, host, port, keepalive):
            assert host == "localhost"
            assert port == 1883
            assert keepalive == 60

        def loop_start(self):
            pass

        def loop_stop(self):
            pass

        def disconnect(self):
            pass

        def publish(self, topic, payload):
            published.append((topic, payload))

    class FakeSensor:
        def __init__(self, sensor_id, sensor_type, rng):
            self.sensor_id = sensor_id
            self.sensor_type = sensor_type

        def step(self, tick):
            assert tick == now
            return 12.5, event if self.sensor_id == "temp-01" else None

        def close(self, tick):
            return None

    monkeypatch.setattr(injector, "SensorSim", FakeSensor)
    monkeypatch.setattr(injector, "make_client", lambda: clients.append(FakeClient()) or clients[-1])
    monkeypatch.setattr(injector.time, "monotonic", iter([0.0, 0.0, 1.0]).__next__)
    monkeypatch.setattr(injector.time, "sleep", lambda interval: None)

    labels_path = tmp_path / "labels" / "faults.jsonl"
    assert injector.main(
        ["--duration", "0.5", "--labels", str(labels_path)],
        clock=lambda: now,
    ) == 0

    assert len(published) == 3
    assert all(topic == "sensors/readings" for topic, _ in published)
    readings = [parse_reading(payload) for _, payload in published]
    assert readings == [
        {
            "sensor_id": sensor_id,
            "type": sensor_type,
            "value": 12.5,
            "timestamp": now,
        }
        for sensor_id, sensor_type in injector.SENSORS
    ]
    assert all("fault_type" not in json.loads(payload) for _, payload in published)

    label_lines = labels_path.read_text(encoding="utf-8").splitlines()
    assert label_lines == [event.to_json_line()]
    assert all(isinstance(json.loads(line), dict) for line in label_lines)