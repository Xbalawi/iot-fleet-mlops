from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class FaultEvent:
    sensor_id: str
    fault_type: str      # "spike" | "drift" | "dropout"
    start: datetime
    end: datetime

    def to_json_line(self) -> str:
        return json.dumps({
            "sensor_id": self.sensor_id,
            "fault_type": self.fault_type,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        })


def to_json_line(fault_event: FaultEvent) -> str:
    return fault_event.to_json_line()


NORMAL_RANGES = {"temperature": (18.0, 30.0), "vibration": (0.0, 5.0)}  # from v1 simulator.py


def baseline(sensor_type, rng):
    if sensor_type == "occupancy":
        return float(rng.choice([0, 1]))
    low, high = NORMAL_RANGES[sensor_type]
    return round(rng.uniform(low, high), 2)


# Spike values per sensor type. Every range lies OUTSIDE the normal range.
# Occupancy is absent on purpose: a binary sensor cannot spike.
SPIKE_RANGES = {
    "temperature": [(31.0, 39.0), (11.0, 17.0), (41.0, 50.0)],  # hard high, hard low, easy
    "vibration": [(6.0, 10.0)],
}


def spike_value(sensor_type, rng):
    """Return one spike reading for this sensor type."""
    low, high = rng.choice(SPIKE_RANGES[sensor_type])
    return round(rng.uniform(low, high), 2)


class SensorSim:
    def __init__(self, sensor_id, sensor_type, rng, p_spike=0.02, p_drift=0.002, p_dropout=0.002):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.rng = rng
        self.p_spike = p_spike
        self.p_drift = p_drift
        self.p_dropout = p_dropout
        self._state = "normal"
        self._fault_start = None
        self._dropout_ticks_remaining = 0
        self._drift_readings = 0
        # self._drift_baseline = None

    def step(self, now):
        """Return the reading and any fault event completed on this tick."""
        if self._state == "dropped":
            self._dropout_ticks_remaining -= 1
            if self._dropout_ticks_remaining == 0:
                event = FaultEvent(self.sensor_id, "dropout", self._fault_start, now)
                self._reset()
                return None, event
            return None, None

        if self._state == "drifting":
            value = round(baseline(self.sensor_type, self.rng) + 0.05 * self._drift_readings, 2)            
            self._drift_readings += 1
            if self._drift_readings == 150:
                event = FaultEvent(self.sensor_id, "drift", self._fault_start, now)
                self._reset()
                return value, event
            return value, None

        roll = self.rng.random()
        if roll < self.p_dropout:
            self._state = "dropped"
            self._fault_start = now
            self._dropout_ticks_remaining = 9
            return None, None

        if (self.sensor_type == "temperature"
                and roll < self.p_dropout + self.p_drift):
            self._state = "drifting"
            self._fault_start = now
            self._drift_readings = 1
            # self._drift_baseline = baseline(self.sensor_type, self.rng)
            return baseline(self.sensor_type, self.rng), None
        
        if (self.sensor_type in SPIKE_RANGES
                and roll < self.p_dropout + self.p_drift + self.p_spike):
                event = FaultEvent(self.sensor_id, "spike", now, now)
                return spike_value(self.sensor_type, self.rng), event

        return baseline(self.sensor_type, self.rng), None

    def close(self, now):
        """Return and clear the event for an episode still in progress."""
        if self._state == "dropped":
            event = FaultEvent(self.sensor_id, "dropout", self._fault_start, now)
            self._reset()
            return event
        if self._state == "drifting":
            event = FaultEvent(self.sensor_id, "drift", self._fault_start, now)
            self._reset()
            return event
        return None

    def _reset(self):
        self._state = "normal"
        self._fault_start = None
        self._dropout_ticks_remaining = 0
        self._drift_readings = 0
        # self._drift_baseline = None