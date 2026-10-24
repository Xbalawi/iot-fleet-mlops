from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
import numpy as np
from sklearn.ensemble import IsolationForest
from src.model.export import export_models

GOOD = {
    "sensor_id": "temp-01",
    "type": "temperature",
    "value": 24.5,
    "timestamp": "2026-10-17T12:00:00Z",
}


class FakeScorer:
    """Stands in for the real scorer: these tests are about the web layer."""

    models = {"temperature": object()}

    def __init__(self):
        self.seen = []

    def score(self, reading):
        self.seen.append(reading)
        if reading["value"] == 999.0:
            raise ValueError("reading is not newer than the previous one of this sensor")
        return {"status": "ok", "is_anomaly": False, "score": 0.42}


@pytest.fixture
def scorer():
    return FakeScorer()


@pytest.fixture
def client(scorer):
    return TestClient(create_app(scorer, run_id="abc123"))


def test_health_reports_the_loaded_models(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_run_id": "abc123", "models": ["temperature"]}


def test_predict_returns_the_scorers_answer(client):
    response = client.post("/predict", json=GOOD)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "is_anomaly": False, "score": 0.42}


def test_the_scorer_receives_a_clean_reading(client, scorer):
    client.post("/predict", json=GOOD)
    reading = scorer.seen[0]
    assert reading["sensor_id"] == "temp-01"
    assert reading["value"] == 24.5
    assert reading["timestamp"] == datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def test_timestamps_with_an_offset_are_converted_to_utc(client, scorer):
    client.post("/predict", json={**GOOD, "timestamp": "2026-10-17T14:00:00+02:00"})
    assert scorer.seen[0]["timestamp"] == datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def test_a_timestamp_without_a_timezone_is_refused(client, scorer):
    response = client.post("/predict", json={**GOOD, "timestamp": "2026-10-17T12:00:00"})
    assert response.status_code == 422
    assert scorer.seen == []


@pytest.mark.parametrize(
    "payload",
    [
        {k: v for k, v in GOOD.items() if k != "value"},  # missing field
        {**GOOD, "value": "hot"},  # not a number
        {**GOOD, "sensor_id": ""},  # empty id
    ],
)
def test_a_bad_reading_is_refused_before_it_reaches_the_scorer(client, scorer, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert scorer.seen == []


def test_a_reading_that_is_not_newer_gets_a_conflict(client):
    response = client.post("/predict", json={**GOOD, "value": 999.0})
    assert response.status_code == 409


def test_the_service_refuses_to_start_without_a_model_run(monkeypatch):
    monkeypatch.delenv("MODEL_RUN_ID", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)
    with pytest.raises(RuntimeError):
        with TestClient(create_app()):
            pass


def test_the_service_can_load_exported_models_from_a_folder(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    model = IsolationForest(n_estimators=5, random_state=0).fit(rng.normal(size=(50, 2)))
    export_models({"temperature": model}, tmp_path, run_id="abc123")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    with TestClient(create_app()) as client:
        body = client.get("/health").json()

    assert body == {"status": "ok", "model_run_id": "abc123", "models": ["temperature"]}