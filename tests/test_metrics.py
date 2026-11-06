from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.metrics import Histogram, install_metrics


def make():
    """A tiny stand-in app, so these tests don't depend on the real scorer."""
    app = FastAPI()

    @app.post("/predict")
    async def predict(body: dict):
        if body.get("conflict"):
            raise HTTPException(status_code=409, detail="stale")
        return {
            "status": body.get("status", "ok"),
            "is_anomaly": body.get("anomaly", False),
            "score": 0.5,
        }

    metrics = install_metrics(app)
    return TestClient(app), metrics


def test_histogram_is_cumulative():
    h = Histogram((1, 2, 5))
    for v in (0.5, 1.5, 1.5, 10):
        h.observe(v)
    lines = h.render("x", "help")
    assert 'x_bucket{le="1"} 1' in lines
    assert 'x_bucket{le="2"} 3' in lines
    assert 'x_bucket{le="5"} 3' in lines
    assert 'x_bucket{le="+Inf"} 4' in lines
    assert "x_sum 13.500000" in lines
    assert "x_count 4" in lines


def test_counts_predictions_and_anomalies():
    client, _ = make()
    client.post("/predict", json={"status": "ok"})
    client.post("/predict", json={"status": "warming_up"})
    client.post("/predict", json={"status": "ok", "anomaly": True})
    lines = client.get("/metrics").text.splitlines()
    assert 'predict_requests_total{code="200"} 3' in lines
    assert 'predict_results_total{status="ok"} 2' in lines
    assert 'predict_results_total{status="warming_up"} 1' in lines
    assert "predict_anomalies_total 1" in lines
    assert "predict_latency_seconds_count 3" in lines


def test_conflict_is_counted_but_not_a_prediction():
    client, metrics = make()
    client.post("/predict", json={"conflict": True})
    text = client.get("/metrics").text
    assert 'predict_requests_total{code="409"} 1' in text.splitlines()
    assert "predict_results_total{" not in text
    assert metrics.age.count == 0


def test_reading_age_is_measured():
    client, metrics = make()
    sent = datetime.now(timezone.utc) - timedelta(seconds=10)
    client.post("/predict", json={"timestamp": sent.isoformat()})
    assert metrics.age.count == 1
    assert 10 <= metrics.age.sum < 20


def test_unusable_timestamps_are_ignored():
    client, metrics = make()
    for body in (
        {"timestamp": "not a date"},
        {"timestamp": "2026-01-01T00:00:00"},  # no timezone
        {"timestamp": 123},
        {},
    ):
        assert client.post("/predict", json=body).status_code == 200
    assert metrics.age.count == 0
    assert metrics.latency.count == 4


def test_only_predict_is_measured():
    client, metrics = make()
    client.get("/metrics")
    client.get("/metrics")
    assert metrics.latency.count == 0
    assert sum(metrics.requests.values()) == 0


def test_metrics_endpoint_is_plain_text():
    client, _ = make()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")