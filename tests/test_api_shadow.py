import numpy as np
from fastapi.testclient import TestClient
from sklearn.ensemble import IsolationForest

from src.api.app import create_app
from src.model.export import export_models

GOOD = {
    "sensor_id": "temp-01",
    "type": "temperature",
    "value": 24.5,
    "timestamp": "2026-10-17T12:00:00Z",
}
CHAMPION_ANSWER = {"status": "ok", "is_anomaly": False, "score": 0.42}


class Champion:
    models = {"temperature": object()}

    def __init__(self, answer=CHAMPION_ANSWER):
        self.answer = answer

    def score(self, reading):
        if reading["value"] == 999.0:
            raise ValueError("reading is not newer than the previous one of this sensor")
        return self.answer


class Challenger:
    run_id = "cand"

    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.seen = []

    def score(self, reading):
        self.seen.append(reading)
        if self.error:
            raise self.error
        return self.answer


def test_without_a_challenger_shadow_is_off():
    client = TestClient(create_app(Champion(), run_id="abc"))
    assert client.get("/shadow").json() == {"enabled": False}


def test_the_champions_answer_wins_whatever_the_challenger_says():
    challenger = Challenger(answer={"status": "ok", "is_anomaly": True, "score": 0.99})
    client = TestClient(create_app(Champion(), run_id="abc", challenger=challenger))

    assert client.post("/predict", json=GOOD).json() == CHAMPION_ANSWER
    assert client.get("/shadow").json()["only_challenger"] == 1


def test_a_crashing_challenger_does_not_break_predict():
    challenger = Challenger(error=RuntimeError("boom"))
    client = TestClient(create_app(Champion(), run_id="abc", challenger=challenger))

    response = client.post("/predict", json=GOOD)

    assert response.status_code == 200
    assert response.json() == CHAMPION_ANSWER
    assert client.get("/shadow").json()["failures"] == 1


def test_the_shadow_report_counts_the_pairs():
    flagged = {"status": "ok", "is_anomaly": True, "score": 0.9}
    challenger = Challenger(answer=flagged)
    client = TestClient(create_app(Champion(flagged), run_id="abc", challenger=challenger))

    for _ in range(3):
        client.post("/predict", json=GOOD)

    body = client.get("/shadow").json()
    assert body["enabled"] is True
    assert body["paired"] == 3
    assert body["both"] == 3
    assert body["overlap"] == 1.0


def test_a_refused_reading_is_not_shown_to_the_challenger():
    challenger = Challenger(answer=CHAMPION_ANSWER)
    client = TestClient(create_app(Champion(), run_id="abc", challenger=challenger))

    response = client.post("/predict", json={**GOOD, "value": 999.0})

    assert response.status_code == 409
    assert challenger.seen == []


def export_tiny_model(folder, run_id):
    folder.mkdir()
    rng = np.random.default_rng(0)
    model = IsolationForest(n_estimators=5, random_state=0).fit(rng.normal(size=(50, 2)))
    export_models({"temperature": model}, folder, run_id=run_id)


def test_the_challenger_is_loaded_from_a_folder(monkeypatch, tmp_path):
    export_tiny_model(tmp_path / "champion", "champ")
    export_tiny_model(tmp_path / "challenger", "cand")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "champion"))
    monkeypatch.setenv("CHALLENGER_DIR", str(tmp_path / "challenger"))

    with TestClient(create_app()) as client:
        body = client.get("/shadow").json()

    assert body["enabled"] is True
    assert body["challenger_run_id"] == "cand"