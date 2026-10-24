from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.model.features import add_features, drop_warm_up
from src.model.scorer import Scorer
from src.model.tracking import log_run
from src.model.train import predict, train_models
from src.model.export import export_models

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def reading(sensor_id, seconds, value, sensor_type="temperature"):
    return {
        "sensor_id": sensor_id,
        "type": sensor_type,
        "value": float(value),
        "timestamp": T0 + timedelta(seconds=seconds),
    }


def make_frame(n=120, spike_at=None):
    """One temperature sensor, one reading per second, calm values."""
    rng = np.random.default_rng(0)
    values = 24 + rng.normal(0, 1, size=n)
    if spike_at is not None:
        values[spike_at] = 60.0
    return pd.DataFrame(
        {
            "sensor_id": "t1",
            "type": "temperature",
            "value": values,
            "timestamp": pd.to_datetime([T0 + timedelta(seconds=i) for i in range(n)], utc=True),
        }
    )


def fitted_models():
    return train_models(drop_warm_up(add_features(make_frame())))


def calm(i):
    return 24 + (i % 3)


def test_the_first_50_readings_warm_up_then_scores_appear():
    scorer = Scorer(fitted_models())
    statuses = [scorer.score(reading("t1", i, calm(i)))["status"] for i in range(52)]
    assert statuses[:50] == ["warming_up"] * 50
    assert statuses[50] == "ok"


def test_a_sensor_type_without_a_model_is_not_modeled():
    answer = Scorer(fitted_models()).score(reading("o1", 0, 1, "occupancy"))
    assert answer == {"status": "not_modeled", "is_anomaly": None, "score": None}


def test_a_long_pause_in_the_whole_feed_starts_a_new_run():
    scorer = Scorer(fitted_models())
    for i in range(60):
        scorer.score(reading("t1", i, calm(i)))
    assert scorer.score(reading("t1", 160, 25))["status"] == "warming_up"


def test_one_silent_sensor_does_not_reset_while_the_others_keep_sending():
    scorer = Scorer(fitted_models())
    for i in range(60):
        scorer.score(reading("a", i, calm(i)))
        scorer.score(reading("b", i, calm(i)))
    for i in range(60, 100):  # "a" is silent for 40 s, "b" keeps sending
        scorer.score(reading("b", i, calm(i)))
    assert scorer.score(reading("a", 100, 25))["status"] == "ok"


def test_every_sensor_has_its_own_history():
    scorer = Scorer(fitted_models())
    for i in range(50):
        scorer.score(reading("a", i, calm(i)))
    assert scorer.score(reading("b", 50, 25))["status"] == "warming_up"
    assert scorer.score(reading("a", 51, 25))["status"] == "ok"


def test_a_reading_that_is_not_newer_is_refused():
    scorer = Scorer(fitted_models())
    scorer.score(reading("t1", 5, 24))
    with pytest.raises(ValueError):
        scorer.score(reading("t1", 4, 24))
    with pytest.raises(ValueError):
        scorer.score(reading("t1", 5, 24))


def test_online_scores_match_the_offline_pipeline():
    df = make_frame(n=120, spike_at=100)
    featured = drop_warm_up(add_features(df))
    models = train_models(featured)
    offline = predict(models, featured)

    scorer = Scorer(models)
    online = []
    for row in df.itertuples():
        answer = scorer.score(
            {
                "sensor_id": row.sensor_id,
                "type": row.type,
                "value": row.value,
                "timestamp": row.timestamp,
            }
        )
        if answer["status"] == "ok":
            online.append(answer)

    assert len(online) == len(offline) == 70
    assert np.allclose([a["score"] for a in online], offline["score"].to_numpy(), atol=1e-6)
    assert [a["is_anomaly"] for a in online] == offline["is_flagged"].tolist()
    assert online[50]["is_anomaly"] is True  # the spike at reading 100


def test_from_run_loads_the_models_saved_in_mlflow(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_id = log_run({}, {"m": 1.0}, models=fitted_models(), tracking_uri=uri)

    scorer = Scorer.from_run(run_id, tracking_uri=uri)

    assert set(scorer.models) == {"temperature"}


def test_from_dir_scores_like_the_original_models(tmp_path):
    models = fitted_models()
    export_models(models, tmp_path, run_id="abc123")
    from_dir = Scorer.from_dir(tmp_path)
    original = Scorer(models)

    for i in range(51):
        expected = original.score(reading("t1", i, calm(i)))
        got = from_dir.score(reading("t1", i, calm(i)))

    assert got["status"] == "ok"
    assert got["score"] == pytest.approx(expected["score"])
    assert from_dir.run_id == "abc123"