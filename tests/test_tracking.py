import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.model.tracking import build_metrics, load_models, log_run


def results():
    table = pd.DataFrame(
        {"fault_type": ["spike", "drift"], "rows": [2, 2], "caught": [1, 0], "recall": [0.5, 0.0]}
    )
    summary = {"precision": 0.5, "false_alarm_rate": 0.25}
    dropouts = {"events": 2, "caught": 2, "recall": 1.0, "false_alarms": 0}
    return table, summary, dropouts


def sqlite_uri(tmp_path):
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_build_metrics_flattens_every_result():
    assert build_metrics(*results()) == {
        "recall_spike": 0.5,
        "recall_drift": 0.0,
        "recall_dropout": 1.0,
        "precision": 0.5,
        "false_alarm_rate": 0.25,
        "dropout_false_alarms": 0.0,
    }


def test_build_metrics_leaves_out_nan():
    table, summary, dropouts = results()
    summary["precision"] = float("nan")
    assert "precision" not in build_metrics(table, summary, dropouts)


def test_log_run_stores_params_and_metrics(tmp_path):
    uri = sqlite_uri(tmp_path)

    run_id = log_run({"seed": 42, "contamination": 0.02}, {"precision": 0.5}, tracking_uri=uri)

    run = mlflow.MlflowClient(tracking_uri=uri).get_run(run_id)
    assert run.data.params["seed"] == "42"  # MLflow stores params as text
    assert run.data.metrics["precision"] == 0.5


def test_logged_models_can_be_loaded_back(tmp_path):
    uri = sqlite_uri(tmp_path)
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    model = IsolationForest(n_estimators=20, random_state=0).fit(data)

    run_id = log_run({}, {"m": 1.0}, models={"temperature": model}, tracking_uri=uri)
    loaded = load_models(run_id, tracking_uri=uri)

    assert set(loaded) == {"temperature"}
    assert (loaded["temperature"].predict(data) == model.predict(data)).all()