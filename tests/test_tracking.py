import mlflow
import pandas as pd

from src.model.tracking import build_metrics, log_run


def results():
    table = pd.DataFrame(
        {"fault_type": ["spike", "drift"], "rows": [2, 2], "caught": [1, 0], "recall": [0.5, 0.0]}
    )
    summary = {"precision": 0.5, "false_alarm_rate": 0.25}
    dropouts = {"events": 2, "caught": 2, "recall": 1.0, "false_alarms": 0}
    return table, summary, dropouts


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
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    run_id = log_run({"seed": 42, "contamination": 0.02}, {"precision": 0.5}, tracking_uri=uri)

    run = mlflow.MlflowClient(tracking_uri=uri).get_run(run_id)
    assert run.data.params["seed"] == "42"  # MLflow stores params as text
    assert run.data.metrics["precision"] == 0.5