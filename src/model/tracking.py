"""Record a training run (settings and scores) in MLflow."""

import math

import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "iot-fleet-anomaly"


def build_metrics(table, summary, dropouts):
    """Flatten the evaluation results into {name: float}. NaN values are left out."""
    values = {
        f"recall_{fault}": recall
        for fault, recall in table.set_index("fault_type")["recall"].items()
    }
    values["recall_dropout"] = dropouts["recall"]
    values["precision"] = summary["precision"]
    values["false_alarm_rate"] = summary["false_alarm_rate"]
    values["dropout_false_alarms"] = dropouts["false_alarms"]
    return {name: float(v) for name, v in values.items() if not math.isnan(float(v))}


def log_run(params, metrics, tracking_uri=TRACKING_URI, run_name=None):
    """Store one run in MLflow and return its run id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    return run.info.run_id