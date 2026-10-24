"""Record a training run (settings, scores and trained models) in MLflow."""

import math

import mlflow
import mlflow.sklearn
import sklearn

from src.model.export import TRUSTED_TYPES

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "iot-fleet-anomaly"
MODEL_TAG_PREFIX = "model_uri."

# Isolation Forests are made of decision trees. skops refuses to load their node
# storage unless we trust it explicitly. We trust only this one type, because we
# train and save these models ourselves.
# TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


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


def log_run(params, metrics, models=None, tracking_uri=TRACKING_URI, run_name=None):
    """Store one run in MLflow (settings, scores, models) and return its run id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for sensor_type, model in (models or {}).items():
            info = mlflow.sklearn.log_model(
                model,
                name=f"isolation_forest_{sensor_type}",
                pip_requirements=[f"scikit-learn=={sklearn.__version__}"],
                skops_trusted_types=TRUSTED_TYPES,
            )
            mlflow.set_tag(f"{MODEL_TAG_PREFIX}{sensor_type}", info.model_uri)
    return run.info.run_id


def load_models(run_id, tracking_uri=TRACKING_URI):
    """Load the models saved by log_run. Returns {sensor_type: model}."""
    mlflow.set_tracking_uri(tracking_uri)
    tags = mlflow.MlflowClient().get_run(run_id).data.tags
    return {
        name[len(MODEL_TAG_PREFIX):]: mlflow.sklearn.load_model(uri)
        for name, uri in tags.items()
        if name.startswith(MODEL_TAG_PREFIX)
    }