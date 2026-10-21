"""Train one Isolation Forest per sensor type and grade it against the fault labels."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.model.data import label_readings, load_labels, load_readings
from src.model.features import add_features, drop_warm_up
from src.model.gaps import evaluate_dropouts


MODELED_TYPES = ("temperature", "vibration")  # occupancy is binary and never faults
MODEL_FEATURES = ["value", "diff1", "z_short", "z_long", "mean_shift"]
CONTAMINATION = 0.02
SEED = 42


def split_by_run(df):
    """Train on the first run, test on every later run (the past predicts the future)."""
    if df["run_id"].nunique() < 2:
        raise ValueError("need at least two runs: one to train on, one to test on")
    first = df["run_id"].min()
    train = df[df["run_id"] == first].copy()
    test = df[df["run_id"] != first].copy()
    return train, test


def train_models(train, contamination=CONTAMINATION, seed=SEED):
    """Fit one IsolationForest per modeled sensor type. Returns {type: model}."""
    models = {}
    for sensor_type in MODELED_TYPES:
        part = train[train["type"] == sensor_type]
        if part.empty:
            continue
        model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed
        )
        models[sensor_type] = model.fit(part[MODEL_FEATURES])
    return models


def predict(models, df):
    """Return a copy of df with is_flagged (bool) and score (higher = more anomalous)."""
    out = df.copy()
    out["is_flagged"] = False
    out["score"] = np.nan
    for sensor_type, model in models.items():
        mask = out["type"] == sensor_type
        if not mask.any():
            continue
        features = out.loc[mask, MODEL_FEATURES]
        out.loc[mask, "is_flagged"] = model.predict(features) == -1
        out.loc[mask, "score"] = -model.score_samples(features)
    return out


def evaluate(scored):
    """Grade the flags against the labels, on the modeled sensor types only.

    Returns (table, summary):
    - table: one row per fault type with rows, caught and recall.
    - summary: precision (share of flagged rows that are really faulty) and
      false_alarm_rate (share of normal rows that were flagged).
    """
    modeled = scored[scored["type"].isin(MODELED_TYPES)]

    rows = []
    for fault in ("spike", "drift"):
        part = modeled[modeled["fault_type"] == fault]
        caught = int(part["is_flagged"].sum())
        recall = caught / len(part) if len(part) else float("nan")
        rows.append({"fault_type": fault, "rows": len(part), "caught": caught, "recall": recall})

    flagged = modeled[modeled["is_flagged"]]
    normal = modeled[modeled["fault_type"] == "normal"]
    summary = {
        "precision": flagged["is_fault"].mean() if len(flagged) else float("nan"),
        "false_alarm_rate": normal["is_flagged"].mean() if len(normal) else float("nan"),
    }
    return pd.DataFrame(rows), summary


def main():
    labels = load_labels()
    readings = label_readings(load_readings(), labels)
    with_features = add_features(readings)
    featured = drop_warm_up(with_features)
    train, test = split_by_run(featured)
    models = train_models(train)
    table, summary = evaluate(predict(models, test))

    print(f"train rows: {len(train)}   test rows: {len(test)}")
    print(table.round(3).to_string(index=False))
    print(f"precision: {summary['precision']:.3f}   false alarm rate: {summary['false_alarm_rate']:.3f}")

    dropouts = evaluate_dropouts(with_features, labels)
    print(
        f"dropouts: {dropouts['events']}   caught: {dropouts['caught']}"
        f"   recall: {dropouts['recall']:.3f}   false alarms: {dropouts['false_alarms']}"
    )


if __name__ == "__main__":
    main()