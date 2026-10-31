"""Retraining pipeline: check for drift, train a candidate, and promote it only if it earns it."""

import argparse
import math

from src.model.data import label_readings, load_labels, load_readings
from src.model.drift import scan_windows
from src.model.export import export_models, load_exported
from src.model.features import LONG_WINDOW, SHORT_WINDOW, add_features, drop_warm_up
from src.model.retrain import promotion_decision, sensors_to_retrain
from src.model.tracking import log_run
from src.model.train import CONTAMINATION, MODELED_TYPES, evaluate, predict, train_models


def split_recent_validation(df, recent_fraction=0.6):
    """Cut the later readings in time: the first part is 'recent', the rest is held back."""
    ordered = df.sort_values("timestamp")
    index = min(int(len(ordered) * recent_fraction), len(ordered) - 1)
    cutoff = ordered["timestamp"].iloc[index]
    return df[df["timestamp"] < cutoff].copy(), df[df["timestamp"] >= cutoff].copy()


def measure(models, validation):
    """The numbers the promotion rule looks at, on labelled validation data."""
    table, summary = evaluate(predict(models, validation))
    spikes = table.set_index("fault_type").loc["spike"]
    return {
        "recall_spike": float(spikes["recall"]),
        "false_alarm_rate": float(summary["false_alarm_rate"]),
        "validation_spikes": int(spikes["rows"]),
    }


def run_retraining(
    recent, validation, reference, current_models, force=False, train_fn=train_models, measure_fn=measure
):
    """Check for sustained drift and, if there is some, train and judge a candidate.

    Returns a dict: triggered (sensor ids), retrained (bool), current / candidate
    (metrics), decision ({"promote", "reasons"}) and candidate_models. The last four
    are None when no retraining happened.
    """
    sensors = sensors_to_retrain(scan_windows(reference, recent))
    if force:
        modeled = recent[recent["type"].isin(MODELED_TYPES)]
        sensors = sorted(modeled["sensor_id"].unique())
    if not sensors:
        return {
            "triggered": [],
            "retrained": False,
            "current": None,
            "candidate": None,
            "decision": None,
            "candidate_models": None,
        }

    # Only the sensor types that triggered get a new model; the others keep the current one.
    types = set(recent.loc[recent["sensor_id"].isin(sensors), "type"])
    trained = train_fn(recent)
    candidate_models = dict(current_models)
    for sensor_type in types:
        if sensor_type in trained:
            candidate_models[sensor_type] = trained[sensor_type]

    current = measure_fn(current_models, validation)
    candidate = measure_fn(candidate_models, validation)
    return {
        "triggered": sensors,
        "retrained": True,
        "current": current,
        "candidate": candidate,
        "decision": promotion_decision(current, candidate),
        "candidate_models": candidate_models,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="model_store")
    parser.add_argument("--recent-fraction", type=float, default=0.6)
    parser.add_argument("--force", action="store_true", help="retrain even without sustained drift")
    parser.add_argument(
        "--apply", action="store_true", help="if the candidate is promoted, overwrite the model folder"
    )
    args = parser.parse_args(argv)

    labels = load_labels()
    featured = drop_warm_up(add_features(label_readings(load_readings(), labels)))
    first_run = featured["run_id"].min()
    reference = featured[(featured["run_id"] == first_run) & (~featured["is_fault"])]
    recent, validation = split_recent_validation(
        featured[featured["run_id"] != first_run], args.recent_fraction
    )
    current_models, current_run = load_exported(args.model_dir)

    result = run_retraining(recent, validation, reference, current_models, force=args.force)
    if not result["retrained"]:
        print("no sustained drift: nothing to do")
        return

    decision = result["decision"]
    print(f"retraining on: {result['triggered']}" + ("  (forced)" if args.force else ""))
    for name in ("current", "candidate"):
        metrics = result[name]
        print(f"{name:<10}recall_spike {metrics['recall_spike']:.3f}   false_alarm_rate {metrics['false_alarm_rate']:.3f}")
    verdict = "PROMOTED" if decision["promote"] else "REJECTED"
    print(f"decision: {verdict} - {'; '.join(decision['reasons'])}")

    params = {
        "triggered": ",".join(result["triggered"]),
        "forced": args.force,
        "recent_fraction": args.recent_fraction,
        "current_run": current_run,
        "contamination": CONTAMINATION,
        "short_window": SHORT_WINDOW,
        "long_window": LONG_WINDOW,
        "decision": verdict.lower(),
        "reasons": "; ".join(decision["reasons"]),
    }
    metrics = {
        f"{who}_{name}": value
        for who in ("current", "candidate")
        for name, value in result[who].items()
        if not math.isnan(value)
    }
    models = result["candidate_models"] if decision["promote"] else None
    run_id = log_run(params, metrics, models=models, run_name="retraining")
    print(f"logged to MLflow, run id: {run_id}")

    if decision["promote"] and args.apply:
        export_models(result["candidate_models"], args.model_dir, run_id)
        print(f"promoted: {args.model_dir} now holds the models of run {run_id}")


if __name__ == "__main__":
    main()