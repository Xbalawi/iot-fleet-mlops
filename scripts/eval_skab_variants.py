"""Test why the global model fails on SKAB: three variants, graded on the same seconds.

  python -m scripts.eval_skab_variants

global           trained once on anomaly-free.csv, all five features (the previous result)
global, no value trained once on anomaly-free.csv, relative features only
per-file         trained on the first normal seconds of each file itself

Every variant is graded on the rows after each file's calibration window, so
the numbers are directly comparable.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from scripts.eval_skab import COLUMNS, VOTES, features_of, metrics, read
from src.model.features import LONG_WINDOW
from src.model.train import CONTAMINATION, MODEL_FEATURES, N_ESTIMATORS, SEED

CAL_SECONDS = 300  # calibration window for the per-file variant (stops earlier if an anomaly starts sooner)
MIN_TRAIN_ROWS = 50
MIN_CAL_END = 150  # a file whose first anomaly comes sooner is skipped by every variant
RELATIVE_FEATURES = [f for f in MODEL_FEATURES if f != "value"]


def fit(features, feats, first_row=0, last_row=None):
    """One forest per column on the chosen rows. Returns None when a column has too few usable rows."""
    models = {}
    for column in COLUMNS:
        part = features[features["sensor_id"] == column].dropna(subset=feats)
        part = part[part["row"] >= first_row]
        if last_row is not None:
            part = part[part["row"] < last_row]
        if len(part) < MIN_TRAIN_ROWS:
            return None
        models[column] = IsolationForest(
            n_estimators=N_ESTIMATORS, contamination=CONTAMINATION, random_state=SEED
        ).fit(part[feats])
    return models


def flag_table(models, features, n_rows, feats):
    """Booleans: one row per second, one column per sensor column. Warm-up rows are never flagged."""
    table = pd.DataFrame(False, index=range(n_rows), columns=COLUMNS)
    for column, model in models.items():
        part = features[features["sensor_id"] == column].dropna(subset=feats)
        part = part[part["row"] >= LONG_WINDOW]
        table.loc[part["row"].to_numpy(), column] = model.predict(part[feats]) == -1
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/skab")
    args = parser.parse_args(argv)
    root = Path(args.data)

    reference = features_of(read(root / "anomaly-free" / "anomaly-free.csv"))
    global_models = {
        "global": (fit(reference, MODEL_FEATURES), MODEL_FEATURES),
        "global, no value": (fit(reference, RELATIVE_FEATURES), RELATIVE_FEATURES),
    }

    variants = {name: [] for name in [*global_models, "per-file"]}
    first_anomalies, skipped = [], []
    paths = sorted(p for folder in ("valve1", "valve2", "other") for p in (root / folder).glob("*.csv"))
    for path in paths:
        frame = read(path)
        truth_all = frame["anomaly"].astype(bool).to_numpy()
        first_anomaly = int(truth_all.argmax())
        first_anomalies.append(first_anomaly)
        cal_end = min(CAL_SECONDS, first_anomaly)
        if cal_end < MIN_CAL_END:
            skipped.append(path.relative_to(root).as_posix())
            continue

        features = features_of(frame)
        own = fit(features, MODEL_FEATURES, first_row=LONG_WINDOW, last_row=cal_end)
        if own is None:
            skipped.append(path.relative_to(root).as_posix())
            continue

        truth = pd.Series(truth_all[cal_end:])
        tables = {name: flag_table(m, features, len(frame), feats) for name, (m, feats) in global_models.items()}
        tables["per-file"] = flag_table(own, features, len(frame), MODEL_FEATURES)
        for name, table in tables.items():
            variants[name].append((table.iloc[cal_end:].reset_index(drop=True), truth))

    used = len(variants["per-file"])
    print(
        f"first anomaly at second: min {min(first_anomalies)}, median {int(np.median(first_anomalies))}, "
        f"max {max(first_anomalies)}"
    )
    print(f"scored {used} files, skipped {len(skipped)}: {skipped}")
    all_truth = pd.concat([t for _, t in variants["per-file"]], ignore_index=True)
    base_rate = all_truth.mean()
    print(f"graded seconds: {len(all_truth)}, share anomalous (what random alerts would score): {base_rate:.3f}\n")

    rows = []
    for name, parts in variants.items():
        pooled = pd.concat([table for table, _ in parts], ignore_index=True)
        for k in VOTES:
            result = metrics(pooled.sum(axis=1) >= k, all_truth)
            detected, delays, false_seconds = 0, [], []
            for table, truth in parts:
                alert = table.sum(axis=1) >= k
                hits = alert & truth
                if hits.any():
                    detected += 1
                    delays.append(int(hits.to_numpy().argmax() - truth.to_numpy().argmax()))
                false_seconds.append(int((alert & ~truth).sum()))
            rows.append(
                {
                    "variant": name,
                    "alert when": f">= {k} of {len(COLUMNS)}",
                    "precision": result["precision"],
                    "lift": result["precision"] / base_rate,
                    "recall": result["recall"],
                    "false_alarm": result["false_alarm_rate"],
                    "files": f"{detected}/{used}",
                    "delay_s": float(np.median(delays)) if delays else float("nan"),
                    "false_s/file": round(float(np.mean(false_seconds)), 1),
                }
            )
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()