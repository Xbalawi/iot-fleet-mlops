"""Export the models of an MLflow run into a plain folder, ready to ship in a Docker image.

The serving side only needs this module and skops, not MLflow.
"""

import argparse
import json
from pathlib import Path

import skops.io

# Isolation Forests are made of decision trees. skops refuses to load their node
# storage unless we trust it explicitly. We trust only this one type, because we
# train and save these models ourselves.
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

META_FILE = "meta.json"
MODEL_SUFFIX = ".skops"


def export_models(models, out_dir, run_id=None):
    """Write one <sensor_type>.skops file per model, plus meta.json. Returns the folder."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sensor_type, model in models.items():
        skops.io.dump(model, out / f"{sensor_type}{MODEL_SUFFIX}")
    meta = {"run_id": run_id, "models": sorted(models)}
    (out / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    return out


def load_exported(model_dir):
    """Read a folder written by export_models. Returns (models, run_id)."""
    folder = Path(model_dir)
    meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    models = {
        path.stem: skops.io.load(path, trusted=TRUSTED_TYPES)
        for path in sorted(folder.glob(f"*{MODEL_SUFFIX}"))
    }
    return models, meta["run_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the models of an MLflow run to a folder.")
    parser.add_argument("run_id")
    parser.add_argument("--out", default="model_store")
    args = parser.parse_args(argv)

    from src.model.tracking import load_models  # needs MLflow: only this command does

    models = load_models(args.run_id)
    out = export_models(models, args.out, args.run_id)
    print(f"exported {sorted(models)} from run {args.run_id} to {out}")


if __name__ == "__main__":
    main()