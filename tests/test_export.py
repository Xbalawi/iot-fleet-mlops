import numpy as np
from sklearn.ensemble import IsolationForest

from src.model.export import export_models, load_exported


def test_exported_models_load_back_with_the_same_predictions(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 3))
    models = {"temperature": IsolationForest(n_estimators=20, random_state=0).fit(data)}

    export_models(models, tmp_path / "store", run_id="abc123")
    loaded, run_id = load_exported(tmp_path / "store")

    assert run_id == "abc123"
    assert set(loaded) == {"temperature"}
    assert (loaded["temperature"].predict(data) == models["temperature"].predict(data)).all()