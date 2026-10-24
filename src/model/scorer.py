"""Score one reading at a time, the way the service will.

The model needs rolling-window features, which need the recent past of each
sensor. So the scorer keeps the last `history` readings of every sensor in
memory and builds the features with the SAME add_features code used in
training, so training and serving cannot drift apart.
"""

from collections import defaultdict, deque

import pandas as pd

from src.model.export import load_exported
from src.model.features import LONG_WINDOW, MODEL_FEATURES, RUN_GAP, add_features

# from src.model.features import LONG_WINDOW, RUN_GAP, add_features
# from src.model.train import MODEL_FEATURES
# from src.model.tracking import TRACKING_URI, load_models


def _answer(status, is_anomaly=None, score=None):
    return {"status": status, "is_anomaly": is_anomaly, "score": score}


class Scorer:
    # OLD version of __init__ and from_run commented out, replaced with new version that includes run_id
    """ def __init__(self, models, history=LONG_WINDOW):
        self.models = models
        self.history = history
        self._buffers = defaultdict(lambda: deque(maxlen=history))
        self._last_seen = None  # newest timestamp of ANY sensor

    @classmethod
    def from_run(cls, run_id, tracking_uri=TRACKING_URI):
        # Build a scorer from the models saved in an MLflow run.
        return cls(load_models(run_id, tracking_uri)) """
    
    def __init__(self, models, history=LONG_WINDOW, run_id=None):
        self.models = models
        self.history = history
        self.run_id = run_id
        self._buffers = defaultdict(lambda: deque(maxlen=history))
        self._last_seen = None  # newest timestamp of ANY sensor

    @classmethod
    def from_run(cls, run_id, tracking_uri=None):
        """Build a scorer from the models saved in an MLflow run."""
        # imported here so that serving from a folder never needs MLflow
        from src.model.tracking import TRACKING_URI, load_models

        return cls(load_models(run_id, tracking_uri or TRACKING_URI), run_id=run_id)

    @classmethod
    def from_dir(cls, model_dir):
        """Build a scorer from a folder written by export_models."""
        models, run_id = load_exported(model_dir)
        return cls(models, run_id=run_id)
    def score(self, reading):
        """Score one reading: {"sensor_id", "type", "value", "timestamp"}.

        Returns {"status", "is_anomaly", "score"}. The status is one of
        ok, warming_up, not_modeled, no_features.
        Raises ValueError if the reading is not newer than the sensor's previous one.
        """
        now = reading["timestamp"]

        # A long pause in the whole feed ends the run (same rule as training).
        if self._last_seen is not None and now - self._last_seen > RUN_GAP:
            self._buffers.clear()
        if self._last_seen is None or now > self._last_seen:
            self._last_seen = now

        model = self.models.get(reading["type"])
        if model is None:
            return _answer("not_modeled")

        buffer = self._buffers[reading["sensor_id"]]
        if buffer and now <= buffer[-1]["timestamp"]:
            raise ValueError("reading is not newer than the previous one of this sensor")

        past = list(buffer)
        buffer.append(reading)
        if len(past) < self.history:
            return _answer("warming_up")

        frame = pd.DataFrame(past + [reading])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["run_id"] = 0  # the runs are handled above, not by add_features
        features = add_features(frame).iloc[[-1]][MODEL_FEATURES]
        if features.isna().to_numpy().any():
            return _answer("no_features")

        flagged = model.predict(features)[0] == -1
        score = -model.score_samples(features)[0]
        return _answer("ok", bool(flagged), float(score))