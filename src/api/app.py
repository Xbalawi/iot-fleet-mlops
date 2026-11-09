"""HTTP service: score one sensor reading per request."""

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.api.metrics import install_metrics
from src.model.scorer import Scorer
from src.shadow.runner import ShadowRunner


class Reading(BaseModel):
    """One sensor reading, as sent by the simulator."""

    sensor_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    value: float = Field(strict=True, allow_inf_nan=False)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def must_have_a_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError("timestamp must include a timezone, e.g. 2026-10-17T12:00:00Z")
        return value.astimezone(timezone.utc)


class Prediction(BaseModel):
    status: str  # ok | warming_up | not_modeled | no_features
    is_anomaly: bool | None
    score: float | None


@asynccontextmanager
async def lifespan(app):
    """Load the models once, when the service starts."""
    if app.state.scorer is None:
        model_dir = os.environ.get("MODEL_DIR")
        run_id = os.environ.get("MODEL_RUN_ID")
        if model_dir:
            app.state.scorer = Scorer.from_dir(model_dir)
        elif run_id:
            app.state.scorer = Scorer.from_run(run_id, os.environ.get("MLFLOW_TRACKING_URI"))
        else:
            raise RuntimeError("Set MODEL_DIR (exported models) or MODEL_RUN_ID (an MLflow run)")
        app.state.run_id = app.state.scorer.run_id

    # Optional: a challenger, exported like the champion, runs in shadow mode.
    challenger_dir = os.environ.get("CHALLENGER_DIR")
    if app.state.shadow is None and challenger_dir:
        app.state.shadow = ShadowRunner(Scorer.from_dir(challenger_dir))
    yield


def create_app(scorer=None, run_id=None, challenger=None):
    """Build the service. Pass a scorer (and a challenger) to skip loading models (used by the tests)."""
    app = FastAPI(title="IoT anomaly service", lifespan=lifespan)
    app.state.scorer = scorer
    app.state.run_id = run_id
    app.state.shadow = ShadowRunner(challenger) if challenger is not None else None

    @app.get("/health")
    async def health(request: Request):
        state = request.app.state
        return {"status": "ok", "model_run_id": state.run_id, "models": sorted(state.scorer.models)}

    @app.get("/shadow")
    async def shadow(request: Request):
        runner = request.app.state.shadow
        if runner is None:
            return {"enabled": False}
        return {"enabled": True, **runner.report()}

    # async on purpose: requests run one at a time, so the scorer's memory is never
    # updated by two requests at once.
    @app.post("/predict", response_model=Prediction)
    async def predict(reading: Reading, request: Request):
        state = request.app.state
        data = reading.model_dump()
        started = time.perf_counter()
        try:
            answer = state.scorer.score(data)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error))
        if state.shadow is not None:
            # after the champion has answered: the challenger can never change the answer
            state.shadow.observe(data, answer, time.perf_counter() - started)
        return answer

    install_metrics(app)
    return app


app = create_app()