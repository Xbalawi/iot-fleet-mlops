"""HTTP service: score one sensor reading per request."""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.model.scorer import Scorer
# from src.model.tracking import TRACKING_URI


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


""" @asynccontextmanager
async def lifespan(app):
    # Load the models once, when the service starts.
    if app.state.scorer is None:
        run_id = os.environ.get("MODEL_RUN_ID")
        if not run_id:
            raise RuntimeError("Set MODEL_RUN_ID to the id of the MLflow run that holds the models")
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI)
        app.state.scorer = Scorer.from_run(run_id, tracking_uri)
        app.state.run_id = run_id
    yield """

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
    yield


def create_app(scorer=None, run_id=None):
    """Build the service. Pass a scorer to skip loading from MLflow (used by the tests)."""
    app = FastAPI(title="IoT anomaly service", lifespan=lifespan)
    app.state.scorer = scorer
    app.state.run_id = run_id

    @app.get("/health")
    async def health(request: Request):
        state = request.app.state
        return {"status": "ok", "model_run_id": state.run_id, "models": sorted(state.scorer.models)}

    # async on purpose: requests run one at a time, so the scorer's memory is never
    # updated by two requests at once.
    @app.post("/predict", response_model=Prediction)
    async def predict(reading: Reading, request: Request):
        try:
            return request.app.state.scorer.score(reading.model_dump())
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error))

    return app


app = create_app()