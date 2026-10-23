"""REST API: failure risk of one machine from its last 24+ hours of data.

uvicorn app.main:app --port 8000      # docs at http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.schemas import MachineWindow, Prediction
from pdm.features import window_features
from pdm.model import CARD_PATH, ModelUnavailable, Predictor
from pdm.monitoring import Monitor

log = logging.getLogger("pdm.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor, app.state.load_error = None, None
    try:
        app.state.predictor = Predictor.load()
        app.state.monitor = Monitor(app.state.predictor.reference)
    except ModelUnavailable as exc:
        app.state.load_error = str(exc)
        app.state.monitor = Monitor({})
        log.error(json.dumps({"event": "model_unavailable", "detail": str(exc)}))
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description="Probability that a machine fails within the next 24 hours.",
    version="2.0.0",
    lifespan=lifespan,
)


def _predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail=request.app.state.load_error)
    return predictor


@app.get("/health")
def health(request: Request) -> dict:
    predictor = request.app.state.predictor
    return {
        "status": "ok" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "model_version": predictor.version if predictor else None,
    }


@app.get("/model-info")
def model_info(request: Request) -> dict:
    _predictor(request)
    return json.loads(CARD_PATH.read_text())


@app.get("/metrics")
def metrics(request: Request) -> dict:
    """Latency, status mix and input drift of the predictions served by this process."""
    return request.app.state.monitor.summary()


@app.post("/predict", response_model=Prediction)
def predict(window: MachineWindow, request: Request) -> Prediction:
    predictor = _predictor(request)
    start = time.perf_counter()
    row = window_features(
        window.machine_id,
        window.model,
        window.age,
        [r.model_dump() for r in window.telemetry],
        [e.model_dump() for e in window.errors],
        [r.model_dump() for r in window.replacements],
    )
    assessment = predictor.assess(row)[0]
    latency_ms = (time.perf_counter() - start) * 1000
    request.app.state.monitor.record(
        latency_ms, assessment.risk, assessment.status, row.iloc[0].to_dict()
    )
    log.info(
        json.dumps(
            {
                "event": "prediction",
                "machine_id": window.machine_id,
                "risk": round(assessment.risk, 4),
                "status": assessment.status,
                "latency_ms": round(latency_ms, 2),
            }
        )
    )
    return Prediction(
        machine_id=window.machine_id,
        as_of=window.telemetry[-1].datetime,
        risk=round(assessment.risk, 4),
        status=assessment.status,
        action=assessment.action,
        signals=assessment.signals,
        thresholds=predictor.thresholds,
        model_version=predictor.version,
    )
