"""REST API: failure risk of one machine from its last 24+ hours of data.

uvicorn app.main:app --port 8000      # docs at http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.schemas import InspectionRequest, InspectionResult, MachineWindow, Prediction
from pdm.features import windows_features
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


def _window(window: MachineWindow) -> dict:
    return {
        "machine_id": window.machine_id,
        "model": window.model,
        "age": window.age,
        "telemetry": [r.model_dump() for r in window.telemetry],
        "errors": [e.model_dump() for e in window.errors],
        "replacements": [r.model_dump() for r in window.replacements],
    }


def _score(windows: list[MachineWindow], request: Request) -> tuple[list[Prediction], float]:
    predictor = _predictor(request)
    start = time.perf_counter()
    rows = windows_features([_window(w) for w in windows])
    assessments = predictor.assess(rows)
    latency_ms = (time.perf_counter() - start) * 1000
    out = []
    for window, (_i, row), a in zip(windows, rows.iterrows(), assessments, strict=True):
        request.app.state.monitor.record(latency_ms / len(windows), a.risk, a.status, row.to_dict())
        out.append(
            Prediction(
                machine_id=window.machine_id,
                as_of=window.telemetry[-1].datetime,
                risk=round(a.risk, 4),
                status=a.status,
                action=a.action,
                signals=a.signals,
                thresholds=predictor.thresholds,
                model_version=predictor.version,
            )
        )
    log.info(
        json.dumps(
            {
                "event": "prediction",
                "machines": len(windows),
                "alerts": sum(p.status == "ALERT" for p in out),
                "latency_ms": round(latency_ms, 2),
            }
        )
    )
    return out, latency_ms


@app.post("/predict", response_model=Prediction)
def predict(window: MachineWindow, request: Request) -> Prediction:
    """Failure risk of one machine in the next 24 hours."""
    return _score([window], request)[0][0]


@app.post("/inspect", response_model=InspectionResult)
def inspect(body: InspectionRequest, request: Request) -> InspectionResult:
    """A shift inspection: score up to 500 machines in one call and list the ones to check."""
    predictions, latency_ms = _score(body.machines, request)
    order = {"ALERT": 0, "WATCH": 1, "OK": 2}
    predictions.sort(key=lambda p: (order[p.status], -p.risk, p.machine_id))
    alert = sum(p.status == "ALERT" for p in predictions)
    watch = sum(p.status == "WATCH" for p in predictions)
    return InspectionResult(
        predictions=predictions,
        summary={
            "machines": len(predictions),
            "alert": alert,
            "watch": watch,
            "to_inspect": alert + watch,
        },
        latency_ms=round(latency_ms, 2),
    )
