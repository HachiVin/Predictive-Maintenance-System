"""In-process monitoring: latency, alert rate and input drift of recent predictions.

Kept in memory on purpose. One API process or one dashboard session is enough to show the
numbers that matter after a release (p95 latency, how often it alerts, whether inputs still look
like the training data); a real deployment would export the same numbers to Prometheus.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

DRIFT_FEATURES = [f"{s}_mean_24h" for s in ("volt", "rotate", "pressure", "vibration")]
# Hourly readings of one machine are strongly correlated; fewer rows than this swing by chance.
MIN_ROWS_FOR_DRIFT = 500


def psi(edges: list[float], values: np.ndarray) -> float:
    """Population Stability Index of ``values`` against training deciles ``edges``.

    Each training decile holds 10% of the training rows; PSI sums how far the share of new
    values in each decile moved from 10%. Rule of thumb: < 0.1 stable, 0.1-0.25 shifting,
    > 0.25 drifted.
    """
    inner = np.asarray(edges[1:-1], dtype=float)
    counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
    actual = np.clip(counts / max(len(values), 1), 1e-4, None)
    expected = np.full_like(actual, 1 / len(actual))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_level(value: float | None) -> str:
    if value is None:
        return "not enough data"
    if value < 0.1:
        return "stable"
    return "shifting" if value < 0.25 else "drifted"


@dataclass
class Record:
    at: float
    latency_ms: float
    risk: float
    status: str
    inputs: dict[str, float]


class Monitor:
    def __init__(self, reference: dict, size: int = 5000) -> None:
        self.reference = reference
        self._records: deque[Record] = deque(maxlen=size)
        self._lock = threading.Lock()

    def record(self, latency_ms: float, risk: float, status: str, inputs: dict) -> None:
        kept = {k: float(inputs[k]) for k in DRIFT_FEATURES if k in inputs}
        with self._lock:
            self._records.append(Record(time.time(), latency_ms, risk, status, kept))

    def records(self) -> list[Record]:
        with self._lock:
            return list(self._records)

    def summary(self) -> dict:
        records = self.records()
        if not records:
            return {"predictions": 0}
        latency = np.array([r.latency_ms for r in records])
        statuses = [r.status for r in records]
        drift = {}
        for feature in DRIFT_FEATURES:
            values = np.array([r.inputs[feature] for r in records if feature in r.inputs])
            score = None
            if len(values) >= MIN_ROWS_FOR_DRIFT:
                score = round(psi(self.reference["deciles"][feature], values), 3)
            drift[feature] = {"psi": score, "level": drift_level(score)}
        return {
            "predictions": len(records),
            "latency_ms": {
                "p50": round(float(np.percentile(latency, 50)), 2),
                "p95": round(float(np.percentile(latency, 95)), 2),
                "max": round(float(latency.max()), 2),
            },
            "status_share": {s: round(statuses.count(s) / len(statuses), 3) for s in set(statuses)},
            "mean_risk": round(float(np.mean([r.risk for r in records])), 4),
            "drift": drift,
        }
