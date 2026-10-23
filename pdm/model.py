"""The trained model bundle and what the API and dashboard do with a risk score."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pdm.features import COMPONENTS, ERRORS, FEATURES, SENSORS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pdm_rf.joblib"
CARD_PATH = ROOT / "models" / "model_card.json"

ALERT, WATCH, OK = "ALERT", "WATCH", "OK"
ACTIONS = {
    ALERT: "Failure likely within 24 h. Inspect before the next shift.",
    WATCH: "Risk is rising. Check at the next planned stop.",
    OK: "No action needed.",
}


class ModelUnavailable(RuntimeError):
    """The model file is missing or cannot be loaded."""


@dataclass
class Assessment:
    risk: float
    status: str
    action: str
    signals: list[str] = field(default_factory=list)


@dataclass
class Predictor:
    model: object
    features: list[str]
    thresholds: dict[str, float]  # {"alert": .., "watch": ..}
    reference: dict  # training-set statistics, see scripts/train.py
    version: str
    importances: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> Predictor:
        path = Path(path)
        if not path.exists():
            raise ModelUnavailable(f"Model file not found: {path}. Run scripts/train.py.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bundle = joblib.load(path)
        mismatch = [w for w in caught if "InconsistentVersionWarning" in type(w.message).__name__]
        if mismatch:
            raise ModelUnavailable(
                f"{path.name} was saved with another scikit-learn version "
                f"({mismatch[0].message}). Install the pinned requirements or retrain."
            )
        if bundle["features"] != FEATURES:
            raise ModelUnavailable("Model features differ from pdm.features.FEATURES. Retrain.")
        return cls(
            model=bundle["model"],
            features=bundle["features"],
            thresholds=bundle["thresholds"],
            reference=bundle["reference"],
            version=bundle["version"],
            importances=bundle.get("importances", {}),
        )

    def risk(self, rows: pd.DataFrame) -> np.ndarray:
        """Probability of failure within 24 h for each row (rows must hold every feature)."""
        missing = [f for f in self.features if f not in rows.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        values = rows[self.features]
        if values.isna().any().any():
            raise ValueError("Features contain NaN; send at least 24 hours of telemetry.")
        return self.model.predict_proba(values)[:, 1]

    def status(self, risk: float) -> str:
        if risk >= self.thresholds["alert"]:
            return ALERT
        if risk >= self.thresholds["watch"]:
            return WATCH
        return OK

    def assess(self, rows: pd.DataFrame) -> list[Assessment]:
        out = []
        for risk, (_i, row) in zip(self.risk(rows), rows.iterrows(), strict=True):
            status = self.status(float(risk))
            out.append(Assessment(float(risk), status, ACTIONS[status], self.signals(row)))
        return out

    def signals(self, row: pd.Series, limit: int = 3) -> list[str]:
        """Readings that stand out against the training data, most important feature first.

        These are not causes: a Random Forest has no single reason for a score. They point a
        technician at what to look at, ordered by how much the model relies on each feature.
        """
        q = self.reference["quantiles"]
        found: list[tuple[float, str]] = []
        for error in ERRORS:
            n = int(row[f"{error}_count_24h"])
            if n:
                key = f"{error}_count_24h"
                found.append((self.importances.get(key, 0), f"{error} ×{n} in 24 h"))
        for comp in COMPONENTS:
            key = f"days_since_{comp}"
            if row[key] > q[key]["p90"]:
                found.append(
                    (self.importances.get(key, 0), f"{comp} {row[key]:.0f} days since replaced")
                )
        for sensor in SENSORS:
            key = f"{sensor}_mean_24h"
            if row[key] > q[key]["p99"]:
                found.append((self.importances.get(key, 0), f"{sensor} 24 h mean high"))
            elif row[key] < q[key]["p01"]:
                found.append((self.importances.get(key, 0), f"{sensor} 24 h mean low"))
        found.sort(key=lambda item: -item[0])
        return [text for _w, text in found[:limit]]
