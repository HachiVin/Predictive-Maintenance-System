"""Request and response models of the REST API.

The client sends what a plant historian actually stores: the last 24+ hourly sensor readings,
the error log and the replacement log of one machine. The server computes the features with
the same code as training (``pdm.features``), so clients never compute rolling statistics.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pdm.features import COMPONENTS, HISTORY_HOURS

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "example_request.json"


def _examples() -> list[dict]:
    try:
        return [json.loads(EXAMPLE.read_text())]
    except (OSError, ValueError):
        return []


class _Timed(BaseModel):
    datetime: datetime

    @field_validator("datetime")
    @classmethod
    def _naive_utc(cls, value: datetime) -> datetime:
        """Times with a zone are converted to UTC; the training data has no zones."""
        return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


class Reading(_Timed):
    volt: float
    rotate: float
    pressure: float
    vibration: float


class ErrorEvent(_Timed):
    error_id: Literal["error1", "error2", "error3", "error4", "error5"]


class Replacement(_Timed):
    comp: Literal["comp1", "comp2", "comp3", "comp4"]


class MachineWindow(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": _examples()})

    machine_id: int = Field(ge=1)
    model: Literal["model1", "model2", "model3", "model4"]
    age: int = Field(ge=0, le=100, description="Machine age in years")
    telemetry: list[Reading] = Field(
        min_length=HISTORY_HOURS,
        max_length=24 * 7,
        description="Hourly readings, oldest first, no gaps; the last one is 'now'",
    )
    errors: list[ErrorEvent] = Field(default_factory=list, description="Errors of the last 24 h")
    replacements: list[Replacement] = Field(
        description="At least the last replacement of every component"
    )

    @model_validator(mode="after")
    def _check(self) -> MachineWindow:
        times = [r.datetime for r in self.telemetry]
        gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
        if any(gap != timedelta(hours=1) for gap in gaps):
            raise ValueError("telemetry must be hourly, oldest first, without gaps")
        known = {r.comp for r in self.replacements if r.datetime <= times[-1]}
        missing = sorted(set(COMPONENTS) - known)
        if missing:
            raise ValueError(f"replacements missing for {missing} (needed for component age)")
        return self


class Prediction(BaseModel):
    machine_id: int
    as_of: datetime
    risk: float = Field(description="Probability of a failure within 24 h")
    status: Literal["ALERT", "WATCH", "OK"]
    action: str
    signals: list[str] = Field(description="Readings that stand out; not causes")
    thresholds: dict[str, float]
    model_version: str
