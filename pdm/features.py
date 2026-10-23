"""Feature engineering shared by training, the API and the dashboard.

There is one function, ``build_features``. Training runs it over a year of history; the API runs
it over the 24+ hours a client sends and keeps the last row. Same code, same columns, same order,
so the model never sees a feature computed differently from how it was trained (v1 of this
project filled the machine-model columns with 0 at serving time and shifted risk by up to 0.77).

Input frames use the column names of the Azure PdM dataset:

* telemetry: datetime, machineID, volt, rotate, pressure, vibration (one row per hour)
* errors:    datetime, machineID, errorID
* maint:     datetime, machineID, comp        (component replacements, failures included)
* machines:  machineID, model, age
"""

from __future__ import annotations

import pandas as pd

SENSORS = ("volt", "rotate", "pressure", "vibration")
WINDOWS = (3, 24)  # hours
ERRORS = ("error1", "error2", "error3", "error4", "error5")
COMPONENTS = ("comp1", "comp2", "comp3", "comp4")
MODELS = ("model1", "model2", "model3", "model4")  # model1 is the reference level

SENSOR_FEATURES = [f"{s}_{stat}_{w}h" for w in WINDOWS for s in SENSORS for stat in ("mean", "std")]
ERROR_FEATURES = [f"{e}_count_24h" for e in ERRORS]
MAINT_FEATURES = [f"days_since_{c}" for c in COMPONENTS]
MACHINE_FEATURES = ["age"] + [f"model_{m}" for m in MODELS[1:]]
FEATURES = list(SENSORS) + SENSOR_FEATURES + ERROR_FEATURES + MAINT_FEATURES + MACHINE_FEATURES
TELEMETRY_ONLY = list(SENSORS) + SENSOR_FEATURES + MACHINE_FEATURES

HISTORY_HOURS = max(WINDOWS)  # rows needed before the first complete feature row


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values(["machineID", "datetime"]).reset_index(drop=True)


def _rolling(telemetry: pd.DataFrame) -> pd.DataFrame:
    grouped = telemetry.groupby("machineID", sort=False)
    columns = {}
    for w in WINDOWS:
        for sensor in SENSORS:
            roll = grouped[sensor].rolling(w)
            columns[f"{sensor}_mean_{w}h"] = roll.mean().reset_index(level=0, drop=True)
            columns[f"{sensor}_std_{w}h"] = roll.std().reset_index(level=0, drop=True)
    return pd.DataFrame(columns, index=telemetry.index)


def _error_counts(telemetry: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """Errors of each type logged in the 24 hours up to and including each telemetry hour."""
    counts = pd.DataFrame(0.0, index=telemetry.index, columns=ERROR_FEATURES)
    if errors.empty:
        return counts
    events = errors.copy()
    events["datetime"] = pd.to_datetime(events["datetime"]).dt.floor("h")
    hourly = (
        pd.crosstab([events["machineID"], events["datetime"]], events["errorID"])
        .reindex(columns=list(ERRORS), fill_value=0)
        .reset_index()
    )
    merged = telemetry[["machineID", "datetime"]].merge(
        hourly, on=["machineID", "datetime"], how="left"
    )
    merged[list(ERRORS)] = merged[list(ERRORS)].fillna(0.0)
    grouped = merged.groupby("machineID", sort=False)
    for error in ERRORS:
        rolled = grouped[error].rolling(HISTORY_HOURS, min_periods=1).sum()
        counts[f"{error}_count_24h"] = rolled.reset_index(level=0, drop=True)
    return counts


def _days_since_replacement(telemetry: pd.DataFrame, maint: pd.DataFrame) -> pd.DataFrame:
    """Days since each component was last replaced, as of each telemetry hour (NaN if never)."""
    keys = telemetry[["machineID", "datetime"]].assign(_row=telemetry.index)
    keys = keys.sort_values("datetime")
    out = pd.DataFrame(index=telemetry.index, columns=MAINT_FEATURES, dtype=float)
    events = maint.copy()
    events["datetime"] = pd.to_datetime(events["datetime"])
    for comp in COMPONENTS:
        last = events.loc[events["comp"] == comp, ["machineID", "datetime"]]
        last = last.rename(columns={"datetime": "replaced"}).sort_values("replaced")
        if last.empty:
            continue
        joined = pd.merge_asof(keys, last, left_on="datetime", right_on="replaced", by="machineID")
        days = (joined["datetime"] - joined["replaced"]).dt.total_seconds() / 86400
        out.loc[joined["_row"].to_numpy(), f"days_since_{comp}"] = days.to_numpy()
    return out


def _machine_columns(telemetry: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    info = telemetry[["machineID"]].merge(machines, on="machineID", how="left")
    unknown = set(info["model"].dropna()) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown machine model(s): {sorted(unknown)}")
    out = pd.DataFrame({"age": info["age"].astype(float).to_numpy()}, index=telemetry.index)
    for model in MODELS[1:]:  # fixed categories, never pd.get_dummies on whatever is present
        out[f"model_{model}"] = (info["model"] == model).astype(float).to_numpy()
    return out


def build_features(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    """One row per machine-hour: ``machineID``, ``datetime`` and every column in ``FEATURES``.

    Telemetry must be hourly without gaps per machine (rolling windows count rows). The first
    23 hours of each machine, and hours before a component's first known replacement, have
    NaN features; callers drop them.
    """
    tel = _sorted(telemetry)
    parts = [
        tel[["machineID", "datetime", *SENSORS]],
        _rolling(tel),
        _error_counts(tel, errors),
        _days_since_replacement(tel, maint),
        _machine_columns(tel, machines),
    ]
    return pd.concat(parts, axis=1)[["machineID", "datetime", *FEATURES]]


def complete(features: pd.DataFrame) -> pd.DataFrame:
    """Rows where every feature is known."""
    return features.dropna(subset=FEATURES)


def window_features(
    machine_id: int,
    model: str,
    age: float,
    telemetry: list[dict],
    errors: list[dict],
    replacements: list[dict],
) -> pd.DataFrame:
    """Features at the last telemetry hour of one machine, from raw records (API payload).

    ``telemetry`` items: datetime, volt, rotate, pressure, vibration; ``errors``: datetime,
    error_id; ``replacements``: datetime, comp. Returns a one-row frame.
    """
    tel = pd.DataFrame(telemetry).assign(machineID=machine_id)
    err = pd.DataFrame(errors, columns=["datetime", "error_id"]).rename(
        columns={"error_id": "errorID"}
    )
    rep = pd.DataFrame(replacements, columns=["datetime", "comp"])
    machines = pd.DataFrame([{"machineID": machine_id, "model": model, "age": age}])
    rows = build_features(
        tel, err.assign(machineID=machine_id), rep.assign(machineID=machine_id), machines
    )
    return rows.tail(1).reset_index(drop=True)
