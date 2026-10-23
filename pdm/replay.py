"""Turn stored history into API requests (used by the sensor client, tests and docs)."""

from __future__ import annotations

import pandas as pd

from pdm.data import Tables
from pdm.features import HISTORY_HOURS


def _iso(value) -> str:
    return pd.Timestamp(value).isoformat()


def payload(tables: Tables, machine_id: int, as_of, hours: int = HISTORY_HOURS) -> dict:
    """The /predict request for ``machine_id`` at hour ``as_of``: the last ``hours`` readings,
    the errors in that window and the last replacement of each component."""
    as_of = pd.Timestamp(as_of)
    start = as_of - pd.Timedelta(hours=hours - 1)
    tel = tables.telemetry
    tel = tel[(tel["machineID"] == machine_id) & tel["datetime"].between(start, as_of)]
    if len(tel) != hours:
        raise ValueError(f"machine {machine_id} has {len(tel)} of {hours} hours before {as_of}")
    err = tables.errors
    err = err[(err["machineID"] == machine_id) & err["datetime"].between(start, as_of)]
    rep = tables.maint
    rep = rep[(rep["machineID"] == machine_id) & (rep["datetime"] <= as_of)]
    rep = rep.sort_values("datetime").groupby("comp").tail(1).sort_values("datetime")
    machine = tables.machines.set_index("machineID").loc[machine_id]
    return {
        "machine_id": int(machine_id),
        "model": str(machine["model"]),
        "age": int(machine["age"]),
        "telemetry": [
            {
                "datetime": _iso(r.datetime),
                "volt": round(r.volt, 3),
                "rotate": round(r.rotate, 3),
                "pressure": round(r.pressure, 3),
                "vibration": round(r.vibration, 3),
            }
            for r in tel.sort_values("datetime").itertuples()
        ],
        "errors": [{"datetime": _iso(r.datetime), "error_id": r.errorID} for r in err.itertuples()],
        "replacements": [{"datetime": _iso(r.datetime), "comp": r.comp} for r in rep.itertuples()],
    }
