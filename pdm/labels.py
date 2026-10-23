"""Training target: will this machine fail within the next ``horizon`` hours?

v1 of this project labelled only the hour of the failure itself (1 positive row per failure),
which teaches a model to recognise a failure that is already happening. A maintenance team
needs the warning before it, so every hour in the ``horizon`` before a failure is positive.
"""

from __future__ import annotations

import pandas as pd

HORIZON_HOURS = 24


def next_failure(features: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """For each machine-hour, the next failure strictly after it: its time and component."""
    keys = features[["machineID", "datetime"]].assign(_row=range(len(features)))
    keys = keys.sort_values("datetime")
    events = failures.rename(columns={"datetime": "failure_time", "failure": "failure_comp"})
    events = events.assign(failure_time=pd.to_datetime(events["failure_time"]))
    events = events.sort_values("failure_time")[["machineID", "failure_time", "failure_comp"]]
    joined = pd.merge_asof(
        keys,
        events,
        left_on="datetime",
        right_on="failure_time",
        by="machineID",
        direction="forward",
        allow_exact_matches=False,
    )
    joined = joined.sort_values("_row")
    return pd.DataFrame(
        {
            "failure_time": joined["failure_time"].to_numpy(),
            "failure_comp": joined["failure_comp"].to_numpy(),
        },
        index=features.index,
    )


def label(features: pd.DataFrame, failures: pd.DataFrame, horizon: int = HORIZON_HOURS):
    """1 when a failure happens within ``horizon`` hours after the row's time, else 0."""
    upcoming = next_failure(features, failures)
    hours = (upcoming["failure_time"] - features["datetime"]).dt.total_seconds() / 3600
    return (hours <= horizon).astype(int)
