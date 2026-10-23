import numpy as np
import pandas as pd
import pytest

from pdm import data
from pdm.features import FEATURES, build_features, complete, window_features
from pdm.labels import label
from pdm.replay import payload


@pytest.fixture(scope="module")
def replay():
    return data.load_replay()


@pytest.fixture(scope="module")
def history(replay):
    return complete(build_features(replay.telemetry, replay.errors, replay.maint, replay.machines))


def test_api_features_equal_training_features(replay, history):
    """Training-serving skew test: a 24 h request gives the same numbers as the full history."""
    rng = np.random.default_rng(0)
    for index in rng.choice(len(history), 25, replace=False):
        row = history.iloc[index]
        request = payload(replay, int(row["machineID"]), row["datetime"])
        served = window_features(
            request["machine_id"],
            request["model"],
            request["age"],
            request["telemetry"],
            request["errors"],
            request["replacements"],
        )
        np.testing.assert_allclose(
            served[FEATURES].to_numpy(dtype=float)[0],
            row[FEATURES].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-6,
            err_msg=f"machine {row['machineID']} at {row['datetime']}",
        )


def test_machine_model_columns_do_not_depend_on_which_models_are_present(replay):
    one = replay.telemetry[replay.telemetry["machineID"] == replay.machines["machineID"].iloc[0]]
    rows = build_features(one, replay.errors, replay.maint, replay.machines)
    model = replay.machines["model"].iloc[0]
    columns = [c for c in FEATURES if c.startswith("model_")]
    expected = {c: float(c == f"model_{model}") for c in columns}
    assert rows[columns].iloc[-1].to_dict() == expected


def test_unknown_machine_model_is_rejected(replay):
    machines = replay.machines.assign(model="model9")
    with pytest.raises(ValueError, match="model9"):
        build_features(replay.telemetry.head(30), replay.errors, replay.maint, machines)


def test_error_count_covers_the_last_24_hours_only():
    hours = pd.date_range("2015-01-01", periods=30, freq="h")
    tel = pd.DataFrame(
        {
            "datetime": hours,
            "machineID": 1,
            "volt": 1.0,
            "rotate": 1.0,
            "pressure": 1.0,
            "vibration": 1.0,
        }
    )
    errors = pd.DataFrame({"datetime": [hours[2]], "machineID": [1], "errorID": ["error3"]})
    maint = pd.DataFrame(
        {"datetime": [hours[0]] * 4, "machineID": 1, "comp": ["comp1", "comp2", "comp3", "comp4"]}
    )
    machines = pd.DataFrame({"machineID": [1], "model": ["model2"], "age": [5]})
    rows = build_features(tel, errors, maint, machines)
    counts = rows["error3_count_24h"].tolist()
    assert counts[1] == 0 and counts[2] == 1 and counts[25] == 1 and counts[26] == 0
    assert rows["days_since_comp1"].iloc[24] == pytest.approx(1.0)


def test_label_marks_the_24_hours_before_a_failure():
    hours = pd.date_range("2015-01-01", periods=60, freq="h")
    frame = pd.DataFrame({"datetime": hours, "machineID": 1})
    failures = pd.DataFrame({"datetime": [hours[40]], "machineID": [1], "failure": ["comp1"]})
    y = label(frame, failures).tolist()
    assert y[15] == 0 and y[16] == 1 and y[39] == 1 and y[40] == 0
    assert sum(y) == 24
