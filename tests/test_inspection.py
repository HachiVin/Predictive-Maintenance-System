import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from pdm import data
from pdm.features import build_features, complete
from pdm.inspection import breakdown_within, estimate_savings, inspect
from pdm.model import ALERT, Predictor
from pdm.replay import payload


@pytest.fixture(scope="module")
def replay():
    return data.load_replay()


@pytest.fixture(scope="module")
def history(replay):
    return complete(build_features(replay.telemetry, replay.errors, replay.maint, replay.machines))


def _before_first_failure(replay, hours=10):
    later = replay.failures[replay.failures["datetime"] >= "2015-09-05"]
    first = later.sort_values("datetime").iloc[0]
    return int(first["machineID"]), first["datetime"] - pd.Timedelta(hours=hours)


def test_inspection_covers_every_machine_and_flags_the_one_about_to_fail(replay, history):
    machine, at = _before_first_failure(replay)
    rows = history[history["datetime"] == at]
    result = inspect(Predictor.load(), rows, at, replay.failures)
    assert len(result.findings) == replay.machines["machineID"].nunique() == 100
    finding = next(f for f in result.findings if f.machine_id == machine)
    assert finding.status == ALERT and finding.breakdown.endswith("after 10 h")
    summary = result.summary()
    assert summary["caught"] >= 1 and summary["to_inspect"] == summary["alert"] + summary["watch"]
    report = result.report()
    assert report["Status"].iloc[0] == "Alert" and len(report) == 100


def test_breakdown_log_window_is_the_next_24_hours(replay):
    machine, at = _before_first_failure(replay, hours=10)
    assert breakdown_within(replay.failures, machine, at)
    assert breakdown_within(replay.failures, machine, at - pd.Timedelta(hours=20)) == ""


def test_savings_need_both_costs_from_the_user():
    summary = {"caught": 2, "to_inspect": 3}
    assert estimate_savings(summary, None, 100) is None
    assert estimate_savings(summary, 1000, 100) == {"avoided": 2000, "inspection": 300, "net": 1700}


def test_inspect_endpoint_scores_many_machines_in_one_call(replay):
    machine, at = _before_first_failure(replay)
    ids = [machine] + [m for m in sorted(replay.machines["machineID"]) if m != machine][:19]
    body = {"machines": [payload(replay, m, at) for m in ids]}
    with TestClient(main.app) as client:
        result = client.post("/inspect", json=body).json()
        assert result["summary"]["machines"] == 20
        assert result["predictions"][0]["status"] == "ALERT"
        assert result["predictions"][0]["machine_id"] == machine
        single = client.post("/predict", json=payload(replay, machine, at)).json()
        assert single["risk"] == result["predictions"][0]["risk"]
        body["machines"].append(body["machines"][0])
        assert client.post("/inspect", json=body).status_code == 422


def test_indonesian_report_for_the_dashboard(replay, history):
    machine, at = _before_first_failure(replay)
    rows = history[history["datetime"] == at]
    result = inspect(Predictor.load(), rows, at, replay.failures, lang="id")
    report = result.report(lang="id")
    assert list(report.columns)[:3] == ["Mesin", "Status", "Risiko rusak"]
    assert report["Status"].iloc[0] == "Bahaya"
    assert report["Rusak dalam 24 jam (log)"].iloc[0].endswith("10 jam kemudian")
    finding = next(f for f in result.findings if f.machine_id == machine)
    assert finding.action.startswith("Kemungkinan rusak")
