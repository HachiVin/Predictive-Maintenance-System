import copy
import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.schemas import EXAMPLE
from pdm.model import ModelUnavailable, Predictor


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def request_body():
    return json.loads(EXAMPLE.read_text())


def test_health_reports_the_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["model_version"]


def test_example_request_ten_hours_before_a_failure_is_an_alert(client, request_body):
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ALERT" and body["risk"] >= body["thresholds"]["alert"]
    assert any("error" in s for s in body["signals"])


def test_metrics_count_served_predictions(client, request_body):
    before = client.get("/metrics").json().get("predictions", 0)
    client.post("/predict", json=request_body)
    summary = client.get("/metrics").json()
    assert summary["predictions"] == before + 1 and summary["latency_ms"]["p95"] > 0


def test_model_info_is_the_model_card(client):
    card = client.get("/model-info").json()
    assert card["thresholds"]["alert"] > card["thresholds"]["watch"] > 0
    assert card["test_metrics"]["alert"]["failures"] > 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda b: b.update(telemetry=b["telemetry"][:23]), "at least 24"),
        (lambda b: b["telemetry"][5].update(datetime=b["telemetry"][4]["datetime"]), "hourly"),
        (lambda b: b.update(replacements=b["replacements"][1:]), "replacements missing"),
        (lambda b: b.update(model="model9"), "model1"),
    ],
)
def test_bad_requests_are_explained(client, request_body, change, message):
    body = copy.deepcopy(request_body)
    change(body)
    response = client.post("/predict", json=body)
    assert response.status_code == 422 and message in response.text


def test_times_with_a_zone_are_read_as_utc(client, request_body):
    naive = client.post("/predict", json=request_body).json()
    for item in request_body["telemetry"] + request_body["errors"] + request_body["replacements"]:
        item["datetime"] += "+00:00"
    zoned = client.post("/predict", json=request_body).json()
    assert zoned["risk"] == naive["risk"] and zoned["as_of"] == naive["as_of"]


def test_missing_model_gives_503_not_500(monkeypatch, request_body):
    def fail(*_args, **_kwargs):
        raise ModelUnavailable("Model file not found")

    monkeypatch.setattr(Predictor, "load", classmethod(lambda cls, *a, **k: fail()))
    with TestClient(main.app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        response = c.post("/predict", json=request_body)
        assert response.status_code == 503 and "not found" in response.text
