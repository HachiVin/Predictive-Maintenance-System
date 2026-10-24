import json

import numpy as np
import pandas as pd
import pytest

from pdm.features import FEATURES
from pdm.model import ALERT, CARD_PATH, OK, WATCH, Predictor
from pdm.monitoring import Monitor, drift_level, psi


@pytest.fixture(scope="module")
def predictor():
    return Predictor.load()


def test_card_and_bundle_agree(predictor):
    card = json.loads(CARD_PATH.read_text())
    assert card["version"] == predictor.version
    assert card["thresholds"] == predictor.thresholds
    assert card["features"] == FEATURES == predictor.features


def test_status_follows_the_thresholds(predictor):
    alert, watch = predictor.thresholds["alert"], predictor.thresholds["watch"]
    assert predictor.status(alert) == ALERT
    assert predictor.status((alert + watch) / 2) == WATCH
    assert predictor.status(watch / 2) == OK


def test_rows_with_missing_history_are_refused(predictor):
    row = pd.DataFrame([{f: np.nan for f in FEATURES}])
    with pytest.raises(ValueError, match="24 hours"):
        predictor.risk(row)


def test_signals_name_errors_and_old_components(predictor):
    median = {f: q["p50"] for f, q in predictor.reference["quantiles"].items()}
    row = pd.Series(median)
    assert predictor.signals(row) == []
    row["error1_count_24h"] = 2
    row["days_since_comp2"] = predictor.reference["quantiles"]["days_since_comp2"]["p90"] + 50
    signals = predictor.signals(row)
    assert "error1 logged 2 times in 24 h" in signals and any("comp2" in s for s in signals)


def test_psi_is_small_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(1)
    train = rng.normal(170, 5, 20000)
    edges = np.quantile(train, np.linspace(0, 1, 11)).tolist()
    assert psi(edges, rng.normal(170, 5, 2000)) < 0.1
    assert psi(edges, rng.normal(178, 5, 2000)) > 0.25
    assert drift_level(None) == "not enough data" and drift_level(0.3) == "drifted"


def test_monitor_summary(predictor):
    monitor = Monitor(predictor.reference)
    for i in range(600):
        monitor.record(2.0 + i % 3, 0.1, OK, {"volt_mean_24h": 170.0 + i % 5})
    summary = monitor.summary()
    assert summary["predictions"] == 600 and summary["latency_ms"]["p50"] == 3.0
    assert summary["drift"]["volt_mean_24h"]["psi"] is not None
    assert summary["drift"]["rotate_mean_24h"]["level"] == "not enough data"
