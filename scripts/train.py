"""Train the failure-risk model and write models/pdm_rf.joblib, models/model_card.json and the
demo replay set in data/replay/.

    python scripts/download_data.py
    python scripts/train.py            # ~3 min on 2 CPU cores
    python scripts/train.py --compare  # also re-runs the v1 setup and a telemetry-only model

Time split (the data covers 2015):
    train       Jan 1 - Jun 29   fit the model
    validation  Jul 1 - Aug 30   pick the alert/watch thresholds
    test        Sep 1 - Dec 31   every number in the model card; the dashboard replays it
A day is dropped before each boundary so no training label looks into the next period.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdm import data  # noqa: E402
from pdm.features import (  # noqa: E402
    FEATURES,
    SENSORS,
    TELEMETRY_ONLY,
    build_features,
    complete,
)
from pdm.labels import HORIZON_HOURS, label  # noqa: E402
from pdm.model import CARD_PATH, MODEL_PATH  # noqa: E402
from pdm.monitoring import DRIFT_FEATURES  # noqa: E402

VERSION = "2.0.0"
TRAIN_END = pd.Timestamp("2015-07-01")
VALID_END = pd.Timestamp("2015-09-01")
GAP = pd.Timedelta(hours=HORIZON_HOURS)
SEED = 42


def forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def thresholds(y: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    """alert: the threshold with the best F1 on validation. watch: the lowest threshold at
    which at least half of the flagged hours still precede a failure, a wider early-look band."""
    precision, recall, cut = precision_recall_curve(y, risk)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    alert = float(cut[np.argmax(f1)])
    watch = float(cut[precision >= 0.5].min())
    return {"alert": round(alert, 3), "watch": round(min(watch, alert), 3)}


def evaluate(frame: pd.DataFrame, risk: np.ndarray, threshold: float, failures: pd.DataFrame):
    """Hour-level and failure-level metrics. ``frame`` holds machineID, datetime and y.

    A failure is caught when at least one alert falls in the 24 h before it; its lead time is
    measured from the first alert in the 48 h before it."""
    y = frame["y"].to_numpy()
    flagged = risk >= threshold
    tp = int((flagged & (y == 1)).sum())
    machine_days = len(frame) / 24
    start, end = frame["datetime"].min() + GAP, frame["datetime"].max()
    events = failures[(failures["datetime"] > start) & (failures["datetime"] <= end)]
    events = events[events["machineID"].isin(frame["machineID"].unique())]
    events = events.drop_duplicates(["machineID", "datetime"])
    alerts = frame.loc[flagged, ["machineID", "datetime"]]
    caught, lead = 0, []
    for event in events.itertuples():
        window = alerts[
            (alerts["machineID"] == event.machineID)
            & (alerts["datetime"] >= event.datetime - 2 * GAP)
            & (alerts["datetime"] < event.datetime)
        ]
        if len(window[window["datetime"] >= event.datetime - GAP]):
            caught += 1
            lead.append((event.datetime - window["datetime"].min()).total_seconds() / 3600)
    return {
        "pr_auc": round(float(average_precision_score(y, risk)), 3),
        "precision": round(tp / max(int(flagged.sum()), 1), 3),
        "recall": round(tp / max(int(y.sum()), 1), 3),
        "alerts_per_machine_day": round(float(flagged.sum()) / machine_days, 3),
        "failures": int(len(events)),
        "failures_caught": int(caught),
        "median_lead_hours": float(np.median(lead)) if lead else None,
        "positive_rate": round(float(y.mean()), 4),
    }


def v1_baseline(raw: data.Tables) -> dict:
    """The original notebook: label = the failure hour itself, random split, 16 features."""
    tel = raw.telemetry.merge(raw.machines, on="machineID")
    fail = raw.failures.drop_duplicates(["machineID", "datetime"]).assign(is_broken=1)
    tel = tel.merge(
        fail[["machineID", "datetime", "is_broken"]], on=["machineID", "datetime"], how="left"
    ).fillna({"is_broken": 0})
    tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    for s in SENSORS:
        roll = tel.groupby("machineID")[s].rolling(24)
        tel[f"{s}_mean_24h"] = roll.mean().reset_index(level=0, drop=True)
        tel[f"{s}_std_24h"] = roll.std().reset_index(level=0, drop=True)
    tel = tel.dropna()
    x = pd.get_dummies(
        tel.drop(columns=["machineID", "datetime", "is_broken"]), columns=["model"], drop_first=True
    )
    y = tel["is_broken"].astype(int)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1
    ).fit(x_tr, y_tr)
    risk = model.predict_proba(x_te)[:, 1]
    flagged = risk >= 0.5
    tp = int((flagged & (y_te.to_numpy() == 1)).sum())
    return {
        "setup": "v1: label = the failure hour, random split, sensors only",
        "pr_auc": round(float(average_precision_score(y_te, risk)), 3),
        "precision": round(tp / max(int(flagged.sum()), 1), 3),
        "recall": round(tp / max(int(y_te.sum()), 1), 3),
        "alerts_per_machine_day": round(float(flagged.sum()) / (len(x_te) / 24), 3),
    }


def latency(model, row: pd.DataFrame, n: int = 300) -> dict[str, float]:
    times = []
    for _ in range(n):
        start = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "p50": round(float(np.percentile(times, 50)), 2),
        "p95": round(float(np.percentile(times, 95)), 2),
    }


def write_replay(raw: data.Tables, test: pd.DataFrame, directory: Path) -> list[int]:
    """Test-period data of every machine, for the dashboard's inspections and the tests."""
    machines = sorted(test["machineID"].unique().tolist())
    start = VALID_END - pd.Timedelta(hours=HORIZON_HOURS)  # history for the first window
    directory.mkdir(parents=True, exist_ok=True)
    pick = lambda frame: frame[frame["machineID"].isin(machines)]  # noqa: E731
    tel = pick(raw.telemetry)
    tel = tel[tel["datetime"] >= start - pd.Timedelta(hours=1)]
    tel = tel.round({s: 3 for s in SENSORS})
    tel.to_csv(directory / "telemetry.csv.gz", index=False)
    errors = pick(raw.errors)
    errors[errors["datetime"] >= start - pd.Timedelta(hours=1)].to_csv(
        directory / "errors.csv", index=False
    )
    pick(raw.maint).to_csv(directory / "maint.csv", index=False)  # full history: days since
    fail = pick(raw.failures)
    fail[fail["datetime"] >= start].to_csv(directory / "failures.csv", index=False)
    pick(raw.machines).to_csv(directory / "machines.csv", index=False)
    return machines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="also train the baselines")
    args = parser.parse_args()

    raw = data.load_raw()
    print("building features", flush=True)
    frame = build_features(raw.telemetry, raw.errors, raw.maint, raw.machines)
    frame["y"] = label(frame, raw.failures)
    frame = complete(frame).reset_index(drop=True)
    t = frame["datetime"]
    train = frame[t < TRAIN_END - GAP]
    valid = frame[(t >= TRAIN_END) & (t < VALID_END - GAP)]
    test = frame[t >= VALID_END]
    print(f"rows train {len(train)} valid {len(valid)} test {len(test)}", flush=True)

    model = forest().fit(train[FEATURES], train["y"])
    model.n_jobs = 1  # one row at a time when serving: no thread start-up cost
    cut = thresholds(valid["y"].to_numpy(), model.predict_proba(valid[FEATURES])[:, 1])
    risk = model.predict_proba(test[FEATURES])[:, 1]
    metrics = {
        "alert": evaluate(test, risk, cut["alert"], raw.failures),
        "watch": evaluate(test, risk, cut["watch"], raw.failures),
    }
    print(json.dumps(metrics, indent=1), flush=True)

    importances = dict(
        sorted(
            zip(FEATURES, model.feature_importances_.round(4).tolist(), strict=True),
            key=lambda item: -item[1],
        )
    )
    reference = {
        "quantiles": {
            f: {f"p{int(q * 100):02d}": float(train[f].quantile(q)) for q in (0.01, 0.5, 0.9, 0.99)}
            for f in FEATURES
        },
        "deciles": {f: train[f].quantile(np.linspace(0, 1, 11)).tolist() for f in DRIFT_FEATURES},
        "valid_risk_deciles": np.quantile(
            model.predict_proba(valid[FEATURES])[:, 1], np.linspace(0, 1, 11)
        ).tolist(),
    }
    speed = latency(model, test[FEATURES].iloc[[0]])

    comparison = []
    if args.compare:
        print("baselines", flush=True)
        comparison.append(v1_baseline(raw))
        tele = forest().fit(train[TELEMETRY_ONLY], train["y"])
        tele_cut = thresholds(
            valid["y"].to_numpy(), tele.predict_proba(valid[TELEMETRY_ONLY])[:, 1]
        )
        tele_metrics = evaluate(
            test, tele.predict_proba(test[TELEMETRY_ONLY])[:, 1], tele_cut["alert"], raw.failures
        )
        comparison.append(
            {"setup": "v2: label = failure in next 24 h, time split, sensors only", **tele_metrics}
        )
    comparison.append(
        {"setup": "v2 shipped: + error counts and days since replacement", **metrics["alert"]}
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "thresholds": cut,
            "reference": reference,
            "version": VERSION,
            "importances": importances,
        },
        MODEL_PATH,
        compress=3,
    )
    machines = write_replay(raw, test, data.REPLAY_DIR)
    card = {
        "version": VERSION,
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "sklearn": sklearn.__version__,
        "target": f"failure of any component within the next {HORIZON_HOURS} hours",
        "algorithm": "RandomForestClassifier(n_estimators=100, max_depth=10, "
        "min_samples_leaf=5, class_weight='balanced')",
        "data": {
            "source": "Microsoft Azure Predictive Maintenance dataset (synthetic), 100 machines, "
            "hourly telemetry for 2015",
            "train": [str(train["datetime"].min().date()), str(train["datetime"].max().date())],
            "validation": [
                str(valid["datetime"].min().date()),
                str(valid["datetime"].max().date()),
            ],
            "test": [str(test["datetime"].min().date()), str(test["datetime"].max().date())],
            "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
            "positive_rate_train": round(float(train["y"].mean()), 4),
        },
        "features": FEATURES,
        "thresholds": cut,
        "test_metrics": metrics,
        "comparison": comparison,
        "latency_ms_single_row": speed,
        "importances": importances,
        "replay_machines": len(machines),
    }
    CARD_PATH.write_text(json.dumps(card, indent=2))
    size = MODEL_PATH.stat().st_size / 1e6
    print(f"saved {MODEL_PATH.name} ({size:.1f} MB) and {CARD_PATH.name}")


if __name__ == "__main__":
    main()
