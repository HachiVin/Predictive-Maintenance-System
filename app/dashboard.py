"""Streamlit dashboard: replays the test period (Sep-Dec 2015) hour by hour.

    streamlit run app/dashboard.py

The layout follows high-performance HMI practice (ISA-101) used in plant control rooms: grey
background, colour only for abnormal states, a shape next to every status so it never depends
on colour alone, and every reading shown against its normal range.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdm import data  # noqa: E402
from pdm.features import SENSORS, build_features, complete  # noqa: E402
from pdm.model import ALERT, CARD_PATH, OK, WATCH, ModelUnavailable, Predictor  # noqa: E402
from pdm.monitoring import DRIFT_FEATURES, Monitor, drift_level, psi  # noqa: E402

ALARM, CAUTION, QUIET = "#d64545", "#d08a1e", "#8a949c"  # readable on light and dark grey
STATUS_LABEL = {ALERT: "▲ Alert", WATCH: "◆ Watch", OK: "Normal"}
SENSOR_UNITS = {"volt": "V", "rotate": "rpm", "pressure": "psi", "vibration": "mm/s"}
DETAIL_DAYS = 7
DRIFT_DAYS = 14
RISK_HEIGHT = 230

st.set_page_config(page_title="Failure risk", layout="wide")
st.html(
    """<style>
    .block-container {padding-top: 2.2rem; max-width: 1400px}
    body, [data-testid="stDataFrame"] {font-variant-numeric: tabular-nums}
    [data-testid="stMetricValue"] {font-size: 1.6rem}
    </style>"""
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor.load()


@st.cache_data
def load_card() -> dict:
    return json.loads(CARD_PATH.read_text())


@st.cache_data
def load_replay() -> tuple[data.Tables, pd.DataFrame, float]:
    """Replay tables and every machine-hour scored once (charts read history from here)."""
    tables = data.load_replay()
    frame = complete(
        build_features(tables.telemetry, tables.errors, tables.maint, tables.machines)
    ).reset_index(drop=True)
    start = time.perf_counter()
    frame["risk"] = load_predictor().risk(frame)
    seconds = time.perf_counter() - start
    return tables, frame, seconds


try:
    predictor = load_predictor()
except ModelUnavailable as exc:
    st.error(f"The model could not be loaded. {exc}")
    st.stop()

card = load_card()
tables, history, batch_seconds = load_replay()
hours = sorted(history["datetime"].unique())
first, last = pd.Timestamp(hours[0]).to_pydatetime(), pd.Timestamp(hours[-1]).to_pydatetime()
machines = tables.machines.set_index("machineID")
failures = tables.failures.drop_duplicates(["machineID", "datetime", "failure"])

state = st.session_state
state.setdefault("now", first + timedelta(hours=18))
state.setdefault("playing", False)
state.setdefault("step", 3)
state.setdefault("machine", "Highest risk")
state.setdefault("monitor", Monitor(predictor.reference))


def next_failure_text(machine_id: int, now: datetime) -> str:
    upcoming = failures[(failures["machineID"] == machine_id) & (failures["datetime"] > now)]
    if upcoming.empty:
        return ""
    hours_left = (upcoming["datetime"].min() - pd.Timestamp(now)).total_seconds() / 3600
    if hours_left > 48:
        return ""
    comps = ", ".join(upcoming.loc[upcoming["datetime"] == upcoming["datetime"].min(), "failure"])
    return f"{comps} in {hours_left:.0f} h"


def skip_to_next_failure() -> None:
    later = failures[failures["datetime"] > pd.Timestamp(state.now) + timedelta(hours=30)]
    if not later.empty:
        state.now = (later["datetime"].min() - timedelta(hours=30)).to_pydatetime()
    state.playing = False


def fleet_now(now: datetime) -> pd.DataFrame:
    """Score the current hour of every machine live, the way the API would."""
    rows = history[history["datetime"] == pd.Timestamp(now)].reset_index(drop=True)
    start = time.perf_counter()
    assessments = predictor.assess(rows)
    per_row_ms = (time.perf_counter() - start) * 1000 / max(len(rows), 1)
    for (_i, row), a in zip(rows.iterrows(), assessments, strict=True):
        state.monitor.record(per_row_ms, a.risk, a.status, row.to_dict())
    order = {ALERT: 0, WATCH: 1, OK: 2}
    table = pd.DataFrame(
        {
            "Machine": rows["machineID"].astype(int),
            "Status": [STATUS_LABEL[a.status] for a in assessments],
            "Risk": [a.risk for a in assessments],
            "Top signal": [
                a.signals[0] if a.status != OK and a.signals else "" for a in assessments
            ],
            "Recorded failure": [next_failure_text(int(m), now) for m in rows["machineID"]],
            "_order": [order[a.status] for a in assessments],
            "_assessment": assessments,
        }
    )
    return table.sort_values(["_order", "Risk"], ascending=[True, False]).drop(columns="_order")


def detail_start(now: datetime) -> pd.Timestamp:
    return max(pd.Timestamp(now) - timedelta(days=DETAIL_DAYS), pd.Timestamp(first))


def risk_chart(machine_id: int, now: datetime) -> alt.LayerChart:
    start = detail_start(now)
    window = history[(history["machineID"] == machine_id) & history["datetime"].between(start, now)]
    base = alt.Chart(window).encode(
        x=alt.X(
            "datetime:T",
            title=None,
            scale=alt.Scale(domain=[start, pd.Timestamp(now)]),
            axis=alt.Axis(format="%a %d %b", tickCount="day", labelOverlap=True),
        )
    )
    line = base.mark_line(color=QUIET, strokeWidth=1.6, interpolate="step-after").encode(
        y=alt.Y("risk:Q", title="Risk of failure within 24 h", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("datetime:T", format="%d %b %H:%M"),
            alt.Tooltip("risk:Q", format=".3f"),
        ],
    )
    flagged = (
        base.mark_circle(size=22)
        .encode(
            y="risk:Q",
            color=alt.Color(
                "level:N",
                scale=alt.Scale(domain=["Alert", "Watch"], range=[ALARM, CAUTION]),
                legend=None,
            ),
        )
        .transform_calculate(
            level=f"datum.risk >= {predictor.thresholds['alert']} ? 'Alert' : "
            f"(datum.risk >= {predictor.thresholds['watch']} ? 'Watch' : 'Normal')"
        )
        .transform_filter("datum.level != 'Normal'")
    )
    limits = pd.DataFrame(
        {
            "y": [predictor.thresholds["alert"], predictor.thresholds["watch"]],
            "label": ["alert threshold", "watch threshold"],
            "color": [ALARM, CAUTION],
        }
    )
    rules = (
        alt.Chart(limits)
        .mark_rule(strokeDash=[4, 3], strokeWidth=1)
        .encode(y="y:Q", color=alt.Color("color:N", scale=None))
    )
    rule_text = (
        alt.Chart(limits)
        .mark_text(align="left", dx=4, dy=-6, fontSize=11)
        .encode(x=alt.value(0), y="y:Q", text="label:N", color=alt.Color("color:N", scale=None))
    )
    layers = [rules, rule_text, line, flagged]

    events = []
    fail = failures[
        (failures["machineID"] == machine_id)
        & failures["datetime"].between(start, pd.Timestamp(now))
    ]
    for r in fail.itertuples():
        events.append({"datetime": r.datetime, "what": f"{r.failure} failed", "kind": "failure"})
    err = tables.errors[
        (tables.errors["machineID"] == machine_id)
        & tables.errors["datetime"].between(start, pd.Timestamp(now))
    ]
    for r in err.itertuples():
        events.append({"datetime": r.datetime, "what": r.errorID, "kind": "error"})
    rep = tables.maint[
        (tables.maint["machineID"] == machine_id)
        & tables.maint["datetime"].between(start, pd.Timestamp(now))
    ]
    failed_at = set(zip(fail["datetime"], fail["failure"], strict=False))
    for r in rep.itertuples():
        if (r.datetime, r.comp) not in failed_at:
            events.append({"datetime": r.datetime, "what": f"{r.comp} replaced", "kind": "planned"})
    if events:
        ev = pd.DataFrame(events)
        kinds = alt.Scale(domain=["failure", "error", "planned"], range=[ALARM, CAUTION, QUIET])
        layers.append(
            alt.Chart(ev[ev["kind"] == "failure"])
            .mark_rule(strokeWidth=2)
            .encode(
                x="datetime:T",
                color=alt.Color("kind:N", scale=kinds, legend=None),
                tooltip=["what:N", alt.Tooltip("datetime:T", format="%d %b %H:%M")],
            )
        )
        layers.append(
            alt.Chart(ev[ev["kind"] != "failure"])
            .mark_tick(thickness=2, size=12)
            .encode(
                x="datetime:T",
                y=alt.datum(0.03),
                color=alt.Color("kind:N", scale=kinds, legend=None),
                tooltip=["what:N", alt.Tooltip("datetime:T", format="%d %b %H:%M")],
            )
        )
    return alt.layer(*layers).properties(height=RISK_HEIGHT)


def sensor_chart(machine_id: int, now: datetime, sensor: str) -> alt.LayerChart:
    start = detail_start(now)
    window = history[(history["machineID"] == machine_id) & history["datetime"].between(start, now)]
    q = predictor.reference["quantiles"][f"{sensor}_mean_24h"]
    band = (
        alt.Chart(pd.DataFrame({"lo": [q["p01"]], "hi": [q["p99"]]}))
        .mark_rect(opacity=0.14, color=QUIET)
        .encode(y="lo:Q", y2="hi:Q")
    )
    base = alt.Chart(window).encode(
        x=alt.X(
            "datetime:T",
            title=None,
            axis=alt.Axis(format="%d %b", tickCount="day", labelOverlap=True),
        )
    )
    raw = base.mark_line(strokeWidth=0.8, opacity=0.35, color=QUIET).encode(
        y=alt.Y(
            f"{sensor}:Q", title=f"{sensor} ({SENSOR_UNITS[sensor]})", scale=alt.Scale(zero=False)
        )
    )
    mean = base.mark_line(strokeWidth=1.8, color=QUIET).encode(y=f"{sensor}_mean_24h:Q")
    high = (
        base.mark_line(strokeWidth=1.8, color=ALARM)
        .encode(y=f"{sensor}_mean_24h:Q")
        .transform_filter(
            f"datum['{sensor}_mean_24h'] > {q['p99']} || datum['{sensor}_mean_24h'] < {q['p01']}"
        )
    )
    return alt.layer(band, raw, mean, high).properties(height=140)


def fleet_view() -> None:
    if state.playing:
        state.now = min(state.now + timedelta(hours=state.step), last)
        if state.now >= last:
            state.playing = False

    st.markdown("## Failure risk, next 24 hours")

    controls = st.columns([6, 1.2, 1.4, 2.4], vertical_alignment="bottom")
    controls[0].slider(
        "Replay time",
        min_value=first,
        max_value=last,
        step=timedelta(hours=1),
        key="now",
        format="DD MMM YYYY, HH:mm",
        label_visibility="collapsed",
    )
    if controls[1].button("Pause" if state.playing else "Play", width="stretch"):
        state.playing = not state.playing
        st.rerun()
    controls[2].selectbox(
        "Hours per second",
        [1, 3, 6, 12],
        key="step",
        label_visibility="collapsed",
        format_func=lambda h: f"{h} h per second",
    )
    controls[3].button(
        "Skip to 30 h before next failure", on_click=skip_to_next_failure, width="stretch"
    )

    table = fleet_now(state.now)
    counts = table["Status"].value_counts()
    alert, watch = counts.get(STATUS_LABEL[ALERT], 0), counts.get(STATUS_LABEL[WATCH], 0)
    st.markdown(
        f"{state.now:%A %d %B %Y, %H:%M}. "
        f"**{alert} {'machine' if alert == 1 else 'machines'} in alert, {watch} on watch,** "
        f"{len(table) - alert - watch} normal. Replaying Sep to Dec 2015, months the model "
        f"never saw, for {len(table)} of the 100 machines."
    )

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.dataframe(
            table.drop(columns="_assessment"),
            hide_index=True,
            width="stretch",
            height=35 * (len(table) + 1) + 3,
            column_config={
                "Machine": st.column_config.NumberColumn(width="small", format="%d"),
                "Status": st.column_config.TextColumn(width="small"),
                "Risk": st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="%.2f", width="small"
                ),
                "Top signal": st.column_config.TextColumn(width="medium"),
                "Recorded failure": st.column_config.TextColumn(
                    width="small",
                    help="From the failure log, to judge the alerts. The model never sees it.",
                ),
            },
        )
        st.caption(
            f"Alert at risk ≥ {predictor.thresholds['alert']:.2f}, watch at ≥ "
            f"{predictor.thresholds['watch']:.2f}; both chosen on Jul to Aug data. "
            "Signals are readings outside the training range, ordered by how much the model "
            "relies on them. They are pointers for the technician, not causes."
        )
    with right:
        options = ["Highest risk", *[int(m) for m in sorted(machines.index)]]
        choice = st.selectbox(
            "Machine",
            options,
            key="machine",
            format_func=lambda m: m if isinstance(m, str) else f"Machine {m}",
        )
        machine_id = int(table["Machine"].iloc[0]) if choice == "Highest risk" else int(choice)
        info = machines.loc[machine_id]
        current = table.loc[table["Machine"] == machine_id, "_assessment"].iloc[0]
        st.markdown(f"#### Machine {machine_id}")
        summary = f"{STATUS_LABEL[current.status]}, risk {current.risk:.2f}. {current.action}"
        if current.signals:
            summary += f" Signals: {'; '.join(current.signals)}."
        st.markdown(summary)
        st.caption(
            f"{info['model']}, {int(info['age'])} years old. Last {DETAIL_DAYS} days. "
            "Ticks along the bottom: errors (amber), planned replacements (grey); "
            "red line: recorded failure."
        )
        st.altair_chart(risk_chart(machine_id, state.now), width="stretch")
        grid = st.columns(2)
        for i, sensor in enumerate(SENSORS):
            with grid[i % 2]:
                st.altair_chart(sensor_chart(machine_id, state.now, sensor), width="stretch")
        st.caption(
            "Thin line: hourly reading. Thick line: 24 h mean, red where it leaves the "
            "grey band (1st to 99th percentile of the training months)."
        )


def model_view() -> None:
    alert = card["test_metrics"]["alert"]
    st.markdown("## How well it works")
    st.markdown(
        f"Tested on Sep to Dec 2015, which the model never saw during training or threshold "
        f"selection. It alerted in the 24 hours before **{alert['failures_caught']} of "
        f"{alert['failures']} failures**, the first alert came a median "
        f"**{alert['median_lead_hours']:.0f} h** ahead, and it raised "
        f"**{alert['alerts_per_machine_day']:.2f} alerts per machine per day**, "
        f"{alert['precision']:.1%} of them in the 24 hours before a real failure."
    )
    rows = []
    for item in card["comparison"]:
        caught = (
            f"{item['failures_caught']} of {item['failures']}"
            if "failures" in item
            else "not measured"
        )
        rows.append(
            {
                "Setup": item["setup"],
                "PR-AUC": item["pr_auc"],
                "Precision": item["precision"],
                "Recall": item["recall"],
                "Alerts per machine-day": item["alerts_per_machine_day"],
                "Failures caught": caught,
            }
        )
    ratio = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            c: ratio for c in ("PR-AUC", "Precision", "Recall", "Alerts per machine-day")
        },
    )
    st.caption(
        "v1 is the first version of this project: it labelled only the failure hour and "
        "split rows at random, so neighbouring hours of the same day were in both train "
        "and test. Its 81% recall came with 99% false alarms."
    )

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("#### What the model relies on")
        top = pd.DataFrame(
            list(card["importances"].items())[:12], columns=["feature", "importance"]
        )
        st.altair_chart(
            alt.Chart(top)
            .mark_bar(color=QUIET)
            .encode(
                x=alt.X("importance:Q", title="Share of impurity reduction"),
                y=alt.Y("feature:N", sort="-x", title=None),
            )
            .properties(height=320),
            width="stretch",
        )
    with right:
        sensors = [
            c
            for c in card["comparison"]
            if c["setup"].startswith("v2:") and c["setup"].endswith("sensors only")
        ]
        sensors_only = (
            f"With sensors only, PR-AUC drops to {sensors[0]['pr_auc']:.2f}." if sensors else ""
        )
        st.markdown("#### Limits of this result")
        st.markdown(
            "- The Azure PdM dataset is **synthetic**. Its errors were generated to precede "
            "failures, which is why error counts are the strongest features and the scores are "
            "near perfect. Real plants are noisier; expect lower numbers.\n"
            "- 100 machines, one year, four component types. Thresholds should be re-chosen on "
            "a plant's own data and its cost of a missed failure vs. a needless inspection.\n"
            f"- The model needs the error log and replacement log. {sensors_only}\n"
            "- Signals compare readings with training percentiles; they are not a "
            "causal explanation."
        )
        st.markdown("#### Model")
        st.markdown(
            f"{card['algorithm']}, {len(card['features'])} features, version {card['version']}, "
            f"trained {card['trained_at']} with scikit-learn {card['sklearn']}. Train "
            f"{card['data']['train'][0]} to {card['data']['train'][1]}, validation "
            f"{card['data']['validation'][0]} to {card['data']['validation'][1]}, test "
            f"{card['data']['test'][0]} to {card['data']['test'][1]}."
        )


def health_view() -> None:
    st.markdown("## Model health")
    summary = state.monitor.summary()
    st.markdown(
        "What you would watch after a release: how fast predictions are, how often the model "
        "alerts, and whether incoming data still looks like the training data. The API serves "
        "the same numbers at `/metrics`."
    )
    cols = st.columns(4)
    count = summary.get("predictions", 0)
    cols[0].metric("Predictions this session", f"{count:,}")
    if count:
        cols[1].metric("Model latency p50", f"{summary['latency_ms']['p50']:.2f} ms")
        cols[2].metric("Model latency p95", f"{summary['latency_ms']['p95']:.2f} ms")
        cols[3].metric("Share in alert", f"{summary['status_share'].get(ALERT, 0):.1%}")
    st.caption(
        f"Per machine-hour, scored in batches of {len(machines)}. A single API request also "
        f"builds features from raw records and takes about 40 ms. Scoring the whole replay "
        f"({len(history):,} machine-hours) at start-up took {batch_seconds:.1f} s."
    )

    st.markdown("#### Input drift, last 14 days")
    now = pd.Timestamp(state.now)
    seen = history[history["datetime"].between(now - timedelta(days=DRIFT_DAYS), now)]
    enough = now - pd.Timestamp(first) >= timedelta(days=7)
    drift = []
    for feature in DRIFT_FEATURES:
        values = seen[feature].to_numpy()
        score = psi(predictor.reference["deciles"][feature], values) if enough else None
        drift.append(
            {
                "Feature": feature,
                "PSI": None if score is None else round(score, 3),
                "Level": drift_level(score),
                "Mean": values.mean(),
                "Training median": predictor.reference["quantiles"][feature]["p50"],
            }
        )
    st.dataframe(
        pd.DataFrame(drift),
        hide_index=True,
        width="stretch",
        column_config={
            "Mean": st.column_config.NumberColumn(format="%.2f"),
            "Training median": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "Population Stability Index of the replayed hours against training deciles: "
        "below 0.1 stable, 0.1 to 0.25 shifting, above 0.25 drifted. It needs a week of "
        "data: hourly readings of one machine move together, so a few days swing by "
        "chance. Drift is a reason to re-check the thresholds or retrain, not proof the "
        "model is wrong."
    )

    if count:
        risks = np.array([r.risk for r in state.monitor.records()])
        st.markdown("#### Risk scores served this session")
        st.altair_chart(
            alt.Chart(pd.DataFrame({"risk": risks}))
            .mark_bar(color=QUIET)
            .encode(
                x=alt.X("risk:Q", bin=alt.Bin(step=0.05), title="Risk"),
                y=alt.Y("count():Q", title="Machine-hours"),
            )
            .properties(height=200),
            width="stretch",
        )


fleet, model, health = st.tabs(["Fleet", "Model", "Health"])
with fleet:
    st.fragment(run_every=1.0 if state.playing else None)(fleet_view)()
with model:
    model_view()
with health:
    st.fragment(run_every=2.0 if state.playing else None)(health_view)()
