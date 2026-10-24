"""Streamlit dashboard: shift inspections and machine history on Sep-Dec 2015 data.

    streamlit run app/dashboard.py

The interface is in Indonesian, for the maintenance team; code, API and README are in English.
Inspection is the main view: pick a shift, check all 100 machines in groups of 16, get the list
of machines to inspect and check it against the failure log. Machine history replays one
machine hour by hour. Each machine is drawn as a small pump set whose lamp and outline change
colour with its status; the status is also written as a word, so it never depends on colour.
"""

from __future__ import annotations

import base64
import json
import random
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
from pdm.inspection import STATUS_WORDS, Inspection, estimate_savings, inspect  # noqa: E402
from pdm.model import (  # noqa: E402
    ALERT,
    CARD_PATH,
    OK,
    WATCH,
    ModelUnavailable,
    Predictor,
)
from pdm.monitoring import DRIFT_FEATURES, Monitor, drift_level, psi  # noqa: E402

ALARM, CAUTION, QUIET = "#d64545", "#d08a1e", "#8a949c"  # readable on light and dark grey
GOOD, SKETCH = "#3f9b5f", "#7d8790"
STATUS_COLOR = {ALERT: ALARM, WATCH: CAUTION, OK: GOOD}
STATUS_WORD = STATUS_WORDS["id"]  # Bahaya, Waspada, Normal
LANG = "id"
# The dataset names the sensors but gives no units; only volt and rotate imply one.
SENSOR_TITLES = {
    "volt": "tegangan (V)",
    "rotate": "putaran (rpm)",
    "pressure": "tekanan",
    "vibration": "getaran",
}
DRIFT_WORDS = {
    "stable": "stabil",
    "shifting": "bergeser",
    "drifted": "berubah",
    "not enough data": "data belum cukup",
}
SETUP_WORDS = {
    "v1: label = the failure hour, random split, sensors only": (
        "v1: label = jam kerusakan, data dibagi acak, sensor saja"
    ),
    "v2: label = failure in next 24 h, time split, sensors only": (
        "v2: label = rusak dalam 24 jam, dibagi per waktu, sensor saja"
    ),
    "v2 shipped: + error counts and days since replacement": (
        "v2 dipakai: + jumlah error dan umur komponen"
    ),
}
BATCH = 16  # machines shown per step of an inspection, as in the first version
DETAIL_DAYS = 7
DRIFT_DAYS = 14
RISK_HEIGHT = 230
VIEWS = ["Inspeksi", "Riwayat mesin", "Seberapa akurat?", "Kesehatan sistem"]
DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
MONTHS = [
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
]

st.set_page_config(page_title="Inspeksi shift", layout="wide")
st.html(
    """<style>
    .block-container {padding-top: 4.5rem; max-width: 1400px}  /* clear of the Cloud header */
    body, [data-testid="stDataFrame"] {font-variant-numeric: tabular-nums}
    [data-testid="stMetricValue"] {font-size: 1.6rem}
    .pdm-tile {border-radius: 6px; padding: 10px 12px 8px; background: rgba(127,127,127,.07);
        margin-bottom: 10px}
    .pdm-head {display: flex; justify-content: space-between; align-items: baseline}
    .pdm-name {font-weight: 600; font-size: 1.02rem}
    .pdm-status {font-weight: 600; font-size: .9rem}
    .pdm-risk {display: flex; justify-content: space-between; font-size: .88rem; margin-top: 2px}
    .pdm-bar {height: 6px; border-radius: 3px; background: rgba(127,127,127,.2); margin: 4px 0 6px}
    .pdm-bar div {height: 6px; border-radius: 3px}
    .pdm-reason {font-size: .86rem; min-height: 1.3em; white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis}
    .pdm-log {font-size: .78rem; opacity: .7; min-height: 1.2em}
    .pdm-sketch {display: block; width: 100%; height: 84px; margin: 6px 0 4px}
    .pdm-compact {padding: 8px 9px 6px}
    .pdm-compact .pdm-head {flex-direction: column; align-items: flex-start}
    .pdm-compact .pdm-sketch {height: 56px}
    .pdm-compact .pdm-log {display: none}
    .pdm-legend {display: flex; flex-wrap: wrap; gap: 6px 22px; font-size: .9rem}
    .pdm-legend i {display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        margin-right: 7px}
    .pdm-map {display: grid; grid-template-columns: repeat(10, minmax(0, 1fr)); gap: 5px}
    .pdm-cell {border-radius: 4px; padding: 5px 0; text-align: center; font-size: .82rem;
        border: 1px solid rgba(127,127,127,.3)}
    .pdm-big {font-size: 1.25rem; font-weight: 600; margin: 4px 0 2px}
    </style>"""
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor.load()


@st.cache_data
def load_card() -> dict:
    return json.loads(CARD_PATH.read_text())


@st.cache_data
def load_replay() -> tuple[data.Tables, pd.DataFrame]:
    """Replay tables and every machine-hour with its features and risk (charts read from it)."""
    tables = data.load_replay()
    frame = complete(
        build_features(tables.telemetry, tables.errors, tables.maint, tables.machines)
    ).reset_index(drop=True)
    frame["risk"] = load_predictor().risk(frame)
    return tables, frame


@st.cache_data
def scaling_test() -> pd.DataFrame:
    """Model time for 100, 1,000 and 10,000 machines, measured on the machine running this."""
    sample = history.sample(10_000, random_state=0, replace=len(history) < 10_000)
    out = []
    for n in (100, 1_000, 10_000):
        rows = sample.head(n)
        start = time.perf_counter()
        predictor.risk(rows)
        out.append({"Jumlah mesin": n, "Waktu model (ms)": (time.perf_counter() - start) * 1000})
    return pd.DataFrame(out)


try:
    predictor = load_predictor()
except ModelUnavailable as exc:
    st.error(f"Model tidak bisa dimuat. {exc}")
    st.stop()

card = load_card()
tables, history = load_replay()
hours = sorted(history["datetime"].unique())
first, last = pd.Timestamp(hours[0]).to_pydatetime(), pd.Timestamp(hours[-1]).to_pydatetime()
machines = tables.machines.set_index("machineID")
machine_ids = sorted(int(m) for m in machines.index)
failures = tables.failures.drop_duplicates(["machineID", "datetime", "failure"])

state = st.session_state
state.setdefault("view", VIEWS[0])
state.setdefault("inspection", None)
state.setdefault("inspections", [])  # every inspection run in this session, for System health
_demo = failures[failures["datetime"] >= "2015-10-01"]["datetime"].min() - timedelta(hours=10)
state.setdefault("shift_day", _demo.date())  # opens on a shift with a breakdown coming
state.setdefault("shift_hour", int(_demo.hour))
state.setdefault("now", first + timedelta(hours=18))
state.setdefault("playing", False)
state.setdefault("step", 3)
state.setdefault("machine", machine_ids[0])
state.setdefault("monitor", Monitor(predictor.reference))


def tanggal(ts, hari: bool = True, jam: bool = True) -> str:
    """Indonesian date, e.g. "Rabu, 30 September 2015, 20:00"."""
    ts = pd.Timestamp(ts)
    text = f"{ts.day} {MONTHS[ts.month - 1]} {ts.year}"
    if hari:
        text = f"{DAYS[ts.weekday()]}, {text}"
    return f"{text}, {ts:%H:%M}" if jam else text


def bind(name: str) -> dict:
    """Widget arguments that keep ``state[name]`` when the widget's view is not shown.

    Streamlit drops a widget's state when the widget is not rendered, so switching views
    would reset the shift time, the machine and the replay clock. The value lives under
    ``name``; the widget gets its own key, filled from it on every run.
    """
    widget_key = f"w_{name}"
    state[widget_key] = state[name]

    def copy_back() -> None:
        state[name] = state[widget_key]

    return {"key": widget_key, "on_change": copy_back}


def legend() -> None:
    st.html(
        f"""<div class="pdm-legend">
        <span><i style="background:{GOOD}"></i>Normal: tidak perlu tindakan</span>
        <span><i style="background:{CAUTION}"></i>Waspada: cek saat berhenti terjadwal</span>
        <span><i style="background:{ALARM}"></i>Bahaya: periksa sebelum shift berikutnya</span>
        </div>"""
    )


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
            axis=alt.Axis(format="%d/%m", tickCount="day", labelOverlap=True),
        )
    )
    line = base.mark_line(color=QUIET, strokeWidth=1.6, interpolate="step-after").encode(
        y=alt.Y(
            "risk:Q",
            title="Risiko rusak",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        tooltip=[
            alt.Tooltip("datetime:T", title="waktu", format="%d/%m/%Y %H:%M"),
            alt.Tooltip("risk:Q", title="risiko", format=".0%"),
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
            "label": ["bahaya", "waspada"],
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
        events.append({"datetime": r.datetime, "what": f"{r.failure} rusak", "kind": "failure"})
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
            events.append({"datetime": r.datetime, "what": f"{r.comp} diganti", "kind": "planned"})
    if events:
        ev = pd.DataFrame(events)
        kinds = alt.Scale(domain=["failure", "error", "planned"], range=[ALARM, CAUTION, QUIET])
        layers.append(
            alt.Chart(ev[ev["kind"] == "failure"])
            .mark_rule(strokeWidth=2)
            .encode(
                x="datetime:T",
                color=alt.Color("kind:N", scale=kinds, legend=None),
                tooltip=[
                    alt.Tooltip("what:N", title="kejadian"),
                    alt.Tooltip("datetime:T", title="waktu", format="%d/%m/%Y %H:%M"),
                ],
            )
        )
        layers.append(
            alt.Chart(ev[ev["kind"] != "failure"])
            .mark_tick(thickness=2, size=12)
            .encode(
                x="datetime:T",
                y=alt.datum(0.03),
                color=alt.Color("kind:N", scale=kinds, legend=None),
                tooltip=[
                    alt.Tooltip("what:N", title="kejadian"),
                    alt.Tooltip("datetime:T", title="waktu", format="%d/%m/%Y %H:%M"),
                ],
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
            axis=alt.Axis(format="%d/%m", tickCount="day", labelOverlap=True),
        )
    )
    raw = base.mark_line(strokeWidth=0.8, opacity=0.35, color=QUIET).encode(
        y=alt.Y(f"{sensor}:Q", title=SENSOR_TITLES[sensor], scale=alt.Scale(zero=False))
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


@st.cache_data
def machine_sketch(status: str) -> str:
    """A pump set on its skid: motor with cooling fins, coupling, pump, pipes and a status lamp,
    as an <img> data URI (st.html strips inline SVG). Mid-grey lines read on light and dark."""
    line, lamp = SKETCH, STATUS_COLOR[status]
    pump = lamp if status == ALERT else line
    blink = "b 1.4s ease-in-out infinite" if status == ALERT else "none"  # only alerts move
    impeller = f'fill="{ALARM}" fill-opacity="0.35"' if status == ALERT else 'fill="none"'
    fins = "".join(
        f'<line x1="{x}" y1="35" x2="{x}" y2="63" stroke-opacity="0.5"/>' for x in range(22, 66, 7)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 80" fill="none"
      stroke="{line}" stroke-width="1.6" stroke-linejoin="round">
      <rect x="8" y="68" width="144" height="6" rx="1"/>
      <rect x="10" y="37" width="4" height="24" rx="1"/>
      <rect x="14" y="30" width="58" height="38" rx="5"/>{fins}
      <rect x="41" y="25" width="4" height="5"/>
      <style>.lamp{{animation:{blink}}}@keyframes b{{50%{{opacity:.3}}}}
        @media (prefers-reduced-motion: reduce){{.lamp{{animation:none}}}}</style>
      <circle class="lamp" cx="43" cy="19" r="5.5" fill="{lamp}" stroke="{lamp}"/>
      <rect x="72" y="44" width="13" height="10" rx="1"/>
      <circle cx="106" cy="49" r="19" stroke="{pump}"/>
      <circle cx="106" cy="49" r="7" stroke="{pump}" {impeller}/>
      <rect x="100" y="12" width="12" height="18" stroke="{pump}"/>
      <line x1="97" y1="12" x2="115" y2="12" stroke-width="2.4"/>
      <rect x="125" y="44" width="24" height="10" stroke="{pump}"/>
      <line x1="149" y1="41" x2="149" y2="57" stroke-width="2.4"/>
    </svg>"""
    encoded = base64.b64encode(svg.encode()).decode()
    return f'<img class="pdm-sketch" alt="sketsa mesin" src="data:image/svg+xml;base64,{encoded}">'


def machine_tile(
    machine_id: int, status: str, risk: float, signals: list[str], log: str, compact: bool = False
) -> str:
    color = STATUS_COLOR[status]
    border = f"2px solid {color}" if status != OK else "2px solid rgba(127,127,127,.28)"
    reason = signals[0] if status != OK and signals else "Berjalan normal"
    width = f"{max(risk, 0.02):.0%}"
    log_line = f"Kunci jawaban (log): {log}" if log else "&nbsp;"
    label = "Risiko" if compact else "Risiko rusak"
    return f"""<div class="pdm-tile{" pdm-compact" if compact else ""}" style="border:{border}">
      <div class="pdm-head"><span class="pdm-name">Mesin {machine_id}</span>
        <span class="pdm-status" style="color:{color}">{STATUS_WORD[status]}</span></div>
      {machine_sketch(status)}
      <div class="pdm-risk"><span>{label}</span><b>{risk:.0%}</b></div>
      <div class="pdm-bar"><div style="width:{width};background:{color}"></div></div>
      <div class="pdm-reason">{reason}</div><div class="pdm-log">{log_line}</div>
    </div>"""


def tiles(findings, columns: int = 4, show_log: bool = False, compact: bool = False) -> None:
    cols = st.columns(columns)
    for i, f in enumerate(findings):
        with cols[i % columns]:
            st.html(
                machine_tile(
                    f.machine_id,
                    f.status,
                    f.risk,
                    f.signals,
                    f.breakdown if show_log else "",
                    compact,
                )
            )


def plant_map(findings) -> str:
    cells = []
    for f in findings:
        color = STATUS_COLOR[f.status]
        style = f"background:{color};color:#fff;border-color:{color}" if f.status != OK else ""
        label = f"Mesin {f.machine_id}: {STATUS_WORD[f.status]}, risiko {f.risk:.0%}"
        cells.append(f'<div class="pdm-cell" style="{style}" title="{label}">{f.machine_id}</div>')
    return f'<div class="pdm-map">{"".join(cells)}</div>'


def shift_time() -> pd.Timestamp:
    at = datetime.combine(state.shift_day, datetime.min.time()) + timedelta(hours=state.shift_hour)
    return pd.Timestamp(min(max(at, first), last))


def random_shift() -> None:
    hour = pd.Timestamp(random.choice(hours))
    state.shift_day, state.shift_hour = hour.date(), int(hour.hour)
    state.inspection = None


def run_inspection(animate: bool = True) -> Inspection:
    at = shift_time()
    rows = history[history["datetime"] == at]
    result = inspect(predictor, rows, at, failures, lang=LANG)
    per_row = result.model_ms / max(len(rows), 1)
    for (_i, row), f in zip(rows.sort_values("machineID").iterrows(), result.findings, strict=True):
        state.monitor.record(per_row, f.risk, f.status, row.to_dict())
    if animate:
        progress = st.progress(0.0)
        stage = st.empty()
        for start in range(0, len(result.findings), BATCH):
            group = result.findings[start : start + BATCH]
            done = start + len(group)
            progress.progress(
                done / len(result.findings),
                text=f"Mengecek mesin {group[0].machine_id} sampai {group[-1].machine_id}",
            )
            with stage.container():
                tiles(group, columns=8, compact=True)
            time.sleep(0.45)
        progress.empty()
        stage.empty()
    state.inspections.append(result)
    state.now = result.at.to_pydatetime()  # Riwayat mesin opens at the same hour
    return result


def open_history(machine_id: int, at: pd.Timestamp) -> None:
    state.machine, state.now, state.view = machine_id, at.to_pydatetime(), "Riwayat mesin"
    state.playing = False


def inspection_view() -> None:
    st.markdown("## Inspeksi shift")
    st.markdown(
        "Mesin yang rusak di tengah shift menghentikan produksi sampai teknisi dan suku cadang "
        f"datang. Di awal shift, inspeksi ini mengecek semua {len(machine_ids)} mesin dan "
        "mendaftar mesin yang kemungkinan **rusak dalam 24 jam ke depan**, jadi tim cukup "
        "memeriksa beberapa mesin, bukan semuanya."
    )
    legend()
    day, hour, rand, go = st.columns([1.4, 1, 1.2, 1.6], vertical_alignment="bottom")
    day.date_input(
        "Tanggal shift",
        **bind("shift_day"),
        min_value=first.date(),
        max_value=last.date(),
        format="DD/MM/YYYY",
    )
    hour.selectbox(
        "Jam mulai", list(range(24)), format_func=lambda h: f"{h:02d}:00", **bind("shift_hour")
    )
    rand.button("Pilih shift acak", on_click=random_shift, width="stretch")
    start = go.button("Mulai inspeksi", type="primary", width="stretch")
    st.caption(
        "Data: September sampai Desember 2015 dari dataset Azure Predictive Maintenance, bulan "
        "yang tidak pernah dilihat model saat training. Pengecekan sengaja diperlambat supaya "
        f"bisa diikuti; model sendiri menilai {len(machine_ids)} mesin dalam beberapa milidetik."
    )

    if start:
        state.inspection = run_inspection()
    result: Inspection | None = state.inspection
    if result is None:
        st.info(
            "Pilih shift lalu tekan **Mulai inspeksi**. *Pilih shift acak* memilihkan satu untukmu."
        )
        return
    if result.at != shift_time():
        st.caption(
            f"Menampilkan inspeksi {tanggal(result.at, hari=False)}. Tekan Mulai untuk shift baru."
        )

    s = result.summary()
    st.divider()
    st.markdown(f"### Laporan inspeksi, {tanggal(result.at)}")
    st.markdown(
        f"<div class='pdm-big'>Periksa {s['to_inspect']} dari {s['machines']} mesin di shift "
        f"ini: {s['alert']} bahaya, {s['watch']} waspada.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Cek dengan kunci jawaban** (hanya bisa dilakukan pada data rekaman)")
    lines = []
    if s["broke_within_24h"] == 0:
        lines.append("Tidak ada mesin yang rusak dalam 24 jam setelah inspeksi ini.")
    else:
        lines.append(
            f"Dalam 24 jam setelah inspeksi ini, **{s['broke_within_24h']} mesin benar-benar "
            f"rusak**, dan **{s['caught']} di antaranya ada di daftar ini.**"
        )
    if s["missed"]:
        lines.append(
            "Terlewat: " + ", ".join(f"mesin {m}" for m in s["missed"]) + " rusak tanpa ditandai."
        )
    if s["false_alarms"]:
        lines.append(
            f"{s['false_alarms']} mesin yang ditandai ternyata tidak rusak, jadi pemeriksaannya "
            "tidak diperlukan."
        )
    st.markdown("\n\n".join(lines))
    st.caption(
        "Dicocokkan dengan log kerusakan di dataset. Model tidak pernah melihat log ini; di "
        "pemakaian nyata, pengecekan yang sama baru bisa dilakukan sehari kemudian."
    )

    with st.expander("Berapa nilai daftar ini? Isi biaya di pabrikmu"):
        a, b = st.columns(2)
        breakdown_cost = a.number_input(
            "Biaya satu kerusakan (perbaikan dan produksi yang hilang), Rp",
            min_value=0,
            value=None,
            step=1_000_000,
            placeholder="misalnya 50000000",
            key="cost_breakdown",
        )
        inspection_cost = b.number_input(
            "Biaya memeriksa satu mesin, Rp",
            min_value=0,
            value=None,
            step=100_000,
            placeholder="misalnya 500000",
            key="cost_inspection",
        )
        savings = estimate_savings(s, breakdown_cost, inspection_cost)
        if savings is None:
            st.caption(
                "Kedua angka ini berbeda di tiap pabrik, jadi tidak ada angka yang diasumsikan."
            )
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Kerusakan yang tertangkap", f"Rp {savings['avoided']:,.0f}")
            m2.metric("Biaya pemeriksaan", f"Rp {savings['inspection']:,.0f}")
            m3.metric("Bersih, jika mesin diperbaiki tepat waktu", f"Rp {savings['net']:,.0f}")

    flagged = [f for f in result.findings if f.flagged]
    if flagged:
        st.markdown("#### Mesin yang harus diperiksa")
        flagged.sort(key=lambda f: -f.risk)
        cols = st.columns(4)
        for i, f in enumerate(flagged):
            with cols[i % 4]:
                st.html(machine_tile(f.machine_id, f.status, f.risk, f.signals, f.breakdown))
                st.button(
                    "Lihat riwayat 7 hari",
                    key=f"open-{f.machine_id}",
                    on_click=open_history,
                    args=(f.machine_id, result.at),
                    type="tertiary",
                )

    st.markdown(f"#### Semua {s['machines']} mesin")
    st.html(plant_map(result.findings))
    st.caption(
        "Arahkan kursor ke nomor untuk melihat risikonya. Kotak berwarna adalah mesin di daftar."
    )

    report = result.report(lang=LANG)
    st.markdown("#### Laporan")
    to_check = report[report["Status"] != "Normal"].copy()
    if to_check.empty:
        st.caption("Tidak ada mesin yang perlu diperiksa di shift ini.")
    else:
        to_check["Risiko rusak"] = (to_check["Risiko rusak"] * 100).round()
        st.dataframe(
            to_check,
            hide_index=True,
            width="stretch",
            column_config={
                "Mesin": st.column_config.NumberColumn(width="small", format="%d"),
                "Status": st.column_config.TextColumn(width="small"),
                "Risiko rusak": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%d%%", width="small"
                ),
            },
        )
    st.download_button(
        "Download laporan (CSV, semua mesin)",
        report.to_csv(index=False).encode("utf-8"),
        file_name=f"inspeksi_{result.at:%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )


def history_limit() -> datetime:
    """The last hour Riwayat mesin may show: the hour of the inspection, as in a real plant,
    where nothing after the moment of the check has been recorded yet."""
    at = state.inspection.at if state.inspection is not None else shift_time()
    return pd.Timestamp(at).to_pydatetime()


def back_to_limit() -> None:
    state.now, state.playing = history_limit(), False


def step_time(hours_: int) -> None:
    state.now = min(max(state.now + timedelta(hours=hours_), first), history_limit())
    state.playing = False


def history_view() -> None:
    limit = history_limit()
    if state.now > limit:
        state.now = limit
    if state.playing:
        state.now = min(state.now + timedelta(hours=state.step), limit)
        if state.now >= limit:
            state.playing = False

    st.markdown("## Riwayat mesin")
    st.markdown(
        "Riwayat satu mesin **sampai jam inspeksi**, untuk melihat bagaimana risikonya berubah "
        "dan kenapa mesin itu masuk daftar inspeksi."
    )
    source = "inspeksi terakhir" if state.inspection is not None else "shift yang dipilih"
    st.info(
        f"**Riwayat berhenti di jam {source}: {tanggal(limit)}.** Sama seperti di pabrik: "
        "saat pengecekan, yang bisa dilihat hanya data yang sudah terekam sampai saat itu. "
        "Di setiap jam, model memakai data 24 jam **sebelumnya** dan menghasilkan satu angka, "
        "yaitu risiko rusak dalam 24 jam ke depan. Untuk melihat jam lain, ganti jam "
        "inspeksinya di tab Inspeksi."
    )
    pick, clock = st.columns([1.4, 3], vertical_alignment="bottom")
    pick.selectbox("Mesin", machine_ids, format_func=lambda m: f"Mesin {m}", **bind("machine"))
    clock.markdown(f"<div class='pdm-big'>{tanggal(state.now)}</div>", unsafe_allow_html=True)

    back, play, forward, speed, home = st.columns(
        [0.8, 1, 0.8, 1.3, 2.4], vertical_alignment="center"
    )
    back.button(
        "◀ 1 jam",
        on_click=step_time,
        args=(-1,),
        width="stretch",
        help="Mundur satu jam di riwayat",
    )
    at_limit = state.now >= limit
    if play.button(
        "Jeda" if state.playing else "Putar",
        width="stretch",
        type="primary",
        disabled=at_limit and not state.playing,
        help="Putar riwayat maju sampai jam inspeksi",
    ):
        state.playing = not state.playing
        st.rerun()
    forward.button(
        "1 jam ▶",
        on_click=step_time,
        args=(1,),
        width="stretch",
        disabled=at_limit,
        help="Maju satu jam, paling jauh sampai jam inspeksi",
    )
    speed.selectbox(
        "Kecepatan",
        [1, 3, 6, 12],
        **bind("step"),
        label_visibility="collapsed",
        format_func=lambda h: f"{h} jam per detik",
    )
    home.button(
        "Kembali ke jam inspeksi", on_click=back_to_limit, width="stretch", disabled=at_limit
    )
    st.slider(
        "Waktu riwayat",
        min_value=first,
        max_value=limit,
        step=timedelta(hours=1),
        **bind("now"),
        format="DD/MM/YYYY, HH:mm",
        label_visibility="collapsed",
    )
    if not at_limit:
        st.caption(
            f"Status di bawah adalah kondisi mesin pada {tanggal(state.now, hari=False)}, "
            f"bukan pada jam inspeksi ({tanggal(limit, hari=False)})."
        )

    machine = state.machine
    row = history[
        (history["machineID"] == machine) & (history["datetime"] == pd.Timestamp(state.now))
    ]
    a = predictor.assess(row, lang=LANG)[0]
    info = machines.loc[machine]
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        st.html(machine_tile(machine, a.status, a.risk, a.signals, ""))
        st.markdown(f"**Yang harus dilakukan:** {a.action}")
        if a.signals:
            st.markdown("**Yang menonjol:**\n" + "\n".join(f"- {s}" for s in a.signals))
        st.caption(
            f"{info['model']}, umur {int(info['age'])} tahun. 'Yang menonjol' adalah bacaan di "
            "luar rentang normal bulan-bulan training, yang paling penting lebih dulu. Itu "
            "petunjuk tempat melihat, bukan diagnosis."
        )
    with right:
        st.markdown("**Risiko rusak, 7 hari terakhir**")
        st.altair_chart(risk_chart(machine, state.now), width="stretch")
        st.caption(
            f"Mesin menjadi kuning (waspada) di atas {predictor.thresholds['watch']:.0%} dan "
            f"merah (bahaya) di atas {predictor.thresholds['alert']:.0%}. Tanda kuning di bawah: "
            "kode error. Tanda abu-abu: penggantian komponen terjadwal. Garis merah tegak: "
            "kerusakan yang sudah terjadi sebelumnya."
        )
    with st.expander("Bacaan sensor, 7 hari terakhir"):
        grid = st.columns(2)
        for i, sensor in enumerate(SENSORS):
            with grid[i % 2]:
                st.altair_chart(sensor_chart(machine, state.now, sensor), width="stretch")
        st.caption(
            "Garis tipis: bacaan per jam. Garis tebal: rata-rata 24 jam, merah saat keluar dari "
            "pita abu-abu, yaitu rentang normal bulan-bulan training (persentil 1 sampai 99)."
        )


def health_view() -> None:
    st.markdown("## Kesehatan sistem")
    st.markdown(
        "Yang dipantau tim setelah model dipakai: apakah masih benar, seberapa cepat dan mahal, "
        "apakah sanggup untuk lebih banyak mesin, dan apakah data yang masuk masih mirip data "
        "training. API melaporkan latency, porsi alert dan drift di `/metrics`."
    )

    st.markdown("#### Akurasi saat dipakai")
    runs = state.inspections
    if not runs:
        st.caption("Jalankan inspeksi dulu; setiap inspeksi dicocokkan dengan log kerusakan.")
    else:
        sums = [r.summary() for r in runs]
        flagged = sum(x["to_inspect"] for x in sums)
        alerts = sum(x["alert"] for x in sums)
        broke = sum(x["broke_within_24h"] for x in sums)
        caught = sum(x["caught"] for x in sums)
        confirmed_alerts = sum(f.status == ALERT and f.broke for r in runs for f in r.findings)
        c = st.columns(4)
        c[0].metric("Inspeksi di sesi ini", len(runs))
        c[1].metric("Kerusakan tertangkap", f"{caught} dari {broke}" if broke else "tidak ada")
        c[2].metric(
            "Status bahaya yang benar",
            f"{confirmed_alerts} dari {alerts}" if alerts else "tidak ada",
        )
        c[3].metric("Mesin ditandai", f"{flagged} dari {sum(x['machines'] for x in sums)}")
        st.caption(
            "Kerusakan dihitung tertangkap jika mesinnya ada di daftar (bahaya atau waspada) dan "
            "benar-benar rusak dalam 24 jam berikutnya. Di seluruh periode test, threshold yang "
            f"dipakai menangkap {card['test_metrics']['alert']['failures_caught']} dari "
            f"{card['test_metrics']['alert']['failures']} kerusakan."
        )

    st.markdown("#### Kecepatan, biaya dan skala")
    summary = state.monitor.summary()
    count = summary.get("predictions", 0)
    c = st.columns(4)
    c[0].metric("Prediksi di sesi ini", f"{count:,}")
    if count:
        c[1].metric("Waktu model per mesin, p50", f"{summary['latency_ms']['p50']:.3f} ms")
        c[2].metric("Waktu model per mesin, p95", f"{summary['latency_ms']['p95']:.3f} ms")
        c[3].metric("Porsi status bahaya", f"{summary['status_share'].get(ALERT, 0):.1%}")
    scale = scaling_test()
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.dataframe(
            scale,
            hide_index=True,
            width="stretch",
            column_config={"Waktu model (ms)": st.column_config.NumberColumn(format="%.1f")},
        )
    with right:
        st.caption(
            "Diukur di komputer yang menjalankan dashboard ini. Lewat API, satu panggilan "
            "/predict butuh sekitar 45 ms, sebagian besar untuk menyusun fitur dari data mentah; "
            "/inspect menyusun fitur semua mesin sekaligus dan menangani 100 mesin dalam sekitar "
            "0,25 detik, dibanding sekitar 4,8 detik untuk 100 panggilan terpisah. File model "
            "0,7 MB dan jalan di satu core CPU, jadi dashboard ini muat di Streamlit Community "
            "Cloud gratis dan API-nya muat di server kecil di pabrik atau gateway edge."
        )

    until = tanggal(state.now, hari=False, jam=False)
    st.markdown(f"#### Drift data masuk, {DRIFT_DAYS} hari sebelum {until}")
    now = pd.Timestamp(state.now)
    seen = history[history["datetime"].between(now - timedelta(days=DRIFT_DAYS), now)]
    enough = now - pd.Timestamp(first) >= timedelta(days=7)
    drift = []
    for feature in DRIFT_FEATURES:
        values = seen[feature].to_numpy()
        score = psi(predictor.reference["deciles"][feature], values) if enough else None
        drift.append(
            {
                "Fitur": feature,
                "PSI": None if score is None else round(score, 3),
                "Tingkat": DRIFT_WORDS[drift_level(score)],
                "Rata-rata": values.mean(),
                "Median training": predictor.reference["quantiles"][feature]["p50"],
            }
        )
    st.dataframe(
        pd.DataFrame(drift),
        hide_index=True,
        width="stretch",
        column_config={
            "Rata-rata": st.column_config.NumberColumn(format="%.2f"),
            "Median training": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "Population Stability Index semua mesin dibanding bulan-bulan training: di bawah 0,1 "
        "stabil, 0,1 sampai 0,25 bergeser, di atas 0,25 berubah. Butuh data minimal seminggu, "
        "karena bacaan per jam dari satu mesin saling berkaitan dan data beberapa hari bisa "
        "naik-turun secara kebetulan. Drift adalah alasan untuk mengecek ulang threshold atau "
        "melatih ulang, bukan bukti model salah. Tanggalnya mengikuti jam di Riwayat mesin."
    )

    if count:
        risks = np.array([r.risk for r in state.monitor.records()])
        st.markdown("#### Sebaran risiko di sesi ini")
        st.altair_chart(
            alt.Chart(pd.DataFrame({"risk": risks}))
            .mark_bar(color=QUIET)
            .encode(
                x=alt.X("risk:Q", bin=alt.Bin(step=0.05), title="Risiko rusak"),
                y=alt.Y("count():Q", title="Jam-mesin"),
            )
            .properties(height=200),
            width="stretch",
        )


def model_view() -> None:
    alert = card["test_metrics"]["alert"]
    st.markdown("## Seberapa akurat?")
    st.markdown(
        "Diuji pada Sep–Des 2015, periode yang tidak pernah dilihat model saat training maupun "
        f"saat memilih threshold. Model memberi status bahaya dalam 24 jam sebelum "
        f"**{alert['failures_caught']} dari {alert['failures']} kerusakan**, peringatan "
        f"pertamanya muncul rata-rata (median) **{alert['median_lead_hours']:.0f} jam** "
        f"sebelumnya, dan menghasilkan **{alert['alerts_per_machine_day']:.2f} status bahaya "
        f"per mesin per hari**, {alert['precision']:.1%} di antaranya benar-benar dalam 24 jam "
        "sebelum kerusakan."
    )
    rows = []
    for item in card["comparison"]:
        caught = (
            f"{item['failures_caught']} dari {item['failures']}"
            if "failures" in item
            else "tidak diukur"
        )
        rows.append(
            {
                "Versi": SETUP_WORDS.get(item["setup"], item["setup"]),
                "PR-AUC": item["pr_auc"],
                "Precision": item["precision"],
                "Recall": item["recall"],
                "Alert per mesin per hari": item["alerts_per_machine_day"],
                "Kerusakan tertangkap": caught,
            }
        )
    ratio = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            c: ratio for c in ("PR-AUC", "Precision", "Recall", "Alert per mesin per hari")
        },
    )
    st.caption(
        "v1 adalah versi pertama project ini: labelnya hanya jam kerusakan itu sendiri dan "
        "datanya dibagi acak, sehingga jam-jam di hari yang sama ada di data train dan test. "
        "Recall 81%-nya datang bersama 99% alarm palsu."
    )

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("#### Fitur yang paling diandalkan model")
        top = pd.DataFrame(
            list(card["importances"].items())[:12], columns=["feature", "importance"]
        )
        st.altair_chart(
            alt.Chart(top)
            .mark_bar(color=QUIET)
            .encode(
                x=alt.X("importance:Q", title="Porsi penurunan impurity"),
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
            f"Dengan sensor saja, PR-AUC turun menjadi {sensors[0]['pr_auc']:.2f}."
            if sensors
            else ""
        )
        st.markdown("#### Batasan hasil ini")
        st.markdown(
            "- Dataset Azure PdM adalah **data simulasi**. Error-nya sengaja dibuat muncul "
            "sebelum kerusakan, karena itu jumlah error jadi fitur terkuat dan skornya hampir "
            "sempurna. Data pabrik asli lebih berantakan; angkanya akan lebih rendah.\n"
            "- 100 mesin, satu tahun, empat jenis komponen. Threshold perlu dipilih ulang dengan "
            "data pabrik sendiri dan biaya kerusakan yang terlewat dibanding pemeriksaan yang "
            "tidak perlu.\n"
            f"- Model butuh log error dan log penggantian komponen. {sensors_only}\n"
            "- 'Yang menonjol' membandingkan bacaan dengan persentil data training; itu bukan "
            "penjelasan sebab-akibat."
        )
        st.markdown("#### Model")
        st.markdown(
            f"{card['algorithm']}, {len(card['features'])} fitur, versi {card['version']}, "
            f"dilatih {card['trained_at']} dengan scikit-learn {card['sklearn']}. Train "
            f"{card['data']['train'][0]} sampai {card['data']['train'][1]}, validasi "
            f"{card['data']['validation'][0]} sampai {card['data']['validation'][1]}, test "
            f"{card['data']['test'][0]} sampai {card['data']['test'][1]}."
        )


st.segmented_control("Tampilan", VIEWS, key="view", required=True, label_visibility="collapsed")
st.write("")
if state.view == VIEWS[0]:
    inspection_view()
elif state.view == VIEWS[1]:
    st.fragment(run_every=1.0 if state.playing else None)(history_view)()
elif state.view == VIEWS[2]:
    model_view()
else:
    health_view()
