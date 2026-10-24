from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from pdm import data

DASHBOARD = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


def _shift_before_a_breakdown():
    failures = data.load_replay().failures
    later = failures[failures["datetime"] >= "2015-09-05"].sort_values("datetime")
    return later["datetime"].iloc[0] - pd.Timedelta(hours=10)


def test_inspection_runs_and_reports_the_coming_breakdown():
    at = _shift_before_a_breakdown()
    app = AppTest.from_file(DASHBOARD, default_timeout=120).run()
    assert not app.exception
    app.date_input(key="w_shift_day").set_value(at.date())
    app.selectbox(key="w_shift_hour").set_value(int(at.hour))
    next(b for b in app.button if b.label == "Mulai inspeksi").click().run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "Laporan inspeksi" in text and "benar-benar rusak" in text and "ada di daftar" in text
    assert app.session_state["inspection"].summary()["caught"] >= 1


def test_other_views_render():
    app = AppTest.from_file(DASHBOARD, default_timeout=120).run()
    for view in ("Riwayat mesin", "Seberapa akurat?", "Kesehatan sistem"):
        app.session_state["view"] = view
        app.run()
        assert not app.exception, view


def test_switching_views_keeps_the_shift_and_history_follows_the_inspection():
    at = _shift_before_a_breakdown()
    app = AppTest.from_file(DASHBOARD, default_timeout=120).run()
    app.date_input(key="w_shift_day").set_value(at.date())
    app.selectbox(key="w_shift_hour").set_value(int(at.hour))
    next(b for b in app.button if b.label == "Mulai inspeksi").click().run()
    app.session_state["view"] = "Riwayat mesin"
    app.run()
    assert pd.Timestamp(app.session_state["now"]) == at
    app.session_state["view"] = "Inspeksi"
    app.run()
    assert app.date_input(key="w_shift_day").value == at.date()
    assert not app.exception


def test_history_never_goes_past_the_inspection_hour():
    at = _shift_before_a_breakdown()
    app = AppTest.from_file(DASHBOARD, default_timeout=120).run()
    app.date_input(key="w_shift_day").set_value(at.date())
    app.selectbox(key="w_shift_hour").set_value(int(at.hour))
    next(b for b in app.button if b.label == "Mulai inspeksi").click().run()
    app.session_state["view"] = "Riwayat mesin"
    app.run()
    slider = app.slider(key="w_now")
    assert pd.Timestamp(slider.max, unit="us") == at  # microseconds since epoch
    forward = next(b for b in app.button if b.label == "1 jam ▶")
    assert forward.disabled
    next(b for b in app.button if b.label == "◀ 1 jam").click().run()
    assert pd.Timestamp(app.session_state["now"]) == at - pd.Timedelta(hours=1)
    assert not app.exception
