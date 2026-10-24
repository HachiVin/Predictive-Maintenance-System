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
    app.date_input(key="shift_day").set_value(at.date())
    app.selectbox(key="shift_hour").set_value(int(at.hour))
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
