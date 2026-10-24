"""A shift inspection: score every machine at one hour and list the ones to check.

This is the batch view of the same model the API serves one machine at a time. A maintenance
planner runs it at the start of a shift; the result is a short list instead of 100 machines.
When the failure log is known (replayed history), each finding is also checked against what
really happened in the next 24 hours.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from pdm.labels import HORIZON_HOURS
from pdm.model import ALERT, OK, WATCH, Predictor


@dataclass
class Finding:
    machine_id: int
    status: str
    risk: float
    action: str
    signals: list[str]
    breakdown: str = ""  # from the failure log, e.g. "comp3 after 7 h"; "" if none in 24 h

    @property
    def flagged(self) -> bool:
        return self.status != OK

    @property
    def broke(self) -> bool:
        return bool(self.breakdown)


@dataclass
class Inspection:
    at: pd.Timestamp
    findings: list[Finding]
    model_ms: float
    checked_against_log: bool = False
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        flagged = [f for f in self.findings if f.flagged]
        broke = [f for f in self.findings if f.broke]
        out = {
            "machines": len(self.findings),
            "alert": sum(f.status == ALERT for f in self.findings),
            "watch": sum(f.status == WATCH for f in self.findings),
            "to_inspect": len(flagged),
            "model_ms": round(self.model_ms, 2),
        }
        if self.checked_against_log:
            out.update(
                broke_within_24h=len(broke),
                caught=sum(f.flagged for f in broke),
                missed=[f.machine_id for f in broke if not f.flagged],
                false_alarms=sum(not f.broke for f in flagged),
            )
        return out

    def report(self, lang: str = "en") -> pd.DataFrame:
        """Flagged machines first (alert, then watch, by risk), one row per machine."""
        order = {ALERT: 0, WATCH: 1, OK: 2}
        rows = sorted(self.findings, key=lambda f: (order[f.status], -f.risk, f.machine_id))
        col = REPORT_COLUMNS[lang]
        frame = pd.DataFrame(
            {
                col["machine"]: [f.machine_id for f in rows],
                col["status"]: [STATUS_WORDS[lang][f.status] for f in rows],
                col["risk"]: [round(f.risk, 4) for f in rows],
                col["signals"]: ["; ".join(f.signals) for f in rows],
                col["action"]: [f.action for f in rows],
            }
        )
        if self.checked_against_log:
            frame[col["log"]] = [f.breakdown or col["no"] for f in rows]
        return frame


STATUS_WORDS = {
    "en": {ALERT: "Alert", WATCH: "Watch", OK: "Normal"},
    "id": {ALERT: "Bahaya", WATCH: "Waspada", OK: "Normal"},
}
REPORT_COLUMNS = {
    "en": {
        "machine": "Machine",
        "status": "Status",
        "risk": "Breakdown risk",
        "signals": "What stands out",
        "action": "Action",
        "log": "Broke down within 24 h (log)",
        "no": "No",
    },
    "id": {
        "machine": "Mesin",
        "status": "Status",
        "risk": "Risiko rusak",
        "signals": "Yang menonjol",
        "action": "Tindakan",
        "log": "Rusak dalam 24 jam (log)",
        "no": "Tidak",
    },
}


def breakdown_within(
    failures: pd.DataFrame, machine_id: int, at: pd.Timestamp, lang: str = "en"
) -> str:
    """Components that failed on this machine in the 24 h after ``at``, from the failure log."""
    window = failures[
        (failures["machineID"] == machine_id)
        & (failures["datetime"] > at)
        & (failures["datetime"] <= at + pd.Timedelta(hours=HORIZON_HOURS))
    ]
    if window.empty:
        return ""
    first = window["datetime"].min()
    comps = ", ".join(sorted(window.loc[window["datetime"] == first, "failure"].unique()))
    hours = (first - at).total_seconds() / 3600
    return f"{comps} after {hours:.0f} h" if lang == "en" else f"{comps}, {hours:.0f} jam kemudian"


def inspect(
    predictor: Predictor,
    rows: pd.DataFrame,
    at,
    failures: pd.DataFrame | None = None,
    lang: str = "en",
) -> Inspection:
    """Score ``rows`` (one feature row per machine, all at hour ``at``)."""
    at = pd.Timestamp(at)
    start = time.perf_counter()
    assessments = predictor.assess(rows, lang=lang)
    model_ms = (time.perf_counter() - start) * 1000
    findings = []
    for machine_id, a in zip(rows["machineID"].astype(int), assessments, strict=True):
        log = breakdown_within(failures, machine_id, at, lang) if failures is not None else ""
        findings.append(Finding(machine_id, a.status, a.risk, a.action, a.signals, log))
    findings.sort(key=lambda f: f.machine_id)
    return Inspection(at, findings, model_ms, checked_against_log=failures is not None)


def estimate_savings(
    summary: dict, breakdown_cost: float | None, inspection_cost: float | None
) -> dict | None:
    """Cost of the breakdowns this list catches minus the cost of inspecting it.

    Both costs come from the user (they differ per plant); nothing is assumed here.
    """
    if not breakdown_cost or inspection_cost is None or "caught" not in summary:
        return None
    avoided = summary["caught"] * breakdown_cost
    spent = summary["to_inspect"] * inspection_cost
    return {"avoided": avoided, "inspection": spent, "net": avoided - spent}
